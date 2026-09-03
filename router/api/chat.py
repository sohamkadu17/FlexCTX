"""Chat completions endpoint and streaming.

This module implements the OpenAI-compatible `/v1/chat/completions` endpoint,
which is the primary inference interface for SmarterRouter. It handles:
- Request validation (Content-Type, body parsing)
- Prompt sanitization and security checks (prompt injection detection, content moderation)
- Model selection via RouterEngine (or override via query parameter)
- VRAM management (loading/unloading models)
- Response caching (semantic cache)
- Streaming and non-streaming responses
- Tool execution loop for function calling

The endpoint integrates with the routing engine to select the optimal model based on
prompt analysis, model capabilities, benchmarks, and feedback scores. It supports
fallback cascades if the selected model fails and includes comprehensive error
handling and logging.
"""


import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Annotated, Any
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse

from router.backends.base import LLMBackend, supports_unload
from router.config import Settings, settings
from router.logging_config import sanitize_for_logging
from router.schemas import (
    ChatCompletionRequest,
    close_unclosed_code_block,
    sanitize_model_name,
    sanitize_prompt,
    strip_signature,
)
from router.skills import skills_registry
from router.state import (
    _log_error_with_context,
    app_state,
    get_available_models_with_cache,
    get_model_vram_estimate,
    get_model_vram_estimates_batch,
    get_settings,
    rate_limit_request,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _parse_text_tool_call(content: Any, tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """Normalize local models that print a tool call instead of using tool_calls."""
    if not isinstance(content, str):
        return None

    # OpenCode may describe its tools in the prompt instead of sending the
    # formal tools array through an OpenAI-compatible provider.
    opencode_tool_names = {
        "bash",
        "delete",
        "edit",
        "glob",
        "grep",
        "ls",
        "npm",
        "question",
        "read",
        "touch",
        "write",
    }
    tool_names = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict)
    } if tools else opencode_tool_names
    candidates = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    candidates.append(content.strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        json_candidates = [candidate]
        if candidate == content.strip():
            json_candidates = [
                candidate[index:]
                for index, character in enumerate(candidate)
                if character == "{"
            ]
        for json_candidate in json_candidates:
            try:
                parsed, _ = decoder.raw_decode(json_candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            name = parsed.get("name")
            arguments = parsed.get("arguments", parsed.get("parameters"))
            normalized_name = "bash" if name == "npm" and "bash" in tool_names else name
            if normalized_name not in tool_names or not isinstance(arguments, (dict, str)):
                continue
            if isinstance(arguments, str):
                try:
                    decoded_arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    decoded_arguments = arguments
                arguments = decoded_arguments
            if not isinstance(arguments, dict):
                arguments = {"questions": arguments}
            if name == "npm" and "npm" not in tool_names:
                name = normalized_name
                workdir = arguments.get("workdir")
                if isinstance(workdir, str) and re.search(
                    r"(?:/path/to|[A-Za-z]:\\path\\to)", workdir, re.IGNORECASE
                ):
                    arguments.pop("workdir", None)
                arguments = {
                    "command": "npm " + arguments.get("command", ""),
                    **({"workdir": arguments["workdir"]} if "workdir" in arguments else {}),
                }
            return {
                "id": f"call_text_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
    return None


@router.post("/v1/chat/completions")
@router.post("/chat/completions")
@router.post("/v1/responses")
@router.post("/responses")
@router.post("/v1/completions")
@router.post("/completions")
async def chat_completions(
    request: Request,
    config: Annotated[Settings, Depends(get_settings)],
):
    """Handle chat completion and response requests via the OpenAI-compatible API."""
    # Rate limit check for chat endpoint
    await rate_limit_request(request, config, is_admin=False, is_chat=True)

    if not app_state.backend or not app_state.router_engine:
        return JSONResponse(
            {"error": {"message": "Service not ready", "type": "service_unavailable"}},
            status_code=503,
        )

    # Validate Content-Type header
    content_type = request.headers.get("content-type", "")
    if not content_type.startswith("application/json"):
        return JSONResponse(
            {
                "error": {
                    "message": "Content-Type must be application/json",
                    "type": "invalid_request_error",
                }
            },
            status_code=415,
        )

    # Parse and validate request body using Pydantic
    try:
        body = await request.json()
        
        # Universal message normalizer for OpenCode / AI-SDK / Responses API
        raw_msgs = body.get("messages")
        if not raw_msgs and "input" in body:
            inp = body["input"]
            if isinstance(inp, str):
                raw_msgs = [{"role": "user", "content": inp}]
            elif isinstance(inp, list):
                raw_msgs = []
                for item in inp:
                    if isinstance(item, str):
                        raw_msgs.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        role = item.get("role", "user")
                        content = item.get("content")
                        if content is None:
                            content = item.get("text") or item.get("value") or ""
                        if isinstance(content, list):
                            parts = []
                            for p in content:
                                if isinstance(p, str):
                                    parts.append(p)
                                elif isinstance(p, dict):
                                    parts.append(p.get("text") or p.get("value") or "")
                            content = "\n".join(parts)
                        raw_msgs.append({"role": role, "content": str(content) if content is not None else ""})
        elif isinstance(raw_msgs, list):
            normalized_msgs = []
            for item in raw_msgs:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content")
                    if isinstance(content, list):
                        parts = []
                        for p in content:
                            if isinstance(p, str):
                                parts.append(p)
                            elif isinstance(p, dict):
                                parts.append(p.get("text") or p.get("value") or "")
                        content = "\n".join(parts)
                    normalized_msgs.append({"role": role, "content": str(content) if content is not None else ""})
            raw_msgs = normalized_msgs

        if not raw_msgs:
            raw_msgs = [{"role": "user", "content": body.get("prompt", "Hello")}]

        body["messages"] = raw_msgs
        validated_request = ChatCompletionRequest(**body)
    except Exception as e:
        logger.warning(f"Request validation failed: {e}")
        return JSONResponse(
            {"error": {"message": f"Invalid request: {str(e)}", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Extract and sanitize prompt from last message (with safe fallback)
    messages = validated_request.messages
    stream = validated_request.stream

    prompt = ""
    for msg in reversed(messages):
        content_text = sanitize_prompt(msg.content)
        if content_text:
            prompt = content_text
            break
    if not prompt:
        prompt = "Hello"

    # Generate response ID early
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    # Check for model override query parameter
    try:
        model_override = sanitize_model_name(request.query_params.get("model"))
    except ValueError as e:
        return JSONResponse(
            {"error": {"message": str(e), "type": "invalid_request_error"}},
            status_code=400,
        )

    # Track request
    if hasattr(app_state, "total_requests"):
        app_state.total_requests += 1

    # Fetch available models once per request (uses cache)
    try:
        available_models = await get_available_models_with_cache()
        model_names = [m.name for m in available_models]

        # Model selection: check query param first, then body model field, then auto-routing
        explicit_model = model_override
        if not explicit_model and validated_request.model:
            cleaned_body_model = validated_request.model.strip()
            if cleaned_body_model.lower() not in ("smarterrouter/main", "smarterrouter", "auto", "default", "main"):
                explicit_model = cleaned_body_model

        if explicit_model:
            selected_model = None
            for name in model_names:
                if name.lower() == explicit_model.lower():
                    selected_model = name
                    break
                if explicit_model.lower() in name.lower() or name.lower() in explicit_model.lower():
                    selected_model = name
                    break

            if not selected_model:
                selected_model = model_names[0] if model_names else "qwen2.5:3b"

            reasoning = f"User-specified model: {selected_model}"
            confidence = 1.0
            logger.debug(f"Explicit model selected: {selected_model}")
        else:
            # Automatic intelligent routing
            last_content = prompt
            routing_result = await app_state.router_engine.select_model(
                last_content, validated_request
            )
            selected_model = routing_result.selected_model
            reasoning = routing_result.reasoning
            confidence = routing_result.confidence
            logger.debug(f"Routed to: {selected_model}, prompt: {sanitize_for_logging(prompt)}")
    except Exception as e:
        _log_error_with_context("Routing failed", request=request, prompt=prompt, exc=e)
        models = await get_available_models_with_cache()
        if models:
            selected_model = models[0].name
            reasoning = "Fallback to first available model"
            confidence = 0.0
        else:
            return JSONResponse(
                {"error": {"message": "No models available", "type": "internal_error"}},
                status_code=500,
            )

    # Convert Pydantic models back to dicts for backend compatibility
    # and strip signatures from previous assistant messages to prevent stacking
    def clean_message_content(msg):
        content = msg.content
        if isinstance(content, str) and msg.role == "assistant":
            # Remove any previous signatures from assistant messages
            content = strip_signature(content)
        return content

    messages_dict = [{"role": msg.role, "content": clean_message_content(msg)} for msg in messages]
    tools_list = validated_request.tools

    # === SEMANTIC VECTOR MEMORY RECALL ===
    if hasattr(app_state, "memory_manager") and app_state.memory_manager:
        try:
            recalled_memories = await app_state.memory_manager.recall_relevant_context(
                query=prompt, top_k=2, min_similarity=0.26
            )
            if recalled_memories:
                memory_block = app_state.memory_manager.format_memory_injection(recalled_memories)
                if messages_dict and messages_dict[0].get("role") == "system":
                    messages_dict[0]["content"] = str(messages_dict[0]["content"]) + f"\n\n{memory_block}"
                else:
                    messages_dict.insert(0, {"role": "system", "content": f"You are an AI coding assistant.\n\n{memory_block}"})
                logger.info(f"Injected {len(recalled_memories)} recalled semantic memories into context")
        except Exception as e:
            logger.debug(f"Memory recall non-fatal error: {e}")

    # === DYNAMIC CONTEXT COMPRESSION & VALUE GATE ===
    if hasattr(app_state, "compression_pipeline") and app_state.compression_pipeline:
        try:
            comp_result = await app_state.compression_pipeline.process_chat_payload(
                messages=messages_dict,
                tools=tools_list,
                backend=app_state.backend,
                vram_monitor=app_state.vram_monitor,
            )
            messages_dict = comp_result.messages
            if comp_result.tools is not None:
                tools_list = comp_result.tools
        except Exception as e:
            logger.warning(f"Context compression encountered non-fatal error: {e}")

    # Collect additional parameters for backend
    backend_kwargs: dict[str, Any] = {
        "temperature": validated_request.temperature,
        "top_p": validated_request.top_p,
        "n": validated_request.n,
        "max_tokens": validated_request.max_tokens,
        "presence_penalty": validated_request.presence_penalty,
        "frequency_penalty": validated_request.frequency_penalty,
        "logit_bias": validated_request.logit_bias,
        "user": validated_request.user,
        "seed": validated_request.seed,
        "tools": tools_list,
        "tool_choice": validated_request.tool_choice,
        "keep_alive": config.model_keep_alive,
    }
    # Remove None values
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}

    if stream:
        # Load model via VRAM manager if enabled, else fallback to traditional unload
        if app_state.vram_manager:
            vram_gb = get_model_vram_estimate(selected_model)
            await app_state.vram_manager.load_model(selected_model, vram_gb)
        else:
            # Traditional: unload current model if different and not pinned before loading new
            current = app_state.current_loaded_model
            pinned = config.pinned_model
            if current and current != selected_model and current != pinned:
                logger.info(
                    f"VRAM management (streaming): unloading {current} to load {selected_model}"
                )
                if supports_unload(app_state.backend):
                    await app_state.backend.unload_model(current)
                app_state.current_loaded_model = None

        # Update current model state
        app_state.current_loaded_model = selected_model

        # Log the decision now that we're committing to it
        await app_state.router_engine.log_decision(
            prompt, selected_model, confidence, reasoning, response_id
        )

        return StreamingResponse(
            stream_chat(
                app_state.backend,
                selected_model,
                messages_dict,
                reasoning,
                config,
                response_id,
                **backend_kwargs,
            ),
            media_type="text/event-stream",
        )

    # Try generation with retries
    response = None
    last_error = None

    # Get available models for fallback
    try:
        available_models = await get_available_models_with_cache()
        fallback_list = [m.name for m in available_models if m.name != selected_model]
        # Put selected_model first in retry list
        fallback_list = [selected_model] + fallback_list
    except Exception:
        fallback_list = [selected_model]

    # Pre-fetch VRAM estimates for all fallback models to avoid N+1 queries
    vram_estimate_map: dict[str, float] = {}
    if app_state.vram_manager:
        # Use batched VRAM estimates function
        vram_estimate_map = get_model_vram_estimates_batch(fallback_list)

    final_model = selected_model

    # Check response cache before generation (include generation params in key)
    cache_key_prompt = prompt
    if app_state.router_engine and app_state.router_engine.semantic_cache:
        cached_response = await app_state.router_engine.semantic_cache.get_response(
            selected_model, cache_key_prompt, params=backend_kwargs
        )
        if cached_response:
            logger.info(f"Response cache hit for {selected_model}")
            await app_state.router_engine.log_decision(
                prompt, selected_model, confidence, reasoning, response_id
            )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(datetime.now(UTC).timestamp()),
                "model": selected_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": cached_response},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "router": {"reasoning": reasoning + " [cached]"},
            }

    for try_model in fallback_list:
        try:
            # Load model via VRAM manager if enabled, else fallback to traditional unload
            if app_state.vram_manager:
                # Use pre-fetched VRAM estimate
                vram_gb = vram_estimate_map.get(try_model, config.vram_default_estimate_gb)
                await app_state.vram_manager.load_model(try_model, vram_gb)
            else:
                # Traditional: unload current model if different and not pinned before loading new
                current = app_state.current_loaded_model
                pinned = config.pinned_model
                if current and current != try_model and current != pinned:
                    logger.info(f"VRAM management: unloading {current} to load {try_model}")
                    if supports_unload(app_state.backend):
                        await app_state.backend.unload_model(current)
                    app_state.current_loaded_model = None

            # Generate response
            response = await app_state.backend.chat(
                model=try_model, messages=messages_dict, stream=False, **backend_kwargs
            )
            final_model = try_model
            app_state.current_loaded_model = final_model
            logger.info(f"Generation succeeded with model: {final_model}")

            # Track stats
            if hasattr(app_state, "requests_by_model"):
                app_state.requests_by_model[final_model] = (
                    app_state.requests_by_model.get(final_model, 0) + 1
                )

            # If we fell back, update reasoning
            if final_model != selected_model:
                reasoning += f" (Fallback from {selected_model})"

            break
        except Exception as try_error:
            # If we loaded this model via VRAM manager and it's still loaded, unload it to free VRAM
            if app_state.vram_manager and app_state.vram_manager.is_loaded(try_model):
                await app_state.vram_manager.unload_model(try_model)
            last_error = try_error

            # Get VRAM state for error context
            vram_context = ""
            if app_state.vram_manager:
                try:
                    available_vram = app_state.vram_manager.get_available_vram()
                    max_vram = app_state.vram_manager.max_vram
                    vram_context = f" | VRAM: {available_vram:.1f}GB/{max_vram:.1f}GB free"
                except Exception:
                    vram_context = " | VRAM: unknown"

            logger.warning(
                f"Model {try_model} failed, trying next: {try_error} | "
                f"Prompt: {sanitize_for_logging(prompt)[:100]}... | "
                f"Response ID: {response_id}{vram_context}",
                exc_info=True,
            )
            continue

    if response is None:
        _log_error_with_context(
            "All models failed",
            request=request,
            model_name=selected_model,
            prompt=prompt,
            exc=last_error,
        )
        if hasattr(app_state, "total_errors"):
            app_state.total_errors += 1
        return JSONResponse(
            {
                "error": {
                    "message": f"All models failed. Last error: {last_error}",
                    "type": "internal_error",
                }
            },
            status_code=500,
        )

    # Log the initial routing decision
    await app_state.router_engine.log_decision(
        prompt, final_model, confidence, reasoning, response_id
    )

    # === TOOL EXECUTION LOOP ===
    max_tool_calls = 5
    tool_calls_made = 0

    while tool_calls_made < max_tool_calls:
        response_message = response.get("message", {})
        tool_calls = response_message.get("tool_calls")
        if not tool_calls:
            text_tool_call = _parse_text_tool_call(
                response_message.get("content"), tools_list
            )
            if text_tool_call:
                response_message["tool_calls"] = [text_tool_call]
                tool_calls = response_message["tool_calls"]
        if not tool_calls:
            break

        logger.info(f"Model {final_model} requested {len(tool_calls)} tool call(s)")

        # Tools supplied by OpenCode belong to the caller and must be returned
        # as structured tool calls so OpenCode can execute them locally.
        external_tool_calls = [
            tool_call
            for tool_call in tool_calls
            if not skills_registry.get_skill(tool_call.get("function", {}).get("name", ""))
        ]
        if external_tool_calls:
            forwarded_tool_calls = []
            for tool_index, tool_call in enumerate(external_tool_calls):
                function = tool_call.get("function", {})
                arguments = function.get("arguments", {})
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                forwarded_tool_calls.append(
                    {
                        "id": tool_call.get("id", f"call_{response_id}_{tool_index}"),
                        "type": "function",
                        "function": {
                            "name": function.get("name", ""),
                            "arguments": arguments,
                        },
                    }
                )
            return {
                "id": response_id,
                "object": "chat.completion",
                "created": int(datetime.now(UTC).timestamp()),
                "model": final_model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": forwarded_tool_calls,
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": response.get("prompt_eval_count", 0),
                    "completion_tokens": response.get("eval_count", 0),
                    "total_tokens": response.get("prompt_eval_count", 0)
                    + response.get("eval_count", 0),
                },
                "router": {"reasoning": reasoning},
            }

        # Add assistant message with tool calls to history
        messages_dict.append(response["message"])

        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            try:
                tool_args = json.loads(tool_call["function"]["arguments"])
            except json.JSONDecodeError as e:
                _log_error_with_context(
                    f"Failed to parse tool arguments for {tool_name}",
                    request=request,
                    model_name=final_model,
                    prompt=prompt,
                    exc=e,
                )
                continue

            logger.info(f"Executing tool: {tool_name}({tool_args})")
            tool_result = await skills_registry.execute_skill(tool_name, **tool_args)

            messages_dict.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

        tool_calls_made += 1

        # Continue conversation with tool results
        response = await app_state.backend.chat(
            model=final_model,
            messages=messages_dict,
            stream=False,
            **backend_kwargs,
        )

    content = response.get("message", {}).get("content", "")

    if config.signature_enabled:
        signature = config.signature_format.format(model=final_model)
        # Strip any existing signature first, then add our own
        content = strip_signature(content)
        # Close any unclosed fenced code block before appending signature
        content = close_unclosed_code_block(content)
        content += signature

    # Cache the response (without signature) for future requests (include generation params)
    if app_state.router_engine and app_state.router_engine.semantic_cache:
        content_for_cache = strip_signature(content)
        await app_state.router_engine.semantic_cache.set_response(
            final_model, prompt, content_for_cache, params=backend_kwargs
        )

    # Ingest turn into Persistent Vector Conversation Memory
    if hasattr(app_state, "memory_manager") and app_state.memory_manager:
        asyncio.create_task(
            app_state.memory_manager.store_turn(
                content=f"User: {prompt}\nAssistant: {content}",
                role="conversation_turn",
                metadata={"model": final_model},
            )
        )

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": datetime.now(UTC).timestamp(),
        "model": final_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": response.get("prompt_eval_count", 0),
            "completion_tokens": response.get("eval_count", 0),
            "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
        },
    }


async def stream_chat(
    client: LLMBackend,
    model: str,
    messages: list[dict[str, str]],
    reasoning: str,
    config: Settings,
    chunk_id: str,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Stream chat completions using Server-Sent Events (SSE).

    This async generator yields SSE-formatted chunks as the LLM generates tokens.
    It handles the streaming HTTP response for the chat completions endpoint.

    Args:
        client: The LLM backend client to use for generation.
        model: The model name to generate with.
        messages: List of message dictionaries with 'role' and 'content'.
        reasoning: Human-readable explanation of why this model was selected.
        config: Application settings.
        chunk_id: Unique ID for this response (included in each chunk).
        **kwargs: Additional backend-specific parameters (temperature, max_tokens, etc.).

    Yields:
        str: SSE-formatted data lines (e.g., "data: {...}\n\n").

    Errors:
        Any exception during streaming is caught and yields an error chunk.
        The error is also logged with context via `_log_error_with_context`.
    """
    created = datetime.now(UTC).timestamp()

    try:
        stream, latency = await client.chat_streaming(model, messages, **kwargs)

        # Initial chunk with metadata
        initial_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
            ],
            "router": {"reasoning": reasoning},
        }
        yield f"data: {json.dumps(initial_chunk)}\n\n"

        accumulated_content = ""
        saw_tool_calls = False

        async for chunk in stream:
            message = chunk.get("message", {})
            content = message.get("content", "")
            if content:
                accumulated_content += content

                content_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(content_chunk)}\n\n"

            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                saw_tool_calls = True
                tool_call_deltas = []
                for tool_index, tool_call in enumerate(tool_calls):
                    function = tool_call.get("function", {})
                    arguments = function.get("arguments", "")
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    tool_call_deltas.append(
                        {
                            "index": tool_index,
                            "id": tool_call.get("id", f"call_{chunk_id}_{tool_index}"),
                            "type": "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
                tool_call_chunk = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"tool_calls": tool_call_deltas},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(tool_call_chunk)}\n\n"

            if chunk.get("done", False):
                text_tool_call = None
                if not saw_tool_calls:
                    text_tool_call = _parse_text_tool_call(
                        accumulated_content, kwargs.get("tools")
                    )
                if text_tool_call:
                    saw_tool_calls = True
                    tool_call_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [text_tool_call]},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(tool_call_chunk)}\n\n"

                # Use schemas.py function to handle code blocks properly
                closed_content = close_unclosed_code_block(accumulated_content)

                # If content was modified (fence added or removed), emit the difference
                if closed_content != accumulated_content:
                    # Find what was added (closing fence or removal)
                    diff = (
                        closed_content[len(accumulated_content) :]
                        if closed_content.startswith(accumulated_content)
                        else ""
                    )

                    # If we need to add a closing fence (not just remove stray)
                    if diff.strip():
                        fence_chunk = {
                            "id": chunk_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": diff},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(fence_chunk)}\n\n"

                # Add signature if enabled
                if config.signature_enabled:
                    signature = config.signature_format.format(model=model)
                    signature_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": signature},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(signature_chunk)}\n\n"
                else:
                    done_chunk = {
                        "id": chunk_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if saw_tool_calls else "stop",
                            }
                        ],
                    }
                    yield f"data: {json.dumps(done_chunk)}\n\n"

                # Ingest stream into Persistent Vector Conversation Memory
                if hasattr(app_state, "memory_manager") and app_state.memory_manager and accumulated_content:
                    last_user_prompt = ""
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            last_user_prompt = str(msg.get("content", ""))
                            break
                    asyncio.create_task(
                        app_state.memory_manager.store_turn(
                            content=f"User: {last_user_prompt}\nAssistant: {accumulated_content}",
                            role="conversation_turn",
                            metadata={"model": model},
                        )
                    )
    except Exception as e:
        prompt_for_hash = None
        if messages:
            last_msg = messages[-1]
            prompt_for_hash = str(last_msg.get("content", ""))
        _log_error_with_context(
            "Streaming failed",
            model_name=model,
            prompt=prompt_for_hash,
            exc=e,
            exc_info=True,
        )
        error_message = str(e)

        # Provide more helpful error messages for common issues
        if "timeout" in error_message.lower():
            error_message = f"Timeout error: The model took too long to respond. Current timeout: {config.generation_timeout}s. Try increasing ROUTER_GENERATION_TIMEOUT."
        elif "connection" in error_message.lower():
            error_message = "Connection error: Could not connect to the LLM backend. Please check that Ollama is running and accessible."

        error_data = {"error": {"message": error_message, "type": "internal_error"}}
        yield f"data: {json.dumps(error_data)}\n\n"

    yield "data: [DONE]\n\n"
