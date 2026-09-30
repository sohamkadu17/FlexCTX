from openai import OpenAI

# Point base_url to SmarterRouter instead of api.openai.com
client = OpenAI(base_url="http://localhost:11436/v1", api_key="dummy-key")

response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "list the last 3 questions I asked you  "}]
)
print(response.choices[0].message.content)
