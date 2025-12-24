from groq import Groq

client = Groq()

def llm(prompt):
    return client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="medium",
    ).choices[0].message.content
