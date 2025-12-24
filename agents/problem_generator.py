from utils.groq_client import llm

def generate_50_problems(problem):
    prompt = f"""
Generate 50 math problems similar to this equation.
Keep same structure, change only numbers.
Return one problem per line.

Problem:
{problem}
"""
    text = llm(prompt)
    return [p.strip() for p in text.split("\n") if p.strip()]
