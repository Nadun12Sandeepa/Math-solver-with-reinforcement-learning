from utils.groq_client import llm

def propose_steps(problem):
    prompt = f"""
Solve the equation step by step.
Return ONLY the steps, one per line.

Problem:
{problem}
"""
    return llm(prompt).split("\n")
