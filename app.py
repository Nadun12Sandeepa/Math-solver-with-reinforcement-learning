import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image
import pytesseract
from dqn.env import MathEnv
from dqn.agent import DQNAgent
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
from agents.problem_generator import generate_50_problems
import re

TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

# ---------------------------
# Clean LLM-generated or OCR problems
# ---------------------------
def clean_problem(problem):
    problem = problem.replace(" ", " ")  # non-breaking spaces
    problem = problem.replace("\xa0", " ")
    problem = problem.strip()
    # Remove leading numbering like "1. "
    problem = re.sub(r"^\d+\.\s*", "", problem)
    return problem

# ---------------------------
# OCR helper
# ---------------------------
def ocr_image(file_path):
    img = Image.open(file_path)
    text = pytesseract.image_to_string(img)
    return text.strip()

# ---------------------------
# Solve problem with DQN + SymPy
# ---------------------------
def solve_problem(problem, agent=None, episodes=50):
    try:
        problem = clean_problem(problem)
        env = MathEnv(problem)
        if agent is None:
            agent = DQNAgent()
        state = env.encode_state()
        steps_taken = []

        # RL steps
        for _ in range(episodes):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            agent.replay()
            state = next_state
            steps_taken.append(str(env.eq))
            if done:
                break

        # SymPy solution
        if "=" not in problem:
            return f"{problem}\nError: Equation must contain '='"

        left, right = problem.split("=")
        left_expr = parse_expr(left.replace(" ", ""), transformations=TRANSFORMATIONS)
        right_expr = parse_expr(right.replace(" ", ""), transformations=TRANSFORMATIONS)
        eq = sp.Eq(left_expr, right_expr)
        solution = sp.solve(eq, sp.symbols("x"))

        solution_text = f"Problem: {problem}\n"
        solution_text += "\n".join(steps_taken)
        solution_text += f"\nReward: {reward}"
        solution_text += f"\nFinal Solution: x = {solution[0]}" if solution else "\nFinal Solution: Could not solve"
        solution_text += "\n" + "-"*70
        return solution_text

    except Exception as e:
        return f"{problem}\nError: {e}\n" + "-"*70

# ---------------------------
# GUI callbacks
# ---------------------------
def open_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
    if file_path:
        text = ocr_image(file_path)
        entry_problem.delete(0, tk.END)
        entry_problem.insert(0, text)

def solve():
    base_problem = entry_problem.get()
    if not base_problem:
        messagebox.showerror("Error", "Please enter or load a math problem.")
        return

    text_solution.config(state=tk.NORMAL)
    text_solution.delete(1.0, tk.END)
    text_solution.insert(tk.END, "🧮 Generating 50 similar problems...\n\n")
    text_solution.update()

    # Generate 50 problems
    problems = generate_50_problems(base_problem)[:50]

    agent = DQNAgent()
    all_rewards = []

    for idx, problem in enumerate(problems):
        solution = solve_problem(problem, agent=agent)
        text_solution.insert(tk.END, f"{solution}\n")
        text_solution.update()

        # Track reward
        if "Reward:" in solution:
            try:
                reward = int(solution.split("Reward:")[1].split("\n")[0])
                all_rewards.append(reward)
            except:
                pass

    avg_reward = sum(all_rewards)/len(all_rewards) if all_rewards else 0
    text_solution.insert(tk.END, f"\n🏆 Average reward over 50 problems: {avg_reward:.2f}\n")
    text_solution.insert(tk.END, "\n📘 Learned Strategy:\n"
                                 "1. Expand brackets\n"
                                 "2. Combine like terms\n"
                                 "3. Move variable terms to one side\n"
                                 "4. Move constants to the other side\n"
                                 "5. Divide by the coefficient\n"
                                 "6. Verify the result\n")
    text_solution.insert(tk.END, f"\n✅ Final Solution for original problem:\n{solve_problem(base_problem, agent=agent)}\n")
    text_solution.config(state=tk.DISABLED)

# ---------------------------
# GUI Layout
# ---------------------------
root = tk.Tk()
root.title("Agentic Math Solver (DQN + SymPy + OCR)")
root.geometry("950x750")

# Input frame
frame_input = tk.Frame(root)
frame_input.pack(pady=10)

entry_problem = tk.Entry(frame_input, width=60, font=("Arial", 14))
entry_problem.pack(side=tk.LEFT, padx=5)

btn_load_image = tk.Button(frame_input, text="Load Image (OCR)", command=open_image)
btn_load_image.pack(side=tk.LEFT, padx=5)

btn_solve = tk.Button(root, text="Solve 50 Problems", command=solve, bg="#4CAF50", fg="white", font=("Arial", 12))
btn_solve.pack(pady=10)

# Solution display (scrollable)
solution_label = tk.Label(root, text="Solutions:", font=("Arial", 12, "bold"))
solution_label.pack()

text_solution = scrolledtext.ScrolledText(root, width=115, height=35, font=("Courier", 12), state=tk.DISABLED)
text_solution.pack(pady=5)

root.mainloop()
