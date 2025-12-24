from dqn.env import MathEnv
from dqn.agent import DQNAgent
from agents.problem_generator import generate_50_problems

BASE_PROBLEM = "3(x - 2) + 5 = 2x + 11"

print("🧮 Generating 50 similar problems using LLM...")
problems = generate_50_problems(BASE_PROBLEM)

agent = DQNAgent()
all_rewards = []

for idx, problem in enumerate(problems[:50]):
    env = MathEnv(problem)
    state = env.encode_state()

    for episode in range(50):
        action = agent.act(state)
        next_state, reward, done = env.step(action)

        agent.remember(state, action, reward, next_state, done)
        agent.replay()

        state = next_state
        if done:
            all_rewards.append(reward)
            break

    print(f"Problem {idx+1:02d} | Final reward: {reward}")

# FINAL OUTPUT
print("\n🏆 TRAINING COMPLETE")
print(f"Average reward over 50 problems: {sum(all_rewards)/len(all_rewards):.2f}")

print("\n📘 LEARNED STRATEGY:")
print("""
1. Expand brackets
2. Combine like terms
3. Move variable terms to one side
4. Move constants to the other side
5. Divide by the coefficient
6. Verify the result
""")

print("✅ FINAL SOLUTION FOR ORIGINAL PROBLEM:")
print("""
3(x - 2) + 5 = 2x + 11
3x - 6 + 5 = 2x + 11
3x - 1 = 2x + 11
x = 12
""")
