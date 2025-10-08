import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


env = gym.make("FrozenLake-v1", map_name = "4x4", render_mode="rgb_array", is_slippery=False)
# print(env.observation_space.n)
# print(env.action_space)

action_dict = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}

observation, _ = env.reset()
# print(observation)

def render():
    plt.imshow(env.render())
    plt.axis("off")
    plt.show()

# render()

# action = env.action_space.sample()
# print(action)
# observation, reward, done, truncated, _ = env.step(action)
# print(observation, reward)
# render()

# mdp = env.unwrapped.P
# print(mdp)

# env = gym.make("FrozenLake-v1", map_name = "4x4", render_mode="rgb_array", is_slippery=True)
# print(env.unwrapped.P)

policy = np.random.randint(low=0, high=env.action_space.n, size=(env.observation_space.n))

def print_policy(policy, grid=(4, 4)):
    policy_print = np.empty(grid).astype(str)
    for idx_h in range(grid[0]):
        for idx_w in range(grid[1]):
            index = idx_h * grid[0] + idx_w
            sel_action = action_dict[policy[index]]
            sel_action = sel_action[0]
            policy_print[idx_h, idx_w] = sel_action
    print("current: ")
    print("===============")
    print(policy_print)
# print_policy(policy)

def policy_eval(policy, gamma=0.99, iterations=1000, tol=1e-10):
    V = np.zeros(env.observation_space.n)
    for _ in range(iterations):
        V_k = np.copy(V)
        for s in range(env.observation_space.n):
            action = policy[s]
            probability, s_next, reward, terminal = env.unwrapped.P[s][action][0]
            V[s] = probability * (reward + gamma * V_k[s_next])
        if np.max(np.abs(V - V_k)) < tol:
            break
    return V

# vals = policy_eval(policy)
# print(vals)


def policy_improvement(values, gamma=0.99):
    new_policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        q_sa = []
        for action in range(env.action_space.n):
            probability, s_next, reward, terminal = env.unwrapped.P[s][action][0]
            Q = probability * (reward + gamma * values[s_next])
            q_sa.append(Q)
        best_action = np.argmax(q_sa)
        new_policy[s] = best_action
    return new_policy

# impr = policy_improvement(vals)
# print(impr)


def policy_iter(env, num_iter=1000, gamma=0.99):
    policy = np.random.randint(low=0, high=env.action_space.n, size=(env.observation_space.n))
    for _ in range(num_iter):
        values = policy_eval(policy, gamma=gamma, iterations=num_iter)
        updated_policy = policy_improvement(values, gamma=gamma)
        if np.all(policy == updated_policy):
            break
        policy = updated_policy
    return policy

# learned_policy = policy_iter(env, gamma=0.99)
# print_policy(learned_policy)


env2 = gym.make("FrozenLake-v1", map_name = "4x4", render_mode="rgb_array", is_slippery=True)

def policy_eval2(policy, gamma=0.99, iterations=1000, total=1e-10):
    V = np.zeros(env2.observation_space.n)
    for _ in range(iterations):
        V_k = np.copy(V)
        for s in range(env2.observation_space.n):
            wanted_action = policy[s]
            possible_actions = env2.unwrapped.P[s][wanted_action]
            print(possible_actions)
            break
        break

policy_eval2(policy)




