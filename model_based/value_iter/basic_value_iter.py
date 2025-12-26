import numpy as np  
import gymnasium as gym  


env = gym.make("FrozenLake-v1", map_name = "4x4", render_mode="rgb_array", is_slippery=True)

action_dict = {
    0: "Left",
    1: "Down",
    2: "Right",
    3: "Up"
}

def print_policy(policy, grid=(4, 4)):
    pp = np.empty(grid).astype(str)
    for idx_h in range(grid[0]):
        for idx_w in range(grid[1]):
            ix = idx_h * grid[0] + idx_w
            sa = action_dict[policy[ix]]
            sa = sa[0]
            pp[idx_h, idx_w] = sa
    print("current policy:")
    print(pp)


def value_iter(env, gamma=0.99, num_iter=1000, tol=1e-5):
    V = np.zeros(env.observation_space.n) 
    for _ in range(num_iter):
        V_k = np.copy(V)
        for s in range(env.observation_space.n):
            Q_s = []
            for wanted_action in range(env.action_space.n):
                possible_taken_actions = env.unwrapped.P[s][wanted_action]
                Q_sa = 0
                for prob, s_next, reward, done in possible_taken_actions:
                    Q_sa += prob * (reward + gamma * V_k[s_next])
                Q_s.append(Q_sa)
            V[s] = np.max(Q_s)
        if np.max(np.abs(V - V_k)) < tol:
            break
    return V 

optimal_values = value_iter(env) 
print(optimal_values.reshape(4, 4))


def policy_improvement(values, gamma=0.99):
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        Q_s = []
        for wanted_action in range(env.action_space.n):
            possible_taken_actions = env.unwrapped.P[s][wanted_action]
            Q_sa = 0
            for prob, s_next, reward, terminal in possible_taken_actions:
                Q_sa += prob * (reward + gamma * values[s_next])
            Q_s.append(Q_sa)
        best_action = np.argmax(Q_s)
        policy[s] = best_action
    return policy

optimal_policy = policy_improvement(optimal_values)
print_policy(optimal_policy)

num_games = 1000
max_steps = 500
game_suc = 0
for _ in range(num_games):
    obs, _ = env.reset()
    for _ in range(max_steps):
        action = int(optimal_policy[obs])
        obs, reward, done, _, _ = env.step(action)
        if done:
            if reward > 0:
                game_suc += 1

print("success ratio", game_suc / num_games)

