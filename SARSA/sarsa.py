import numpy as np  
import gymnasium as gym


def epsilon_greedy(Q, state, epsilon, env):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(Q[state])

def sarsa_(env, num_episodes=25000, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    for episode in range(num_episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon, env)
        done = False
        while not done:
            nexst_state, reward, done, _, _ = env.step(action)
            next_action = epsilon_greedy(Q, nexst_state, epsilon, env) 
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * Q[nexst_state, next_action] - Q[state, action]) 
            state, action = nexst_state, next_action
    return Q 



def test_pol(policy, env, num_episodes=500):
    sc = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = policy[state]
            state, reward, done, _, _ = env.step(action)
            if done and reward == 1.0:
                sc += 1
    print(sc / num_episodes)

env = gym.make("FrozenLake-v1", is_slippery=True)
Q_s = sarsa_(env) 
policy = np.argmax(Q_s, axis=-1)
print(policy)

test_pol(policy, env)

