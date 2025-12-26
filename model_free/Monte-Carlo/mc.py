import numpy as np 
import gymnasium as gym
import matplotlib.pyplot as plt  

env = gym.make("FrozenLake-v1", is_slippery=True)

def sample_trajectory(pi, env, max_steps=50, epsilon=0.1):
    done = False
    trajectory = []
    num_steps = 0

    state, _ = env.reset()

    while not done:
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = pi[state]

        next_state, reward, done, _, _ = env.step(action)
        experience = (state, int(action), reward, next_state, done)
        trajectory.append(experience)

        num_steps += 1

        if num_steps >= max_steps:
            done = True
            break 

        state = next_state

    return trajectory

# policy = np.random.choice(env.action_space.n, size=(env.observation_space.n))
# trajectory = sample_trajectory(policy, env)  
# print(trajectory)


def compute_returns(trajectory, gamma=0.99):
    returns = {}
    G = 0
    for t in reversed(trajectory):
        state, action, reward, _, _ = t
        G = reward + gamma * G
        if (state, action) not in returns:
            returns[(state, action)] = G 
    return returns

# print(compute_returns(trajectory))


def monte_carlo_estimation(pi, env, gamma=0.99, max_steps=50, num_episodes=5000):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    returns = {(s, a): [] for s in range(env.observation_space.n) for a in range(env.action_space.n)}

    for _ in range(num_episodes):
        trajectory = sample_trajectory(pi, env, max_steps)
        returns_for_episode = compute_returns(trajectory)
        for (state, action), G in returns_for_episode.items():
            returns[(state, action)].append(G)

    for (state, action), returns_list in returns.items():
        if len(returns_list) > 0:
            Q[state, action] = np.mean(returns_list)

    return Q

# mc1 = monte_carlo_estimation(policy, env) 
# print(mc1)

def policy_improvement(Q):
    return np.argmax(Q, axis=-1) 


def mc_iter(env, gamma=0.99, max_steps=50, num_episodes=10000):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n, ))
    while True:
        Q = monte_carlo_estimation(policy, env, gamma, max_steps, num_episodes)
        new_policy = policy_improvement(Q) 

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, Q


# opt_policy, opt_Q = mc_iter(env)
# print(opt_policy)

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

# test_pol(opt_policy, env)


def online_mc_estimate(pi, env, gamma=0.99, max_steps=50, num_episodes=5000):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    N = np.zeros((env.observation_space.n, env.action_space.n)) # no of steps a state, action pair is visited
    for _ in range(num_episodes):
        trajectory = sample_trajectory(pi, env, max_steps)
        returns = compute_returns(trajectory, gamma)

        for (state, action), G in returns.items():
            Q[state, action] = Q[state, action] + (G - Q[state, action]) / (N[state, action] + 1)
            N[state, action] += 1 
        
    return Q

def oneline_mc_iter(env, gamma=0.99, max_steps=50, num_episodes=10000):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n, ))
    while True:
        Q = online_mc_estimate(policy, env, gamma, max_steps, num_episodes)
        new_policy = policy_improvement(Q) 

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, Q


# opt_p, opt_q = oneline_mc_iter(env) 
# print(opt_p)
# test_pol(opt_p, env)


def lr_scheduler(start_val, min_val, decay_factor, num_episodes):
    alphas = [start_val * decay_factor**episode for episode in range(num_episodes)]
    alphas = [a if a >= min_val else min_val for a in alphas]
    return alphas

alphas = lr_scheduler(1.0, 0.01, 0.99, 5000)

def online_mc_estimate_w_lr(pi, env, gamma=0.99, max_steps=50, num_episodes=5000, lr_start_val=0.8, lr_min_val=0.01, lr_decay_factor=0.99):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    alphas = lr_scheduler(lr_start_val, lr_min_val, lr_decay_factor, num_episodes)
    for i in range(num_episodes):
        trajectory = sample_trajectory(pi, env, max_steps)
        returns = compute_returns(trajectory, gamma)

        for (state, action), G in returns.items():
            Q[state, action] = Q[state, action] + alphas[i] * (G - Q[state, action])
        
    return Q

def oneline_mc_iter_w_lr(env, gamma=0.99, max_steps=50, num_episodes=10000):
    policy = np.random.choice(env.action_space.n, size=(env.observation_space.n, ))
    while True:
        Q = online_mc_estimate_w_lr(policy, env, gamma, max_steps, num_episodes)
        new_policy = policy_improvement(Q) 

        if np.array_equal(new_policy, policy):
            break
        policy = new_policy
    return policy, Q

opt_p, opt_q = oneline_mc_iter(env)
print("online monte carlo with lr")
print(opt_p)
test_pol(opt_p, env)

