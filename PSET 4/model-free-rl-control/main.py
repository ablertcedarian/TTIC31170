import gymnasium as gym
import numpy as np
import torch
import random
import argparse
import os
from qlearning import Policy as QLearningPolicy
from policy_gradient import Policy as PolicyGradientPolicy
import matplotlib.pyplot as plt

def train(env_name, num_episodes, algorithm):
    if env_name == 'CartPole-v1':
        env = gym.make(env_name, sutton_barto_reward=True)
        # env = gym.make(env_name, sutton_barto_reward=False)
    else:
        env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm == "qlearning":
        policy = QLearningPolicy(state_dim, action_dim)
    elif algorithm == "policy_gradient":
        policy = PolicyGradientPolicy(state_dim, action_dim)

    rewards = []
    lengths = []
    max_positions = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        max_pos = state[0]
        done = False
        truncated = False

        while not (done or truncated):
            action = policy.sample_action(state)

            next_state, reward, done, truncated, _ = env.step(action)
            # reward_shaped = policy.shape_reward(state, action, reward, next_state, done)

            if env_name == "MountainCar-v0":
                max_pos = max(max_pos, next_state[0])

            policy(state, action, reward, next_state, done or truncated)
            # policy(state, action, reward_shaped, next_state, done or truncated)

            state = next_state
            # episode_reward += reward_shaped
            episode_reward += reward
            episode_length += 1
            # print(reward)
        rewards.append(episode_reward)
        lengths.append(episode_length)

        if env_name == "MountainCar-v0":
            max_positions.append(max_pos)
            print(f"Episode {episode+1}: max-x = {max_pos: .3f}")

        if (episode + 1) % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            avg_length = sum(lengths[-10:]) / 10
            # print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.2f}")
            if env_name == "MountainCar-v0":
                avg_max = sum(max_positions[-10:]) / 10
                print(
                    f"Episode {episode+1}/{num_episodes}, "
                    f"Avg Reward: {avg_reward: .2f}, "
                    f"Avg Length: {avg_length: .2f}, "
                    f"Avg max-x: {avg_max: .3f}"
                )
            else:
                print(
                    f"Episode {episode+1}/{num_episodes}, "
                    f"Avg Reward: {avg_reward: .2f}, "
                    f"Avg Length: {avg_length: .2f}"
                )

        if env_name == 'CartPole-v1' and len(rewards) >= 50:
            avg_length = sum(lengths[-50:]) / 50
            if avg_length >= 400:
                print(f"\nEnvironment solved in {episode+1} episodes!")
                break

    env.close()
    return policy, rewards, lengths

def eval_policy(policy, env_name, num_episodes=5):
    env = gym.make(env_name, render_mode="human")

    for i in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = truncated = False

        while not (done or truncated):
            action = policy.sample_action(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        print(f"Test Episode {i+1}: Reward = {total_reward}")

    env.close()

def load_policy(env_name, model_path, algorithm="qlearning"):
    """Load a pre-trained policy"""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    if algorithm == "qlearning":
        policy = QLearningPolicy(state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=0.01)
        policy.q_network.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Q-network loaded from {model_path}")
    elif algorithm == "policy_gradient":
        policy = PolicyGradientPolicy(state_dim, action_dim, lr=0.001, gamma=0.99)
        policy.policy_net.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"Policy loaded from {model_path}")

    env.close()
    return policy

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1',
                      choices=['CartPole-v1', 'MountainCar-v0'],
                      help='Environment (CartPole-v1 or MountainCar-v0)')
    parser.add_argument('--algorithm', type=str, default='qlearning',
                      choices=['qlearning', 'policy_gradient'],
                      help='RL algorithm to use')
    parser.add_argument('--episodes', type=int, default=20000,
                      help='Number of episodes to train')
    parser.add_argument('--eval', default=False,
                      help='Skip training and only test a saved model')
    parser.add_argument('--model-path', type=str, default='default.pt',
                      help='Path to saved model (for testing only)')
    args = parser.parse_args()

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    if args.eval:
        policy = load_policy(args.env, args.model_path, args.algorithm)
        print(f"Testing {args.algorithm} policy on {args.env}...")
        eval_policy(policy, args.env)
    else:
        print(f"Training {args.algorithm} on {args.env}...")
        policy, rewards, lengths = train(args.env, num_episodes=args.episodes, algorithm=args.algorithm)
        if args.algorithm == "qlearning":
            torch.save(policy.q_network.state_dict(), args.model_path)
        else:
            torch.save(policy.policy_net.state_dict(), args.model_path)
        print(f"Policy saved to {args.model_path}")
        # Plot reward curve
        plt.figure()
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title(f'Reward Curve - {args.algorithm} on {args.env}')
        plt.savefig('reward_curve.png')
        plt.close()
        print("Reward curve saved to reward_curve.png")
        print(f"\nTesting policy on {args.env}...")
        eval_policy(policy, args.env)

