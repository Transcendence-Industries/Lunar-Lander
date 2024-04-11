import random
import numpy as np
import gymnasium as gym
from tqdm import tqdm
from datetime import datetime
from gymnasium.spaces import MultiDiscrete

from lunar_lander.utils.logging import MLflowLogger
from lunar_lander.utils.file_handling import save_pickle

"""
Observation vector (8 dims):
0: x position (-1.5, 1.5)
1: y position (-1.5, 1.5)
2: x velocity (-5, 5)
3: y velocity (-5, 5)
4: angle (-pi, pi)
5: angular velocity (-5, 5)
6: left leg ground contact (0, 1)
7: right leg ground contact (0, 1)

Discrete actions:
0: do nothing
1: fire left engine
2: fire main engine
3: fire right engine
"""


class Agent:
    """Tabular Q-learning agent with observation discretization."""

    def __init__(self, render: bool = False) -> None:
        # Environment setup
        self.env = gym.make(
            "LunarLander-v2",
            render_mode="human" if render else None,
            continuous=False,
            gravity=-10.0,
            enable_wind=False,
            wind_power=15.0,
            turbulence_power=1.5,
        )

        # Observation discretization
        self.discrete_observation_bins = np.array(
            (8, 8, 8, 8, 6, 6, 2, 2), dtype=np.int16)
        self.discrete_observation_window_size = (
            (self.env.observation_space.high - self.env.observation_space.low)
            / self.discrete_observation_bins
        )
        discrete_observation_space = MultiDiscrete(
            self.discrete_observation_bins)
        action_space = self.env.action_space

        # Q-table initialization
        self.q_table = np.zeros(
            (*self.discrete_observation_bins, action_space.n), dtype=np.float32)
        print(f"Observation space: {discrete_observation_space}")
        print(f"Action space: {action_space}")
        print(
            f"Shape/Size of Q-table: {self.q_table.shape} / {self.q_table.size}")

        self.logger = MLflowLogger(experiment="LunarLander")

        print("Agent is ready!")
        print(30 * "_")

    def get_discrete_observation(self, obs: np.ndarray) -> np.ndarray:
        """Clip and bucketize continuous observations into discrete bins."""
        obs = np.clip(obs, self.env.observation_space.low,
                      self.env.observation_space.high)
        discrete_obs = (obs - self.env.observation_space.low) / \
            self.discrete_observation_window_size
        return discrete_obs.astype(np.int16)

    @staticmethod
    def value_in_tolerance(value: float, target: float, tolerance: float) -> bool:
        return target - tolerance <= value <= target + tolerance

    def train(
        self,
        n_episodes: int,
        learning_rate: float,
        gamma: float,
        min_epsilon: float,
        max_epsilon: float,
        epsilon_decay: float,
    ) -> None:
        # Logging setup
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.logger.create_run(run=timestamp)
        self.logger.log_parameters(
            {
                "n_episodes": n_episodes,
                "learning_rate": learning_rate,
                "gamma": gamma,
                "min_epsilon": min_epsilon,
                "max_epsilon": max_epsilon,
                "epsilon_decay": epsilon_decay,
            }
        )
        self.logger.log_metric(key="epsilon", value=max_epsilon, step=0)
        self.logger.log_metric(key="reward", value=0, step=0)
        self.logger.log_metric(key="reward_moving_avg", value=0, step=0)
        rewards = []
        moving_avg_window = 100

        for episode in tqdm(range(1, n_episodes + 1)):
            # Reset environment
            state, _ = self.env.reset()
            state = self.get_discrete_observation(state)

            # Update epsilon for epsilon-greedy exploration
            epsilon = min_epsilon + \
                (max_epsilon - min_epsilon) * np.exp(-epsilon_decay * episode)
            self.logger.log_metric(key="epsilon", value=epsilon, step=episode)
            episode_reward = 0
            end_episode = False

            while not end_episode:
                # Select action (greedy vs random)
                if random.uniform(0, 1) > epsilon:
                    action = np.argmax(self.q_table[state])
                else:
                    action = self.env.action_space.sample()

                # Execute action
                new_state, reward, terminated, truncated, _ = self.env.step(
                    action)
                new_state = self.get_discrete_observation(new_state)
                episode_reward += reward

                # Q-learning update
                self.q_table[state][action] = self.q_table[state][action] + learning_rate * (
                    float(reward) + gamma *
                    np.max(self.q_table[new_state]) -
                    self.q_table[state][action]
                )
                state = new_state

                # Optional early-exit condition for centered landing
                if (self.value_in_tolerance(state[0], 0, 0.1)
                        and self.value_in_tolerance(state[1], 0, 0.1)):
                    print("Goal reached!")
                    end_episode = True

                # Episode termination (env signals)
                if terminated or truncated:
                    end_episode = True

            rewards.append(episode_reward)

            if len(rewards) >= moving_avg_window:
                moving_avg = np.mean(rewards[-moving_avg_window:])
            else:
                moving_avg = np.mean(rewards)

            self.logger.log_metric(
                key="reward", value=episode_reward, step=episode)
            self.logger.log_metric(
                key="reward_moving_avg", value=moving_avg, step=episode)

        # Close environment and save model
        print(30 * "_")
        print("Agent shutdown.")
        self.env.close()
        self.logger.end_run()
        save_pickle(model=self.q_table, run=timestamp)

    def run(self, q_table: np.ndarray, n_steps: int) -> None:
        if q_table.shape != self.q_table.shape:
            raise Exception(
                f"The given Q-table has not the desired shape of {self.q_table.shape}!")

        self.q_table = q_table
        state, _ = self.env.reset()
        state = self.get_discrete_observation(state)

        for _ in range(n_steps):
            action = np.argmax(self.q_table[state])
            new_state, _, _, _, _ = self.env.step(action)
            state = self.get_discrete_observation(new_state)

        self.env.close()
