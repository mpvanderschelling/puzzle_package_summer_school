import random
from collections import defaultdict

import numpy as np
import torch

from agent import PolicyGradientAgent
from environment import EnvParams, EnvState, PackingGameEnv


class VanillaReinforceTrainer:
    """
    Vanilla REINFORCE algorithm.
    There is no value function.
    Perhaps the simplest policy gradient algorithm.
    """

    def __init__(self,
                 env: PackingGameEnv,
                 eval_env: PackingGameEnv,
                 agent: torch.nn.Module,
                 optimizer: str = "Adam",
                 optimizer_kwargs: dict = {'lr': 0.001},
                 device: str = "cpu"):
        self.env = env
        self.eval_env = eval_env
        self.agent = agent
        try:
            self.optimizer = getattr(torch.optim, optimizer)(
                self.agent.parameters(), **optimizer_kwargs)
        except Exception:
            raise ValueError(f"Optimizer {optimizer} not found")
        self.device = device

    def interact(self, num_steps: int):
        """
        Interact with the environment and store the experience in the exp_buffer.
        Interacts for num_steps steps, where each episode is n_pieces steps.
        """
        episode_num = 0
        self.exp_buffer = defaultdict(defaultdict)
        state = self.env.reset()
        for step in range(num_steps):
            action, logprob = self.agent.act(state)
            next_state, reward, done = self.env.step(action)
            self.exp_buffer[episode_num]['obs'].append(state)
            self.exp_buffer[episode_num]['action'].append(action)
            self.exp_buffer[episode_num]['logprob'].append(logprob)
            self.exp_buffer[episode_num]['reward'].append(reward)
            state = next_state
            if done:
                state = self.env.reset()
                episode_num += 1

    def get_rewards_to_go(self):
        """
        Compute the undiscounted sum of future rewards.
        """
        for episode_num in self.exp_buffer.keys():
            rewards = self.exp_buffer[episode_num]['reward']
            rewards_to_go = []
            for i in range(len(rewards)):
                # no discount factor, horizon = num_pieces
                rewards_to_go.append(sum(rewards[i:]))
            self.exp_buffer[episode_num]['rewards_to_go'] = rewards_to_go

    def policy_update(self):
        """
        All of the exp_buffer will only cause one update.
        """
        self.optimizer.zero_grad()
        for episode_num in self.exp_buffer.keys():
            logprobs = self.exp_buffer[episode_num]['logprob']
            rewards_to_go = self.exp_buffer[episode_num]['rewards_to_go']
            loss = -torch.sum(logprobs * rewards_to_go)
            loss.backward()
        self.optimizer.step()

    def train(self, num_episodes=1000, max_steps=1000, eval_interval=10):
        """
        Args:
            num_episodes: int, number of episodes to train
            max_steps: int, number of steps per episode
            eval_interval: int, number of episodes between evaluations
        """
        for episode in range(num_episodes):
            self.interact(max_steps)
            self.get_rewards_to_go()
            self.policy_update()
            if episode % eval_interval == 0:
                eval_reward = self.policy_eval()
                print(f"Episode {episode}, eval reward: {eval_reward}")

    def policy_eval(self, num_episodes=10):
        total_reward = []
        for episode in range(num_episodes):
            state = self.eval_env.reset()
            while True:
                action, logprob = self.agent.act(state)
                next_state, reward, done = self.eval_env.step(action)
                state = next_state
                total_reward.append(reward)
                if done:
                    break
        return torch.mean(total_reward)


if __name__ == "__main__":
    env = PackingGameEnv(EnvParams(board_size=10, n_pieces=10))
    eval_env = PackingGameEnv(EnvParams(board_size=10, n_pieces=10))
    agent = PolicyGradientAgent(board_size=10, n_pieces=10, channel_num_list=[
                                32, 32, 32], kernel_size_list=[3, 3, 3])
    trainer = VanillaReinforceTrainer(env, eval_env, agent)
    trainer.train(num_episodes=1000, max_steps=1000, eval_interval=10)
