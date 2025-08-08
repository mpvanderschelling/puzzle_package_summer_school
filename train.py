from collections import defaultdict

import torch

from gymnax.experimental.pytorch_rollout import BatchedTrainerCompatibleWrapper

from agent import PolicyGradientAgent
import jax.random as jr


class VanillaReinforceTrainer:
    """
    Vanilla REINFORCE algorithm.
    There is no value function.
    Perhaps the simplest policy gradient algorithm.
    """

    def __init__(self,
                 env: BatchedTrainerCompatibleWrapper,
                 eval_env: BatchedTrainerCompatibleWrapper,
                 agent: torch.nn.Module,
                 optimizer: str = "Adam",
                 optimizer_kwargs: dict = {'lr': 0.001},
                 jr_key: int = 37,
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

        # Proper seed generation using JAX random to avoid collisions
        master_key = jr.key(jr_key)
        self.train_key, self.eval_key = jr.split(master_key)
        self.episode_count = 0
        
        # Store n_pieces for episode length management
        self.n_pieces = env.max_steps

    def interact(self, num_steps: int):
        """
        Interact with the environment and store the experience in the exp_buffer.
        Buffer structure: exp_buffer[episode_num] contains tensors of shape (steps, batch_size, ...)
        """
        episode_num = 0
        self.exp_buffer = defaultdict(lambda: defaultdict(list))
        
        # Generate episode key and reset
        episode_key = jr.fold_in(self.train_key, self.episode_count)
        obs, state = self.env.reset(episode_key)
        batch_size = obs.shape[0]
        
        for step in range(num_steps):
            action, logprob = self.agent.act(obs)
            next_obs, next_state, reward, done, info = self.env.step(state, action)
            
            # Store experience as lists (will convert to tensors later)
            self.exp_buffer[episode_num]['obs'].append(obs)     # (batch_size, ...)
            self.exp_buffer[episode_num]['action'].append(action) # (batch_size,)
            self.exp_buffer[episode_num]['logprob'].append(logprob) # (batch_size,)
            self.exp_buffer[episode_num]['reward'].append(reward) # (batch_size,)
            
            state = next_state
            obs = next_obs

            # Handle batched done flags - all envs finish simultaneously for PuzzlePacking
            if done.any():  # In batched case, all will be True or all False
                self.episode_count += 1
                episode_key = jr.fold_in(self.train_key, self.episode_count)
                obs, state = self.env.reset(episode_key)
                episode_num += 1

    def get_rewards_to_go(self):
        """
        Compute the undiscounted sum of future rewards.
        Input: exp_buffer[episode]['reward'] is list of (batch_size,) tensors
        Output: exp_buffer[episode]['rewards_to_go'] is (steps, batch_size) tensor
        """
        for episode_num in self.exp_buffer.keys():
            # Stack rewards: list of (batch_size,) -> (steps, batch_size)
            rewards = torch.stack(self.exp_buffer[episode_num]['reward'])
            steps, batch_size = rewards.shape
            
            # Compute rewards-to-go for each step and batch element
            rewards_to_go = torch.zeros_like(rewards)
            for t in range(steps):
                rewards_to_go[t] = rewards[t:].sum(dim=0)  # Sum future rewards
                
            self.exp_buffer[episode_num]['rewards_to_go'] = rewards_to_go

    def policy_update(self):
        """
        All of the exp_buffer will only cause one update.
        Input: logprobs and rewards_to_go are both (steps, batch_size) tensors
        """
        self.optimizer.zero_grad()
        total_loss = 0
        
        for episode_num in self.exp_buffer.keys():
            # Stack logprobs: list of (batch_size,) -> (steps, batch_size)
            logprobs = torch.stack(self.exp_buffer[episode_num]['logprob']).squeeze(-1)
            rewards_to_go = self.exp_buffer[episode_num]['rewards_to_go']
            
            # REINFORCE loss: -log_prob * reward_to_go, summed over all steps and batch
            episode_loss = -torch.sum(logprobs * rewards_to_go)
            total_loss += episode_loss
        
        if total_loss != 0:  # Only backward if we have loss
            total_loss.backward()
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
        self.agent.eval()  # Set to evaluation mode
        
        with torch.no_grad():  # No gradients during evaluation
            for episode in range(num_episodes):
                episode_reward = 0
                eval_episode_key = jr.fold_in(self.eval_key, episode)
                obs, state = self.eval_env.reset(eval_episode_key)
                
                while True:
                    action, logprob = self.agent.act(obs)
                    
                    next_obs, next_state, reward, done, info = self.eval_env.step(state, action)
                    
                    # Average rewards across batch dimension for evaluation
                    episode_reward += reward.mean().item()
                    
                    state = next_state
                    obs = next_obs
                    
                    if done.any():  # Handle batched done flags
                        break
                        
                total_reward.append(episode_reward)
        
        self.agent.train()  # Set back to training mode
        return torch.tensor(total_reward).mean().item()


if __name__ == "__main__":
    # Environment configuration
    env_config = {
        "batch_size": 128,  # Single environment for trainer compatibility
        "env_name": "PuzzlePacking",
        "device": "cpu",
        "env_kwargs": {"grid_size": 4, "n_pieces": 4},
        "env_params": {"penalty_factor": 0.0}
    }
    
    env = BatchedTrainerCompatibleWrapper(**env_config)
    eval_env = BatchedTrainerCompatibleWrapper(**env_config)
    
    # Create agent
    obs_shape = env.observation_shape
    agent = PolicyGradientAgent(
        board_size=obs_shape[-1], 
        n_pieces=obs_shape[0] - 1, 
        channel_num_list=[16, 32, 64, 32], 
        kernel_size_list=[3]
    )
    
    trainer = VanillaReinforceTrainer(env, eval_env, agent, device="cpu")
    trainer.train(num_episodes=100, max_steps=trainer.n_pieces * 5, eval_interval=5)
