import numpy as np
from gymnasium import Wrapper

class RewardShapingWrapper(Wrapper):
    def __init__(self, env):
        super(RewardShapingWrapper, self).__init__(env)
        self.previous_ram = None

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.previous_ram = self.get_ram()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_ram = self.get_ram()
        
        # Shape the reward
        reward_shaped = self.shape_reward(reward, current_ram)
        
        self.previous_ram = current_ram
        return obs, reward_shaped, done, info

    def get_ram(self):
        """Unwrap and access the underlying ALE RAM."""
        return self.env.envs[0].unwrapped.ale.getRAM()

    def shape_reward(self, reward, ram):
        # Example: Reward shaping logic based on RAM
        brick_rewards = {
            0: 5,  # Uppermost row
            1: 4,
            2: 3,
            3: 2,
            4: 1   # Bottommost row
        }

        if reward > 0:
            row_hit = self.detect_row_hit(ram)  # Implement row detection
            reward = brick_rewards.get(row_hit, 1)
        
        return reward

    def detect_row_hit(self, ram):
        # RAM index for the ball's vertical position
        ball_y = ram[131]  # 0x83 in hexadecimal
        
        # Map ball_y to rows (tune thresholds based on environment)
        if ball_y < 32:
            return 0
        elif ball_y < 64:
            return 1
        elif ball_y < 96:
            return 2
        elif ball_y < 128:
            return 3
        else:
            return 4
