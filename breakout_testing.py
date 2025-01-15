import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = "./checkpoints/PPO/ppo_breakout_31600000_steps" #154
model = PPO.load(checkpoint_path, device=device)

eval_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=1, seed=0)
eval_env.metadata['render_fps'] = 5
eval_env = VecFrameStack(eval_env, n_stack=4)

num_episodes = 5
total_reward = 0
for episode in range(num_episodes):
    obs = eval_env.reset()
    episode_reward = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        episode_reward += reward.item()
        eval_env.render("human")

        if reward.item() > 0:
            print(reward.item())
        # time.sleep(0.01)  # Add a delay between frames (for smoother playback)
        if done:
            break
    print(f"Episode {episode + 1}: Reward = {episode_reward}")
    total_reward += episode_reward

print(f"Total reward: {total_reward}")
