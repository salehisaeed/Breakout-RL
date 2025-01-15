import os
import torch
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from utils.lr_scheduler import linear_schedule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vec_env = make_atari_env("BreakoutNoFrameskip-v4", n_envs=4, seed=0)
vec_env = VecFrameStack(vec_env, n_stack=4)

checkpoint_dir = "./checkpoints/PPO"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".zip")]
if checkpoints:
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[-2]))
    model_path = os.path.join(checkpoint_dir, latest_checkpoint)
    model = PPO.load(model_path, env=vec_env, device=device)
    print(f"Loaded model from {latest_checkpoint}")
else:
    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=linear_schedule(2.5e-4, 2.5e-5),
        n_steps=128,
        batch_size=256,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
    )
    print("No checkpoint found. Created a new model.")

checkpoint_callback = CheckpointCallback(
    save_freq=50_000, save_path=checkpoint_dir, name_prefix="ppo_breakout"
)

model.learn(total_timesteps=40_000_000, callback=checkpoint_callback)
