import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from ur3_diverse_object_gym_env import Ur3DiverseObjectEnv

import wandb
from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 100000,
    "env_name": "Ur3DiverseObjectGrasping-v0",
}

run = wandb.init(
    project="ur3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# Parallel environments
env = Ur3DiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20)
# env = make_vec_env('KukaDiverseObjectGrasping-v0', n_envs=4)

model = PPO(config["policy_type"], env,
            n_steps=128,#原本20
            verbose=1,
            learning_rate=2e-4,
            batch_size=2**9,
            gae_lambda=0.98,
            tensorboard_log=f"runs/{run.id}")

# model.learn(total_timesteps=10000)
# model.save("ppo_ur3")

model.learn(total_timesteps=config["total_timesteps"],
            callback=WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{run.id}",
                verbose=2,
            ),
            )
run.finish()
# # Sync TensorBoard with WandB.
# wandb.init(dir='logs_train', tensorboard=True, sync_tensorboard=True)