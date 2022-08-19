import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from ur3_diverse_object_gym_env import Ur3DiverseObjectEnv


# env = gym.make('KukaDiverseObjectGrasping-v0', renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20)
env = Ur3DiverseObjectEnv(renders=False, isDiscrete=False, removeHeightHack=False, maxSteps=20)

model = PPO.load("models/chpl2tku/model")

mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=30, render=False)
env.close()
print("Mean reward: {0}, Std reward: {1}".format(mean_reward,  std_reward
), end="\n")
# print("Mean reward: {0}, Std reward: {1}".format(mean_reward, mean_reward, std_reward
# ), end="\n")