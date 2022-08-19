import gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from ur3_grasp_gym.ur3_gym.ur3_diverse_object_gym_env import Ur3DiverseObjectEnv

# env = gym.make('KukaDiverseObjectGrasping-v0', renders=True, isDiscrete=False, removeHeightHack=False, maxSteps=20)
env = Ur3DiverseObjectEnv(renders=True, isDiscrete=False, removeHeightHack=False, maxSteps=20)

model = PPO.load("ppo_ur3")

episode = 20
success = 0
for n_ep in range(episode):
    score = 0
    state = env.reset()
    done = False 
    while not done:
        action, _ = model.predict(observation=state)
        state, reward, done, info = env.step(action=action)
        success += reward
        score += reward
        env.render()
    env.close()
    print("Episode:{0} Scores: {1}".format(n_ep, score))
print("Success rate:{0}".format(success/episode))