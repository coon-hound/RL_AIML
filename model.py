import gym
from gym import spaces
import pybullet as p
import pybullet_data
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

class RoboticArmEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, render_mode=p.DIRECT):
        super(RoboticArmEnv, self).__init__()
        self.physicsClient = p.connect(render_mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])
        num_joints = p.getNumJoints(self.robotId)
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(num_joints,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        p.resetSimulation(self.physicsClient)
        p.setGravity(0, 0, -10)
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])
        return self.get_observation()

    def step(self, action):
        p.stepSimulation(self.physicsClient)
        observation = self.get_observation()
        reward = -np.linalg.norm(observation)
        done = np.linalg.norm(observation) < 0.1
        return observation, reward, done, {}

    def get_observation(self):
        state = p.getBasePositionAndOrientation(self.robotId)
        return np.array(state[0])

    def close(self):
        p.disconnect(self.physicsClient)

    def render(self, mode='human'):
        pass

def make_env():
    def _init():
        return RoboticArmEnv(render_mode=p.DIRECT)
    return _init

if __name__ == "__main__":
    num_envs = 4
    envs = SubprocVecEnv([make_env() for _ in range(num_envs)])
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=50000)
    model.save("ppo_kuka_multi")
    envs.close()