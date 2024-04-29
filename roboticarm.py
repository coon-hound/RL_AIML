import pybullet as p
import pybullet_data
import time
import numpy as np
import random

class RoboticArmEnv:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])
        p.setGravity(0, 0, -10)

        # Initialize the endEffectorIndex to the last link
        # This is a common assumption, check your specific URDF structure
        self.endEffectorIndex = p.getNumJoints(self.robotId) - 1

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0])

        self.target_position = np.random.uniform(-0.5, 0.5, size=3)
        self.target_position[2] = 0.5
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_shape_id, basePosition=self.target_position)
        
        for joint in range(p.getNumJoints(self.robotId)):
            p.resetJointState(self.robotId, joint, 0)
        # p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0])
        return self.get_observation()

    def get_observation(self):
        state = p.getLinkState(self.robotId, self.endEffectorIndex)
        if state is not None:
            end_effector_pos = state[0]
            return end_effector_pos
        return None

    def step(self, action):
        for joint in range(p.getNumJoints(self.robotId)):
            p.setJointMotorControl2(self.robotId, joint, p.POSITION_CONTROL, targetPosition=action[joint])
        p.stepSimulation()
        time.sleep(1./240.)
        observation = self.get_observation()
        if observation is None:
            print("Failed to get a valid observation")
            return None, 0, True, {}  # Handle the failure case
        reward = -np.linalg.norm(np.array(observation) - np.array(self.target_position))  # More meaningful reward
        done = np.linalg.norm(np.array(observation) - np.array(self.target_position)) < 0.1
        return observation, reward, done, {}

    def close(self):
        p.disconnect()

def random_policy(env):
    return [random.uniform(-3.14, 3.14) for _ in range(p.getNumJoints(env.robotId))]

env = RoboticArmEnv()
for episode in range(5):
    observation = env.reset()
    for step in range(50):
        action = random_policy(env)
        observation, reward, done, _ = env.step(action)
        if done:
            print("Reached target!")
            break
env.close()