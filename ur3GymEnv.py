import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
print("current_dir=" + currentdir)
os.sys.path.insert(0, currentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import ur3
import random
import pybullet_data
from pkg_resources import parse_version

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class Ur3GymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=1000):
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._maxSteps = maxSteps
    self.terminated = 0
    self._cam_dist = 1.3
    self._cam_yaw = 180
    self._cam_pitch = -40

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high)
    self.observation_space = spaces.Box(-observation_high, observation_high)
    self.viewer = None

  def reset(self):
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.55 + 0.12 * random.random()
    ypos = 0 + 0.2 * random.random()
    ang = 3.14 * 0.5 + 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.15,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._kuka = ur3.Ur3(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._kuka.getObservation()
    gripperState = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)

    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.005
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    else:
      dv = 0.005
      dx = action[0] * dv
      dy = action[1] * dv
      da = action[2] * 0.05
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._kuka.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter += 1
    if self._renders:
      time.sleep(self._timeStep)
    self._observation = self.getExtendedObservation()

    done = self._termination()
    npaction = np.array([
        action[3]
    ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction) * 10.
    reward = self._reward() - actionCost

    return np.array(self._observation), reward, done, {}

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._kuka.kukaUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
    actualEndEffectorPos = state[0]

    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("terminating, closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          break
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000

    numPt = len(closestPoints)
    if (numPt > 0):
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      reward = reward + 10000
      print("successfully grasped a block!!!")
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step