import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class Kuka:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 3
    self.fingerBForce = 3.5
    self.fingerTipForce = 3
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.kukaEndEffectorIndex = 6
    self.kukaGripperIndex = 8 # lbr_iiwa_with_wsg50__gripper_to_arm
    # #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):
    objects = p.loadURDF(os.path.join(self.urdfRootPath, "ur3/urdf/ur3_with_gripper4.urdf"), [0.000000, -0.150000, 0.100000],
                                                                                           [0.000000, 0.000000, 0.000000, 1.000000] )
    self.kukaUid = objects

    self.jointPositions = [
        0.000000, 0.006418, -0.813184, 0.911401, -0.789317, 0.005379, 0.837684, 0.000000, 
        -0.006539, 0.000048, 0.000000, -0.299912, 0.000000, 0.000000, 0.299960
    ]
    self.numJoints = p.getNumJoints(self.kukaUid)
    for jointIndex in range(self.numJoints):
      p.resetJointState(self.kukaUid, jointIndex, self.jointPositions[jointIndex])
      p.setJointMotorControl2(self.kukaUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
                              0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
    self.endEffectorPos = [0.537, 0.0, 0.5]
    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.kukaUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.kukaUid, self.kukaGripperIndex)
    pos = state[0]
    orn = state[1]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands):

    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]

      state = p.getLinkState(self.kukaUid, self.kukaEndEffectorIndex)
      actualEndEffectorPos = state[0]

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      #prevent the robot from working beyond its workplace
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50

      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.05):
        self.endEffectorPos[1] = -0.05
      if (self.endEffectorPos[1] > 0.05):
        self.endEffectorPos[1] = 0.05

      self.endEffectorPos[2] = self.endEffectorPos[2] + dz

      self.endEffectorAngle = self.endEffectorAngle + da

      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos, orn)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.kukaUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.kukaUid, self.kukaEndEffectorIndex, pos)

      if (self.useSimulation):
        for i in range(self.kukaEndEffectorIndex + 1):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.kukaUid, i, jointPoses[i])
      #fingers
      p.setJointMotorControl2(self.kukaUid,
                              8,
                              p.POSITION_CONTROL,
                              targetPosition=self.endEffectorAngle,
                              force=self.maxForce)
      p.setJointMotorControl2(self.kukaUid,
                              9,
                              p.POSITION_CONTROL,
                              targetPosition=-fingerAngle,
                              force=self.fingerAForce)
      p.setJointMotorControl2(self.kukaUid,
                              12,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

      p.setJointMotorControl2(self.kukaUid,
                              11,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)
      p.setJointMotorControl2(self.kukaUid,
                              14,
                              p.POSITION_CONTROL,
                              targetPosition=0,
                              force=self.fingerTipForce)

    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kukaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)