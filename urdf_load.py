import pybullet as p
import os

cid = p.connect(p.GUI)
robot_id = p.loadSDF(os.path.join(os.path.dirname(__file__), "kuka_gym/pybullet_data/kuka_iiwa/kuka_with_gripper2.sdf"))
robot_id = 0
for number_joint in range(p.getNumJoints(robot_id)):
    print(p.getJointInfo(robot_id, number_joint))

while(True):
    pass