import math
import pybullet as p
import numpy as np
import transforms3d
import utilities as util
import random

INITIAL_POS = np.array([0.000, 0.000, 0.43]) 
INITIAL_ORN = np.array([math.radians(0), math.radians(-10), 0]) 

TARGET_POS = np.array([-0.048485, 0.04305, 0])
TARGET_ORN = np.array([0, 0, 0])
HOLE_OFFSET = [0.048485, -0.04305, 0.098]
# HOLE_OFFSET = [0, 0, 0]

URDF_PATH_TOOL = 'envs/urdf/robotless_peg_in_hole/tool'
URDF_PATH_TARGET = 'envs/urdf/robotless_peg_in_hole/insertion_box'


class RobotSimRobotlessPegInHole():

    def __init__(self):

        random_position = False
        random_rotation = False

        if random_position:
            pass

        if random_rotation:
            pass

        self.uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TOOL),
                              basePosition=INITIAL_POS,
                              baseOrientation=util.xyzw_by_euler(INITIAL_ORN, 'sxyz'),
                              useFixedBase=0)
        self.link_member = 2
        self.link_gripper = 1
        self.link_sensor = 0

        self.target_uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TARGET),
                                     basePosition=TARGET_POS,
                                     baseOrientation=util.xyzw_by_euler(TARGET_ORN, 'sxyz'),
                                     useFixedBase=1)
        self.link_target = 0
        # util.display_frame_axis(self.target_uid, self.link_target)

        self.max_force = -1
        
        # set friction
        """
        p.changeDynamics(self.uid,
                         self.link_member,
                         lateralFriction=1.0)
        p.changeDynamics(self.target_uid,
                         1,
                         lateralFriction=1.0)
        """

        self.base_constraint = p.createConstraint(
            parentBodyUniqueId=self.uid,
            parentLinkIndex=-1,  # base index
            childBodyUniqueId=-1,  # base index
            childLinkIndex=-1,  # base index
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=INITIAL_POS,
            childFrameOrientation=INITIAL_ORN
        )

        self.base_pose = p.getBasePositionAndOrientation(self.uid)
        self.gripper_pose = p.getLinkState(self.uid, self.link_gripper)[4:6]
        self.member_pose = p.getLinkState(self.uid, self.link_member)[4:6]

        self.from_base_to_member = util.get_f1_to_f2_xform(self.base_pose, self.member_pose)

        # for link_index in range(p.getNumJoints(self.uid)):
        #     util.display_frame_axis(self.uid, link_index)
        util.display_frame_axis(self.uid, self.link_member)

    def get_member_pose(self):
        # base_pose = p.getBasePositionAndOrientation(self.uid)
        # member_pose_mat = util.transform_mat(self.from_base_to_member,
        #                                      util.mat44_by_pos_quat(base_pose[0], base_pose[1]))
        # member_pose = util.mat44_to_pos_quat(member_pose_mat)
        member_pose = p.getLinkState(self.uid, self.link_member)
        return [member_pose[0], member_pose[1]]

    # DEBUG
    def set_member_pose(self, pose):
        p.changeConstraint(self.base_constraint, pose[0], pose[1], self.max_force)

    def get_target_pose(self):
        pose = p.getLinkState(self.target_uid, self.link_target)[4:6]
        # Offset the hole center from corner
        return [(np.array(pose[0]) + HOLE_OFFSET).tolist(), pose[1]]

    def enable_force_torque_sensor(self):
        # FT sensor is measured at the center of mass in Pybullet
        p.enableJointForceTorqueSensor(self.uid, self.link_sensor)

    def get_force_torque(self):
        # FT reading in pybullet is negated and needs to be scaled down
        return np.multiply(-0.1, p.getJointState(self.uid, self.link_sensor)[2]).tolist()

    def apply_action_pose(self, delta_pose):
        # pybullet oddity:
        # mysterious 5-time relationship between commanded velocity and resulting velocity, when using the changeConstraint() method
        delta_pose = np.multiply(np.array(delta_pose), 5.0)

        relative_pos = np.array(delta_pose[0:3])
        base_pos, base_orn = p.getBasePositionAndOrientation(self.uid)
        new_pos = np.add(np.array(base_pos), relative_pos).tolist()

        ang_vel_quat = [0, delta_pose[3], delta_pose[4], delta_pose[5]]
        new_orn = np.add(base_orn, np.multiply(0.5,
                  util.wxyz_to_xyzw(transforms3d.quaternions.qmult(ang_vel_quat, util.xyzw_to_wxyz(base_orn)))))

        p.changeConstraint(self.base_constraint, new_pos, new_orn, self.max_force)

    def apply_action_position(self, delta_pos):
        # pybullet oddity:
        # mysterious 5-time relationship between commanded velocity and resulting velocity, when using the changeConstraint() method
        delta_pos = np.multiply(np.array(delta_pos), 5.0)

        base_pos, base_orn = p.getBasePositionAndOrientation(self.uid)
        new_pos = np.add(np.array(base_pos), delta_pos).tolist()

        p.changeConstraint(self.base_constraint, new_pos, INITIAL_ORN, self.max_force)
    
    # [YUNING] add simple interface for sampling
    def get_base_pose(self):
        base_pose = p.getBasePositionAndOrientation(self.uid)
        return [base_pose[0], base_pose[1]]
    
    def set_base_pose(self, pose):
        p.resetBasePositionAndOrientation(self.uid, pose[0], pose[1])
