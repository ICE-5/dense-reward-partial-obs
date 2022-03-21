import math
from envs.robot_sim_robotless_lap_joint import RobotSimRobotlessLapJoint
# from envs.robot_sim_robotless_peg_in_hole import RobotSimRobotlessPegInHole

t = 'sim' 

if t == 'sim':
    from envs.task_sim import TaskSim
    def env_creator(env_config):
        environment = TaskSim(env_robot=RobotSimRobotlessLapJoint,  # choose the sim robot class -> RobotSimRobotlessPegInHole or RobotSimRobotlessLapJoint 
                              self_collision_enabled=True,  # collision setting for pybullet
                              
                              # check it between run and train mode
                              renders=False,  # normally for running sim and rolling out in sim, this is set to True; for training, False.
                              
                              ft_noise=False,  # domain randomization on force/torque observation
                              pose_noise=False,  # domain randomization on pose observation
                              action_noise=False,  # domain randomization on actions
                              physical_noise=False,  # domain randomization on physical parameters
                              time_step=1./250.,  # sets the control frequency of the robot
                              
                              max_steps= 998, #998,  # max number of steps in each episode
                                               # 998 for lap-joint, 3998 for peg-in-hole 

                              # check it between run and train mode
                              step_limit=True,  # limit the length of an episode by max_steps?
                              
                              action_dim=6,  # dimension of action space

                              # 0.02 for lap-joint; 0.04 for peg-in-hole
                              max_vel=0.02, # max linear velocity (m/s) along each axis, 
                              max_rad=0.02, # max rotational velocity (rad/s) around each axis,
                              
                              ft_obs_only=True,  # only use force/torque as observation?
                              
                              limit_ft=False,  # limit force/torque based on max_ft?
                              
                              max_ft= [2500, 2500, 4000, 400, 400, 400], # [1000, 1000, 2500, 100, 100, 100],  # max force (N) and torque (Nm)
                              max_position_range= [2]*3, # max observation space for positions (m)
                              
                              dist_threshold=0.001,  # an episode is considered successful when distance is within the threshold.
                                                    # 0.005 for lap-joint, 0.01 for peg-in-hole, 0.015 for testing a straight-down policy 
                              
                              orn_dist_factor=0.05 # 0.05 for lap-joint and 0 for peg-in-hole
                              )

        return environment


if t == 'pyatk':
    from envs.pyatk_env import ATKEnv
    def env_creator(env_config):

        # initial pose for lap-joint
        # panda:[600,0,237,0,0,0]
        # ur10: [700,-200,370,0,0,0]
        # kuka: [2530,0,877,0,0,0]

        # initial pose for peg-in-hole
        # panda: [0, 0, 170, 0, 0, 0]
        # ur10: [0, 0, 170+325, 0, 0, math.radians(-33.16)]
        # kuka: [200, 0, 170+1121, 0, 0, 0]

        # vertical eval
        # initial_pose = [600-37,0,237-37+600,0,math.radians(-90),0]# x-y-z-rx-ry-rz
        # initial_pose = [-170+200, 0, 200+500, 0, math.radians(-90), 0]# x-y-z-rx-ry-rz
        
        initial_pose = [2530,0,877,0,0,0]
        # initial_pose = [2400,0,877,0,0,0]
        # initial_pose = [0, 0, 170, 0, 0, 0]

        environment = ATKEnv(robot = "kuka",
                             task = "lap-joint",
                             render=True,
                             max_steps=2999,
                             max_vel=0.02, # 0.04 for peg-in-hole, 0.02 for lap-joint
                             max_rad=0.02, # 0.04 for peg-in-hole, 0.02 for lap-joint
                             dist_threshold=0.002,
                             time_step=1./250.,
                             initial_pose=initial_pose,
                             tool_friction=1.0,
                             target_friction=1.0,
                             debug_mode=False)
        return environment 

if t == 'robosuite':
    import robosuite as suite
    def env_creater(env_config):
        pass