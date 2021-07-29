import pyatk
from pyatk import Transform, Vector
import time
import gym
from gym import spaces
import numpy as np
import math
from pathlib import Path
from datetime import datetime

# import utilities as util
import random


class ATKEnv(gym.Env):
    def __init__(
        self,
        robot=None,
        task=None,
        render=None,
        max_steps=None,
        max_vel=None,
        max_rad=None,
        dist_threshold=None,
        time_step=None,
        initial_pose=None,
        tool_friction=None,
        target_friction=None,
        debug_mode=False,
    ):

        self._max_steps = max_steps
        self._max_vel = max_vel
        self._max_rad = max_rad
        self._time_step = time_step
        self._debug_mode = debug_mode
        self._dist_threshold = dist_threshold
        self._initial_pose = initial_pose

        # initialize pyatk based on the robot and the task
        gravity = pyatk.Vector(0, 0, 0)
        pyatk.init(visualization=render)

        current_dir = Path(__file__).resolve().parent
        pyatk_dir = str(current_dir / "pyatk_robots/Sample")
        pyatk.set_project_dir(pyatk_dir)

        if robot == "panda" and task == "peg-in-hole":
            w = pyatk.load_workcell("PandaRL", gravity=gravity)
            self.p = w.add_part(
                "peg", "peg0", Transform(500, 0, 250, 0, 0, 0)
            )  # why do we need a pose here?
            target = w.get_peripheral("insertion_box-0")
            target.set_collision_model(True, True)
            self.g = w.get_gripper("PdDefault-0")
        elif robot == "panda" and task == "lap-joint":
            w = pyatk.load_workcell("PandaRL-LapJoint", gravity=gravity)
            self.p = w.add_part(
                "panda_lap_0mm", "panda_lap_0mm0", Transform(500, 0, 250, 0, 0, 0)
            )
            target = w.get_peripheral("task_lap_90deg_2mm-0")
            target.set_collision_model(True, True)  # collision, cancave
            self.g = w.get_gripper("PdDefault-0")
        elif robot == "ur10" and task == "peg-in-hole":
            w = pyatk.load_workcell("UR10-PegInHole", gravity=gravity)
            self.p = w.add_part(
                "UR10_peg", "UR10_peg0", Transform(500, 0, 250, 0, 0, 0)
            )
            target = w.get_peripheral("insertion_box-0")
            target.set_collision_model(True, True)  # collision, cancave
            self.g = w.get_gripper("RqThin-0")
        elif robot == "ur10" and task == "lap-joint":
            w = pyatk.load_workcell("UR10-LapJoint", gravity=gravity)
            self.p = w.add_part(
                "ur10_lap_0mm", "ur10_lap_0mm0", Transform(500, 0, 250, 0, 0, 0)
            )
            target = w.get_peripheral("task_lap_90deg_2mm-0")
            target.set_collision_model(True, True)  # collision, cancave
            self.g = w.get_gripper("RqThin-0")
        elif robot == "kuka" and task == "lap-joint":
            w = pyatk.load_workcell("Kuka-Lap-Joint", gravity=gravity)
            self.p = w.add_part(
                "kuka_lap_0mm",
                "kuka_lap_0mm0",
                Transform(2550, 0, 1182, 0, math.radians(33), math.radians(90)),
            )
            target = w.get_peripheral("task_lap_90deg_2mm-0")
            target.set_collision_model(True, True)  # collision, cancave
            self.g = w.get_gripper("SchunkDual-0")
        elif robot == "kuka" and task == "peg-in-hole":
            w = pyatk.load_workcell("Kuka-Peg-in_Hole", gravity=gravity)
            self.p = w.add_part("peg", "peg0", Transform(500, 0, 250, 0, 0, 0))
            target = w.get_peripheral("insertion_box-0")
            target.set_collision_model(True, True)  # collision, cancave
            self.g = w.get_gripper("PdDefault-0")
        # elif robot == "robotless" and task == "lap-joint":
        #     w = pyatk.load_workcell("robotless-lap-joint", gravity=gravity)
        #     self.p = w.add_part("kuka_lap_0mm", "kuka_lap_0mm0", Transform(0,0,0,0,math.radians(33),math.radians(90)))
        #     target = w.get_peripheral("task_lap_90deg_2mm-0")
        #     target.set_collision_model(True, True) # collision, cancave
        #     self.g = w.get_gripper("SchunkDual-0")
        # elif robot == "robotless" and task == "peg-in-hole":
        #     w = pyatk.load_workcell("robotless-peg-in-hole", gravity=gravity)
        #     self.p = w.add_part("peg", "peg0", Transform(0,0,0,0,math.radians(33),math.radians(90)))
        #     target = w.get_peripheral("insertion_box-0")
        #     target.set_collision_model(True, True) # collision, cancave
        #     self.g = w.get_gripper("PdDefault-0")
        else:
            print("invalid robot name {} or task {}".format(robot, task))
            exit()

        if task == "peg-in-hole":
            self._hole_offset = [48.485, -43.05, 98]
        else:
            self._hole_offset = [0, 0, 0]

        # set frictions
        self.p.set_friction(tool_friction)
        target.set_friction(target_friction)

        # define action and observation space
        action_bound = 1
        action_dim = 6
        action_high = np.array([action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        observation_high = np.array([2500, 2500, 4000, 400, 400, 400])
        self.observation_space = spaces.Box(-observation_high, observation_high)

        self.current_step = 0

        self.target_pose = target.get_pose()

    def reset(self, initial_pose=None):
        self.g.attach(self.p, False)

        if initial_pose is None:
            # introduce noise in initial pose
            self.initial_pose = Transform(
                self._initial_pose[0] + random.randint(-5, 5),
                self._initial_pose[1] + random.randint(-5, 5),
                self._initial_pose[2],
                self._initial_pose[3],
                self._initial_pose[4],
                self._initial_pose[5],
            )
        else:
            self.initial_pose = Transform(
                initial_pose[0][0],
                initial_pose[0][1],
                initial_pose[0][2],
                initial_pose[1][0],
                initial_pose[1][1],
                initial_pose[1][2],
            )

        self.p.set_pose(self.initial_pose, 0, self.g)
        self.g.attach(self.p, True)

        self.p.create_vel_controlled_tool(
            True, workcell_orientation=True, force_threshold=2500.0
        )

        if self._debug_mode:
            print("The target's global pose is: \n{}\n".format(self.target_pose))
            print("The tool's initial global pose is: \n{}\n".format(self.p.get_pose()))
            # self._ft_reading_file_name = 'ft_reading-' + str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S")) + '.csv'
            # util.write_csv(["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], self._ft_reading_file_name, True)

        # assuming this is the initial state
        raw_states = self.p.set_vel(Vector(0, 0, 0), Vector(0, 0, 0), self._time_step)
        states = self.parse_states(raw_states)

        # reset step count
        self.current_step = 0

        return states

    def step(self, action):
        for i in range(len(action)):
            if i < 3:
                action[i] = action[i] * self._max_vel * 1000 * 5
            else:
                action[i] = action[i] * self._max_rad

        raw_states = self.p.set_vel(
            Vector(action[0], action[1], action[2]),
            Vector(action[3], action[4], action[5]),
            self._time_step,
        )
        states = self.parse_states(raw_states)

        dist = self.dist_to_target(self.p.get_pose(), self.target_pose)
        # print(dist)
        done = False
        reward = -dist
        self.current_step += 1

        # print(dist)

        if dist < self._dist_threshold:
            reward += 100
            done = True
            self.p.create_vel_controlled_tool(False)
            print("finish the assembly task at step {}".format(self.current_step))
            time.sleep(2)

        # print("The tool's current global pose is: \n{}\n".format(self.p.get_pose()))

        if self.current_step == self._max_steps:
            done = True
            self.p.create_vel_controlled_tool(False)
            # print("reach the maximum steps")

        if raw_states[0] != "valid":
            done = True
            self.p.create_vel_controlled_tool(False)
            print("the reason to cause the stop: {}".format(raw_states[0]))

        time.sleep(self._time_step)

        # debug
        # np.set_printoptions(precision=3, suppress=True)
        # print("States: {}".format(states))
        # print(self.current_step)

        # [YUNING] add position and orientation as additional infomation
        pos = self.p.get_pose().get_values()[0:3]
        orn = self.p.get_pose().get_values()[3:6]
        info = {}
        info["pos"] = pos
        info["orn"] = orn

        return states, reward, done, info

    def parse_states(self, raw_states):
        force = np.array([raw_states[1].x, raw_states[1].y, raw_states[1].z])
        torque = np.array([raw_states[2].x, raw_states[2].y, raw_states[2].z])
        states = np.append(force, torque)

        # write FT values into csv
        # if self._debug_mode:
        #     util.write_csv(states, self._ft_reading_file_name, False)

        # make the state consistent with PyBullet
        # states = np.multiply(2.5, states)

        # add Gaussian noise
        # noise = np.random.normal(0, [max(1, 0.2*states[0]), max(1, 0.2*states[1]), max(1, 0.2*states[2]),
        #                              max(1, 0.2*states[3]), max(1, 0.2*states[4]), max(1, 0.2*states[5])],
        #                          np.shape(states.tolist()))
        # states = np.add(states, noise)

        return states

    # calculate the distance between the tool and the target
    def dist_to_target(self, tool_pose, target_pose):
        tool_pos = tool_pose.get_values()[0:3]

        target_pos = (
            np.array(target_pose.get_values()[0:3]) + self._hole_offset
        ).tolist()

        dist_pos = (
            np.linalg.norm(np.subtract(tool_pos, target_pos)) / 1000
        )  # linear dist in m

        tool_orn = tool_pose.get_values()[3:6]
        tool_orn = [
            math.radians(tool_orn[0]),
            math.radians(tool_orn[1]),
            math.radians(tool_orn[2]),
        ]
        target_orn = target_pose.get_values()[3:6]
        target_orn = [
            math.radians(target_orn[0]),
            math.radians(target_orn[1]),
            math.radians(target_orn[2]),
        ]
        dist_orn = math.fabs(
            2 * math.acos(math.fabs(np.dot(tool_orn, target_orn))) - math.pi
        )  # angle diff in rad

        return dist_pos + 0.00 * dist_orn
