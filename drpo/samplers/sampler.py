from sre_constants import SUCCESS
import h5py
import numpy as np
import pathlib
import pickle

from abc import ABC, abstractmethod

from robosuite.utils.mjcf_utils import postprocess_model_xml

from drpo.envs.envs_launcher import env_creator
from drpo.utils import prGreen, prYellow


class Sampler(ABC):
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> None:
        super().__init__()
        # load config
        self.config = config

        # create env based on demo
        self.env, self.env_name = env_creator(demo_path)

        # load demo_file
        self.demo_file = h5py.File(demo_path, "r")
        self.demo_names = list(self.demo_file["data"].keys())

        # load samping hyperparameters
        self.sampling_forward = config["sampling_forward"]
        self.sampling_interval = config["sampling_interval"]
        self.num_branches = config["num_branches"]
        self.num_steps_per_branch = config["num_steps_per_branch"]

        self.out_dir = pathlib.Path(out_dir)

    def sample(self, demo_names=None):
        """General sampling interface"""

        # use data.hdf5 to store data
        # use samples.pkl to store sample codes
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.data_file = h5py.File(self.out_dir / "data.hdf5", "w")
        self.grp = self.data_file.create_group("data")
        self.codes = []

        # determine which demo(s) to sample
        if demo_names is None:
            demo_names = self.demo_names
        else:
            for demo_name in list(demo_names):
                if demo_name not in self.demo_names:
                    raise ValueError("Invalid demo name provided.")

        # sample
        for demo_name in demo_names:
            # create group per each demo
            self.grp.create_group(demo_name)

            # reset env per demo
            self.env.reset()
            model_xml = self.demo_file[f"data/{demo_name}"].attrs["model_file"]
            xml = postprocess_model_xml(model_xml)
            self.env.reset_from_xml_string(xml)
            self.env.sim.reset()

            # get demo states and actions
            states = self.demo_file[f"data/{demo_name}/states"][()]
            actions = self.demo_file[f"data/{demo_name}/actions"][()]

            # load original state
            self.load_state_with_actions(timestep=-1, states=states, actions=actions)
            # self.load_state(states[0])

            # record original demo
            self.record_branch(
                demo_name=demo_name,
                demo_states=states,
                demo_actions=actions,
                branch_index=0,
                initial_global_timestep=-1,
                actions=actions,
            )

            # sample demo
            self.sample_demo(demo_name=demo_name, states=states, actions=actions)

        # save and close file
        self.data_file.close()

        pickle.dump(
            self.codes,
            open(self.out_dir / "codes.pkl", "wb"),
        )

    def sample_demo(
        self, demo_name: str, states: np.ndarray, actions: np.ndarray
    ) -> None:
        n = len(states)

        # get steps to sample
        # NOTE: manually skipped a few steps (5 steps)
        padding = 5
        if self.sampling_forward:
            sampling_timesteps = np.arange(padding, n - padding, self.sampling_interval)
        else:
            sampling_timesteps = np.arange(
                n - padding, padding, -self.sampling_interval
            )

        # sample at (global) timestep
        for level, timestep in enumerate(sampling_timesteps):
            kwargs = {
                "demo_name": demo_name,
                "demo_states": states,
                "demo_actions": actions,
                "level": level,
                "initial_global_timestep": timestep,
            }
            self.sample_step(**kwargs)

    @abstractmethod
    def sample_step(self, **kwargs):
        """Specify step-wise sampling
        - # of branches to generate
        - # of steps to generate per branch
        - sampling variance
        """
        raise NotImplementedError()

    def load_state(self, state):
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    def load_state_with_actions(
        self, timestep: int, states: np.ndarray, actions: np.ndarray
    ):
        self.env.sim.reset()
        self.env.sim.set_state_from_flattened(states[0])
        self.env.sim.forward()

        if timestep >= 0:
            for j, action in enumerate(actions):
                self.env.step(action)
                if j > timestep:
                    break

    def record_branch(
        self,
        demo_name: str,
        demo_states: np.ndarray,
        demo_actions: np.ndarray,
        branch_index: (int or str),
        initial_global_timestep: int,
        actions: np.ndarray,
    ) -> None:
        """Record obs given an initial state and a sequence of actions

        Args:
            demo_name (str): demo name
            branch_index (int or str): branch index, 0 if stem
            initial_state (np.ndarray): initial state to start recording
            initial_global_timestep (int): initial state's global timestep
            actions (np.ndarray): sequence of actions
        """
        # create branch group in demo group
        demo_grp = self.grp[demo_name]
        branch_grp = demo_grp.create_group(str(branch_index))

        # # reset env
        self.load_state_with_actions(
            timestep=initial_global_timestep, states=demo_states, actions=demo_actions
        )
        playback_state = self.env.sim.get_state().flatten()

        prYellow(f"WARNING| state diff: {np.linalg.norm(playback_state - demo_states[initial_global_timestep + 1])}")

        # create container by num of actions
        n = len(actions)
        ft_arr = np.zeros([n, 6])
        image_arr = np.zeros([n, self.config["image_dim"], self.config["image_dim"], 3])
        depth_arr = np.zeros([n, self.config["image_dim"], self.config["image_dim"], 1])
        proprio_arr = np.zeros([n, self.config["proprio_dim"]])
        object_arr = np.zeros([n, self.config["object_dim"]])
        action_arr = np.zeros([n, self.config["action_dim"]])
        reward_arr = np.zeros(n)

        for j, action in enumerate(actions):
            obs, reward, _, _ = self.env.step(action)
            robot = self.env.robots[0]
            force = robot.ee_force
            torque = robot.ee_torque

            # save to dataset
            ft_arr[j, :3] = force
            ft_arr[j, 3:] = torque
            image_arr[j, :, :, :] = obs["agentview_image"]
            depth_arr[j, :, :, :] = obs["agentview_depth"]
            proprio_arr[j, :] = obs["robot0_proprio-state"]
            object_arr[j, :] = obs["object-state"]
            action_arr[j, :] = action
            reward_arr[j] = reward

            # add code, skip first one for pair obs completeness
            global_timestep = initial_global_timestep + 1 + j
            local_timestep = j
            code = f"{demo_name}.{branch_index}.{global_timestep}.{local_timestep}"
            if global_timestep > 0:
                self.codes.append(code)

        # save container in hdf5
        branch_grp.create_dataset("image", data=np.array(image_arr))
        branch_grp.create_dataset("depth", data=np.array(depth_arr))
        branch_grp.create_dataset("ft", data=np.array(ft_arr))
        branch_grp.create_dataset("proprio", data=np.array(proprio_arr))
        branch_grp.create_dataset("object", data=np.array(object_arr))
        branch_grp.create_dataset("action", data=np.array(action_arr))
        branch_grp.create_dataset("reward", data=np.array(reward_arr))
