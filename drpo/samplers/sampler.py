import h5py
import pathlib
import pickle
import numpy as np


from abc import ABC, abstractmethod

from robosuite.utils.mjcf_utils import postprocess_model_xml

from drpo.envs.envs_launcher import env_creator
# from drpo.dataloader.utils import Sample


class Sampler(ABC):
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> None:
        super().__init__()

        # create env based on demo
        self.env, self.env_name = env_creator(demo_path)

        # load demo_file
        self.demo_file = h5py.File(demo_path, "r")
        self.demo_names = list(self.demo_file["data"].keys())

        # load samping hyperparameters
        self.sensors = config["sensor_used_in_sampling"]
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
            self._sample_demo(demo_name)

        # save and close file
        self.data_file.close()

        pickle.dump(
            self.codes,
            open(self.out_dir / "codes.pkl", "wb"),
        )

    def _sample_demo(self, demo_name: str):
        """Specify demo-wise sampling
        - forward / backward sampling
        - sampling interval (# of steps to skip between sampling)
        """
        # create group per each demo
        self.grp.create_group(demo_name)

        # reset env per demo
        self.env.reset()
        model_xml = self.demo_file[f"data/{demo_name}"].attrs["model_file"]
        xml = postprocess_model_xml(model_xml)
        self.env.reset_from_xml_string(xml)
        self.env.sim.reset()

        # record demo
        states = self._record_demo(demo_name)
        n = len(states)

        # get steps to sample
        # NOTE: manually skipped a few steps (5 steps)
        if self.sampling_forward:
            sampling_timesteps = np.arange(5, n, self.sampling_interval)
        else:
            sampling_timesteps = np.arange(n, 5, -self.sampling_interval)

        # sample step
        for level, timestep in enumerate(sampling_timesteps):
            state = states[timestep]
            self.__load_state(state)
            identifier = {
                "demo_name": demo_name,
                "level": level,
                "global_timestep": timestep,
            }
            self._sample_step(**identifier)
        

    @abstractmethod
    def _sample_step(self, **kwargs):
        """Specify step-wise sampling
        - # of branches to generate
        - # of steps to generate per branch
        - sampling variance
        """
        raise NotImplementedError()

    def __load_state(self, state):
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()

    # TODO: currently use last state of episode as done
    def _record_demo(self, demo_name):
        """Store a demo episode's FT, image, proprio, action as branch in data.hdf5

        Args:
            demo_name (str): name of demo to record
            branch_name (str, optional): _description_. Defaults to "branch_0".
        """
        # by default, the real demo is the stem branch, with branch_index=0
        branch_index = 0

        # get demo actions and states
        states = self.demo_file[f"data/{demo_name}/states"][()]
        actions = self.demo_file[f"data/{demo_name}/actions"][()]
        n = len(states)
        print(n)

        # get demo droup in data
        demo_grp = self.grp[demo_name]

        # create branch
        branch_grp = demo_grp.create_group(str(branch_index))

        # set initial state
        self.env.sim.set_state_from_flattened(states[0])
        self.env.sim.forward()

        # BEST: peel hardcode dims
        ft_arr = np.zeros([n, 6])
        image_arr = np.zeros([n, 128, 128, 3])
        proprio_arr = np.zeros([n, 32])
        action_arr = np.zeros([n, 7])
        reward_arr = np.zeros(n)

        for j, action in enumerate(actions):
            obs, reward, _, _ = self.env.step(action)
            robot = self.env.robots[0]
            force = robot.ee_force
            torque = robot.ee_torque

            ft = np.concatenate([force, torque])

            # save to dataset
            ft_arr[j, :3] = force
            ft_arr[j, 3:] = torque
            image_arr[j, :, :, :] = obs["agentview_image"]
            proprio_arr[j, :] = obs["robot0_proprio-state"]
            action_arr[j, :] = action
            reward_arr[j] = reward

            # add code, skip first one for pair consideration
            if j > 0:
                code = f"{demo_name}.{branch_index}.{j}.{j}"
                self.codes.append(code)

        # save container in hdf5
        branch_grp.create_dataset("image", data=np.array(image_arr))
        branch_grp.create_dataset("ft", data=np.array(ft_arr))
        branch_grp.create_dataset("proprio", data=np.array(proprio_arr))
        branch_grp.create_dataset("action", data=np.array(action_arr))
        branch_grp.create_dataset("reward", data=np.array(reward_arr))

        return states
