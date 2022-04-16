import pathlib
import numpy as np

from drpo.samplers.sampler import Sampler

class BasicSampler(Sampler):
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> None:
        super().__init__(config, demo_path, out_dir)

        # get env action spec
        self.action_low, self.action_high = self.env.action_spec


    def _sample_step(self, **kwargs):

        demo_grp = self.grp[kwargs["demo_name"]]
        level = kwargs["level"]
        
        for b in range(self.num_branches):
            # branch_index=0 is the stem, all sampled branches start from branch_index=1
            branch_index = level * self.num_branches + b + 1
            branch_grp = demo_grp.create_group(str(branch_index))

            ft_arr = []
            image_arr = []
            proprio_arr = []
            action_arr = []
            reward_arr = []

            # TODO: test!
            global_timestep = kwargs["global_timestep"]
            local_timestep = -1

            for _ in range(self.num_steps_per_branch):
                # sample randomly without control
                action = np.random.uniform(self.action_low, self.action_high)

                obs, reward, _, _ = self.env.step(action)
                force = self.robot.ee_force
                torque = self.robot.ee_torque
                ft = np.concatenate([force, torque])

                # add obs to dataset
                ft_arr.append(ft)
                image_arr.append(obs["agentview_image"])
                proprio_arr.append(obs["robot0_proprio-state"])
                action_arr.append(action)
                reward_arr.append(reward)

                global_timestep += 1
                local_timestep += 1

                # add code
                code = f"{branch_index}.{global_timestep}.{local_timestep}"
                self.codes.append(code)
            
            # save dataset in hdf5
            branch_grp.create_dataset("image", data=np.array(image_arr))
            branch_grp.create_dataset("ft", data=np.array(ft_arr))
            branch_grp.create_dataset("proprio", data=np.array(proprio_arr))
            branch_grp.create_dataset("action", data=np.array(action_arr))
            branch_grp.create_dataset("reward", data=np.array(reward_arr))


        