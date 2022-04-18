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

        demo_name = kwargs["demo_name"]
        demo_grp = self.grp[demo_name]
        level = kwargs["level"]
        
        for b in range(self.num_branches):
            # branch_index=0 is the stem, all sampled branches start from branch_index=1
            branch_index = level * self.num_branches + b + 1
            branch_grp = demo_grp.create_group(str(branch_index))
            n = self.num_steps_per_branch

            ft_arr = np.zeros([n, 6])
            image_arr = np.zeros([n, 128, 128, 3])
            proprio_arr = np.zeros([n, 32])
            action_arr = np.zeros([n, 7])
            reward_arr = np.zeros(n)

            # TODO: test!
            # position at branch root
            global_timestep = kwargs["global_timestep"]
            local_timestep = -1

            for j in range(n):
                # sample randomly without control
                action = np.random.uniform(self.action_low, self.action_high)

                obs, reward, _, _ = self.env.step(action)
                robot = self.env.robots[0]
                force = robot.ee_force
                torque = robot.ee_torque

                # save to dataset
                ft_arr[j, :3] = force
                ft_arr[j, 3:] = torque
                image_arr[j, :, :, :] = obs["agentview_image"]
                proprio_arr[j, :] = obs["robot0_proprio-state"]
                action_arr[j, :] = action
                reward_arr[j] = reward

                global_timestep += 1
                local_timestep += 1

                # add code
                code = f"{demo_name}.{branch_index}.{global_timestep}.{local_timestep}"
                self.codes.append(code)
            
            # save dataset in hdf5
            branch_grp.create_dataset("image", data=np.array(image_arr))
            branch_grp.create_dataset("ft", data=np.array(ft_arr))
            branch_grp.create_dataset("proprio", data=np.array(proprio_arr))
            branch_grp.create_dataset("action", data=np.array(action_arr))
            branch_grp.create_dataset("reward", data=np.array(reward_arr))


        