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

        # test
        self.action_dim = self.env.action_dim

    def sample_step(self, **kwargs):

        try:
            demo_name = kwargs["demo_name"]
            level = kwargs["level"]
            initial_state = kwargs["initial_state"]
            initial_global_timestep = kwargs["initial_global_timestep"]
        except KeyError:
            print("Missing necessary parameters for sampling.")

        for b in range(self.num_branches):
            sampled_actions = np.random.uniform(
                self.action_low,
                self.action_high,
                [self.num_steps_per_branch, self.action_dim],
            )

            branch_index = level * self.num_branches + b + 1

            self.record_branch(
                demo_name=demo_name,
                branch_index=branch_index,
                initial_state=initial_state,
                initial_global_timestep=initial_global_timestep,
                actions=sampled_actions,
            )
