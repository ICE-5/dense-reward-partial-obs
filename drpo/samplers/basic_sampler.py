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
        self.action_dim = self.env.action_dim

    def sample_step(self, **kwargs):

        try:
            demo_name = kwargs["demo_name"]
            demo_states = kwargs["demo_states"]
            demo_actions = kwargs["demo_actions"]
            level = kwargs["level"]
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
                demo_states=demo_states,
                demo_actions=demo_actions,
                branch_index=branch_index,
                initial_global_timestep=initial_global_timestep,
                actions=sampled_actions,
            )
