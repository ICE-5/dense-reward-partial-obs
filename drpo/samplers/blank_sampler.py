import pathlib
from drpo.samplers.sampler import Sampler

class BlankSampler(Sampler):  
    def __init__(
        self,
        config: dict,
        demo_path: pathlib.Path,
        out_dir: pathlib.Path,
    ) -> None:
        super().__init__(config, demo_path, out_dir)

    def sample_step(self, **kwargs):
        pass


        