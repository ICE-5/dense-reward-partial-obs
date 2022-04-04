import argparse
import json
import os
# import random

import h5py
import numpy as np

from matplotlib import pyplot as plt
        

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml
from robosuite.environments.base import MujocoEnv
# from robosuite.robots.single import 


def env_creator(demo_path: str) -> MujocoEnv:
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--folder",
    #     type=str,
    #     help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
    #     "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    # ),
    # parser.add_argument(
    #     "--use-actions",
    #     action="store_true",
    # )
    # args = parser.parse_args()

    # hdf5_path = os.path.join(demo_path, "demo.hdf5")
    f = h5py.File(demo_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    return (env, env_name, f)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-path",
        type=str,
        default="../demo/demo.hdf5",
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()
    demo_path = args.demo_path

    # test env_creator
    env = env_creator(demo_path)

    # preview env step
    env.reset()
    action = np.random.randn(7)
    for i in range(100):
        obs, reward, done, info = env.step(action)
        # env.render()

        if i == 0:
            for key in obs.keys():
                print(f"{key : <25} {type(obs[key])}  {np.shape(obs[key])}")
            # NOTE: check proprio


        # NOTE: test image and camera
        # if i == 0:
        #     # print(">>>", type(obs))
        #     # print(">>>", obs.keys())
        #     img = obs["agentview_image"]
        #     # print(">>>", img)
        #     plt.imsave(f"../debug/vis_env/env_camera.png", img, cmap="gray")
        
        # NOTE: test force and torque
        robot = env.robots[0]
        force, torque = robot.ee_force, robot.ee_torque
        # print(force, torque)
