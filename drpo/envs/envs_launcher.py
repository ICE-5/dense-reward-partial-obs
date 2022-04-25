import argparse
import json
import os
import pathlib

import h5py
import numpy as np


import robosuite
from robosuite.environments.base import MujocoEnv
from robosuite.renderers import load_renderer_config


def env_creator(demo_path: pathlib.Path) -> MujocoEnv:
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
        # camera_names="frontview",
        # renderer="igibson",
        # render_mode="headless",
        # renderer_config=config
    )

    # # change renderer config
    # config = load_renderer_config("igibson")

    # if args.vision_modality == "rgb":
    #     config["vision_modalities"] = ["rgb"]
    # if args.vision_modality == "segmentation":
    #     config["vision_modalities"] = ["seg"]
    #     config["msaa"] = False
    # if args.vision_modality == "depth":
    #     config["vision_modalities"] = ["3d"]
    # if args.vision_modality == "normal":
    #     config["vision_modalities"] = ["normal"]

    # config["camera_obs"] = True
    # config["render_mode"] = "headless"

    return (env, env_name)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo-path",
        type=str,
        default="demos/demo.hdf5",
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "demos/",
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    args = parser.parse_args()
    demo_path = args.demo_path

    # test env_creator
    env, _ = env_creator(demo_path)

    # preview env step
    env.reset()
    action = np.random.randn(7)
    for i in range(10):
        obs, reward, done, info = env.step(action)
        # env.render()

        if i == 0:
            for key in obs.keys():
                print(f"{key : <25} {type(obs[key])}  {np.shape(obs[key])}")


        # NOTE: test image and camera
        # from matplotlib import pyplot as plt
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
