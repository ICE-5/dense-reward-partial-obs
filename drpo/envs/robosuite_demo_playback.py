"""
A convenience script to playback random demonstrations from
a set of demonstrations stored in a hdf5 file.

Arguments:
    --folder (str): Path to demonstrations
    --use-actions (optional): If this flag is provided, the actions are played back
        through the MuJoCo simulator, instead of loading the simulator states
        one by one.
    --visualize-gripper (optional): If set, will visualize the gripper site

Example:
    $ python playback_demonstrations_from_hdf5.py --folder ../models/assets/demonstrations/SawyerPickPlace/
"""

import argparse
import json
import os
import random

import h5py
import numpy as np

import robosuite
from robosuite.utils.mjcf_utils import postprocess_model_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        help="Path to your demonstration folder that contains the demo.hdf5 file, e.g.: "
        "'path_to_assets_dir/demonstrations/YOUR_DEMONSTRATION'",
        default="demos"
    ),
    parser.add_argument(
        "--use-actions",
        action="store_true",
        default=True,
    )
    args = parser.parse_args()

    demo_path = args.folder
    hdf5_path = os.path.join(demo_path, "demo_3.hdf5")
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]
    env_info = json.loads(f["data"].attrs["env_info"])

    env = robosuite.make(
        **env_info,
        has_renderer=True,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
    )

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    ep = random.choice(demos)
    print(f"Playing back random episode {ep} (press ESC to quit)")

    # read the model xml, using the metadata stored in the attribute for this episode
    model_xml = f["data/{}".format(ep)].attrs["model_file"]

    env.reset()
    xml = postprocess_model_xml(model_xml)
    env.reset_from_xml_string(xml)
    env.sim.reset()
    env.viewer.set_camera(0)

    # load the flattened mujoco states
    states = f["data/{}/states".format(ep)][()]
    actions = np.array(f["data/{}/actions".format(ep)][()])

    n = len(states)
    ft = np.zeros([n, 6])

    if args.use_actions:
        # load the initial state
        env.sim.set_state_from_flattened(states[0])
        env.sim.forward()

        # load the actions and play them back open-loop
        num_actions = actions.shape[0]
        action_dim = actions.shape[1]

        for j, action in enumerate(actions):
            obs, reward, done, info = env.step(action)
            env.render()

            # robot = env.robots[0]
            # force = robot.ee_force
            # torque = robot.ee_torque
            # ft[j, :3] = force
            # ft[j, 3:] = torque

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                if not np.all(np.equal(states[j + 1], state_playback)):
                    err = np.linalg.norm(states[j + 1] - state_playback)
                    print(f"[warning] playback diverged by {err} for ep {ep} at step {j}")

    else:

        # force the sequence of internal mujoco states one by one
        for i, state in enumerate(states):
            env.sim.set_state_from_flattened(state)
            env.sim.forward()
            # env.render()

            # get force/torque
            robot = env.robots[0]
            force = robot.ee_force
            torque = robot.ee_torque
            ft[i, :3] = force
            ft[i, 3:] = torque

        # NOTE: compare abrupt playback to successive playback result
    num_trials = 10
    tmp_ft_0 = np.zeros(6)
    tmp_ft_1 = np.zeros(6)
    for _ in range(num_trials):
        j = np.random.randint(n)
        state = states[j]
        env.sim.set_state_from_flattened(state)
        env.sim.forward()
        env.render()

        robot = env.robots[0]
        force = robot.ee_force
        torque = robot.ee_torque
        tmp_ft_0[:3] = force
        tmp_ft_0[3:] = torque

        # print(j, tmp_ft, ft[j, :].flatten(), np.linalg.norm(tmp_ft - ft[j, :].flatten()))
        print(j)
        print("before action:", np.linalg.norm(tmp_ft_0 - ft[j, :].flatten()))

        env.step(actions[j])
        robot = env.robots[0]
        force = robot.ee_force
        torque = robot.ee_torque
        tmp_ft_1[:3] = force
        tmp_ft_1[3:] = torque
        print("after action:", np.linalg.norm(tmp_ft_1 - ft[j+1, :].flatten()))
        print("\n")



    
    # visualize each dimension of FT
    import matplotlib.pyplot as plt
    plt.figure(figsize=(25, 5))
    plt.title(f"Episode {ep}")
    for i in range(6):
        plt.plot(np.arange(n), ft[:, i], linewidth=0.5)
    plt.show()


    f.close()

