import numpy as np
import robosuite

# create environment instance
env = robosuite.make(
    env_name="Door",
    robots="Panda",             # load a Sawyer robot and a Panda robot
    gripper_types="default",                # use default grippers per robot arm
    # controller_configs=controller_config,   # each arm is controlled using OSC
    # env_configuration="single-arm-opposed", # (two-arm envs only) arms face each other
    has_renderer=False,                     # no on-screen rendering
    has_offscreen_renderer=True,            # off-screen rendering needed for image obs
    control_freq=20,                        # 20 hz control for applied actions
    horizon=200,                            # each episode terminates after 200 steps
    use_object_obs=False,                   # don't provide object observations to agent
    use_camera_obs=True,                   # provide image observations to agent
    camera_names="agentview",               # use "agentview" camera for observations
    camera_heights=84,                      # image height
    camera_widths=84,                       # image width
    reward_shaping=True,                    # use a dense reward signal for learning
)


# reset the environment
env.reset()

for i in range(1000):
    action = np.random.randn(env.robots[0].dof) # sample random action
    obs, reward, done, info = env.step(action)  # take action in the environment
    # env.render()  # render on display
    print(reward)