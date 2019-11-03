from gym.envs.registration import register

register(
    id='ContinuousCartPole-v0',
    entry_point='continuous_cartpole.envs:ContinuousCartPole_v0',
    max_episode_steps=200,
    reward_threshold=195
)

# register(
    # id='ContinuousCartPole-v1',
    # entry_point='continuous_cartpole.envs:ContinuousCartPole_v1',
    # max_episode_steps=200
# )