from gym.envs.registration import register

register(
    id='comet-v0',
    entry_point='gym_comet.envs:CoMet',
)