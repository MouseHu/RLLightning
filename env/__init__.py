from gym import register

register(
    id='Kelly',
    entry_point='env.kelly::Kelly',
    max_episode_steps=1000,
    reward_threshold=100.0,
)
