def train(num_timesteps, seed, nenvs):
    from stable_baselines.common import set_global_seeds
    from stable_baselines.common.vec_env.vec_normalize import VecNormalize
#    from stable_baselines.ppo2 import ppo2
#    from stable_baselines.ppo2.policies import CnnPolicy
    import ppo2 
    from policies import CnnPolicy
    import tensorflow as tf
    from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    from environment import CarlaEnv
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    def make_env(i, gpu_num):
        def make_carla():
            env = CarlaEnv()
            return env
        return make_carla

#    env = DummyVecEnv([make_env(8013+i*3, i % 3 + 1) for i in range(nenvs)])
#    env = VecNormalize(env)
    env=CarlaEnv()
    set_global_seeds(seed)
    policy = CnnPolicy
    ppo2.learn(policy=policy, env=env, nsteps=32, nminibatches=32,
        lam=0.95, gamma=0.99, noptepochs=10, log_interval=1,
        ent_coef=0.0,
        lr=3e-4,
        cliprange=0.2,
        total_timesteps=num_timesteps)


def main():
#    logger.configure(dir=args.logdir, format_strs=['tensorboard', 'stdout'])
    train(num_timesteps=100_000, seed=0, nenvs=1)


if __name__ == '__main__':
    main()
