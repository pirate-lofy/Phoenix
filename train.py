from ppo2 import PPO2
from environment import CarlaEnv
from vec_wrappers import make_vec_env
from policy import CustomPolicy
import logging
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.vec_env.vec_normalize import VecNormalize

try:
#    def make_env():
#        def make_it():
#            return CarlaEnv()
#        return make_it
#    env = DummyVecEnv([make_env() for _ in range(1)])
#    env = VecNormalize(env)
    env=CarlaEnv()
#    env = make_vec_env(lambda: env, n_envs=1)
    model=PPO2(CustomPolicy,env,save_each=20,tensorboard_log='/home/colab/Desktop/logs')
    model.learn(10000000)
except Exception:
    env.close()
    logging.exception("An exception was thrown!")