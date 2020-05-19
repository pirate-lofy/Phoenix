#from new_env import CarlaEnv
#
#try:
#    env=CarlaEnv()
#    env.reset()
#    for _ in range(100000):
#        data,measures,hl_command,reward,done,_=env.step([[0,0.3,0]])
#        if done:
#            env.reset()
#
#except:
#    env.destroy()

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
    model=PPO2(CustomPolicy,env)
    model.learn(10000)
except Exception:
    env.close()
    logging.exception("An exception was thrown!")