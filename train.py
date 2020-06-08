from ppo2 import PPO2
from environment import CarlaEnv
from policy import CustomPolicy
import logging
from vec_wrappers import DummyVecEnv
from stable_baselines.common import set_global_seeds

try:
    def make_env(rank, seed=0):
        def _init():
            env = CarlaEnv(rank)
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(8)])
    model=PPO2(CustomPolicy,env,save_each=10)
    model.learn(10000000)
except Exception:
    env.close()
    logging.exception("An exception was thrown!")

