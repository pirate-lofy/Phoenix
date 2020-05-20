from environment import CarlaEnv
import logging

try:
    env=CarlaEnv()
    env.reset()
    for _ in range(100000):
        data,measures,hl_command,reward,done,_=env.step([0,0.3,0])
        if done:
            env.reset()

except Exception:
    env.close()
    logging.exception('')