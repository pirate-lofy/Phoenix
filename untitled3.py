from new_env import CarlaEnv

env=CarlaEnv()
env.reset()

try:
    for _ in range(10000):
        env.step([[0.5,0.5,0]])
finally:
    env.destroy()        
