from environment import CarlaEnv

env=CarlaEnv()
env.reset()

for _ in range(100):
    _,_, rewards, done, _ = env.step([[0,0.5,0]])
#    print(rewards,done)