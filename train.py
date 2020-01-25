from environment import CarlaEnv
from environment import TCPConnectionError
import random
import cv2 as cv

env=CarlaEnv()
env.start()
env.reset()

def train():
    for _ in range(10000):
        (observation,_), reward, done, _ = env.step([random.uniform(-1.0, 1.0)
                                                    ,0.5,0])

        if done:
            env.reset()
        

def main():
    while True:
        try:
            train()
        except TCPConnectionError:
#            print('waiting for the server....')
            pass


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('cancelled by the user')