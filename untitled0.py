from new_env import CarlaEnv

env=CarlaEnv()
def main():
    env.reset()
    print('start')
    while True:
        _,_,_,done,_=env.step([[0,0.25,0]])
    print('done')
#    while True:
#        env.step([[0,0,0]])
        

try:
    main()
except:
    env.destroy()        


