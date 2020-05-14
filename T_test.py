from new_env import CarlaEnv
from model import Model
from T_policies import CnnPolicy
import numpy as np
import tensorflow as tf


def config():
    ncpu=4
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    
config()
env=CarlaEnv()


'''model parameters'''
ob_img_space = env.observation_space.spaces[0]
ob_measure_space = env.observation_space.spaces[1]
ac_space = env.action_space  

'''learn parameters'''
n_envs = 1 # n_batch_actor
n_steps=4
n_min_patches=4
n_batch = n_envs * n_steps
n_batch_critic = n_batch // n_min_patches 

'''numbers parameters according to the papper'''
ent_coef=0.01
vf_coef=0.5
max_grad_norm=0.5
frame_stack=2

 
model=Model(CnnPolicy,ob_img_space,ob_measure_space,ac_space,n_envs,
                n_batch_critic,ent_coef,vf_coef,max_grad_norm,
                frame_stack)

model.load('checkpoints\\410')

i,m=env.reset()
i=np.expand_dims(i,0)
m=np.expand_dims(m,0)
for _ in range(10000):   
    actions,v,_=model.step(i,m)
    i,m,reward,done,_=env.step(actions)
    i=np.expand_dims(i,0)
    m=np.expand_dims(m,0)
    if done:
        env.reset()