from new_env import CarlaEnv
from model import Model
from T_policies import CnnPolicy
import tensorflow as tf
from runner import Runner
from T_ppo2 import learn


def constfn(val):
    def f(_):
        return val
    return f

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
lam=0.95
gamma=0.99
n_epochs=1000000
n_opt_epochs=4
save_each=10
log_interval=1
lr=0.0003
clip_range=0.1

    
if isinstance(lr, float): lr = constfn(lr)
else: assert callable(lr)
if isinstance(clip_range, float): clip_range = constfn(clip_range)
else: assert callable(clip_range)
 
model=Model(CnnPolicy,ob_img_space,ob_measure_space,ac_space,n_envs,
                n_batch_critic,ent_coef,vf_coef,max_grad_norm,
                frame_stack)

path='/home/Desktop/checkpoints'
i=model.load_latest(path)
runner=Runner(env,model,n_steps,gamma,lam)

try: 
    learn(model,runner,n_epochs,n_steps,n_min_patches,n_opt_epochs,n_batch,
          clip_range,save_each,log_interval,lr,i)
except :
    env.destroy()