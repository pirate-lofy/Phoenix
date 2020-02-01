import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import fc
from stable_baselines.common.distributions import make_proba_dist_type
from feature_extractor import impala_cnn

def process_measurements(X_measurements):
    activ = tf.nn.relu
    h=activ(fc(X_measurements, 'fc1_m', nh=8, init_scale=np.sqrt(2)))
    return activ(fc(h, 'fc2_m', nh=4, init_scale=np.sqrt(2)))

class CnnPolicy():

    def __init__(self, sess, ob_img_space, ob_measure_space, ac_space, reuse=False): #pylint: disable=W0613
        ob_img_shape=(None,*ob_img_space.shape)
        measures_shape = (None,ob_measure_space.shape[0])
        n_actions = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_img_shape,name='X') #obs
        X_measurements = tf.placeholder(tf.float32, shape=measures_shape,name='measures')

        with tf.variable_scope("model", reuse=reuse):
            h = tf.cast(impala_cnn(X), tf.float32, name='cast_1')
            h_measurements = tf.cast(process_measurements(X_measurements), tf.float32, name='cast_2')
            
            h_concat = tf.concat([h, h_measurements], axis=1, name='concat_1')
            h_concat=tf.nn.relu(fc(h_concat, 'after_conc_layer',128,init_scale=np.sqrt(2)))
            
            ''' actor model branch '''
            pi=tf.nn.relu(fc(h_concat,'actor_branch_1',64,init_scale=np.sqrt(2)))
            pi=tf.nn.relu(fc(pi,'actor_branch_2',16,init_scale=np.sqrt(2)))
            pi = fc(pi, 'pi', n_actions, init_scale=0.01)
            
            logstd = tf.get_variable(name="logstd", shape=[1, n_actions], initializer=tf.zeros_initializer())
            
            ''' critic branch '''
            vf= tf.nn.relu(fc(h_concat,'critic_branch_1',32,init_scale=np.sqrt(2)))
            vf= tf.nn.relu(fc(vf,'critic_branch_2',8,init_scale=np.sqrt(2)))
            vf = fc(vf, 'v', 1)[:,0]
        
        pdparam = tf.concat([pi,logstd], axis=1)

        self.pdtype = make_proba_dist_type(ac_space)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        a1=tf.identity(a0,'actions')
        self.initial_state = None

        def step(ob_img, ob_measure, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob_img, X_measurements:ob_measure})
            return a, v, self.initial_state, neglogp

        def value(ob_img, ob_measure, *_args, **_kwargs):
            return sess.run(vf, {X:ob_img, X_measurements:ob_measure})

        self.X = X
        self.X_measurements = X_measurements
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value