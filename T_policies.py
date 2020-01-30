import numpy as np
import tensorflow as tf
from stable_baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from stable_baselines.common.distributions import make_proba_dist_type

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32)
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

def process_measurements(X_measurements):
    activ = tf.nn.relu
    return activ(fc(X_measurements, 'fc1_m', nh=4, init_scale=np.sqrt(2)))

class CnnPolicy():

    def __init__(self, sess, ob_img_space, ob_measure_space, ac_space, reuse=False): #pylint: disable=W0613
        ob_img_shape=(None,*ob_img_space.shape)
        measures_shape = (None,ob_measure_space.shape[0])
        n_actions = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_img_shape) #obs
        X_measurements = tf.placeholder(tf.float32, shape=measures_shape)

        with tf.variable_scope("model", reuse=reuse):
            h = tf.cast(nature_cnn(X), tf.float32, name='cast_1')
            h_measurements = tf.cast(process_measurements(X_measurements), tf.float32, name='cast_2')
            h_concat = tf.concat([h, h_measurements], axis=1, name='concat_1')
            pi = fc(h_concat, 'pi', n_actions, init_scale=0.01)
            vf = fc(h_concat, 'v', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, n_actions], initializer=tf.zeros_initializer())
        
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_proba_dist_type(ac_space)
        self.pd = self.pdtype.proba_distribution_from_flat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
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