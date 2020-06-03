import tensorflow as tf
from base_policies import ActorCriticPolicy,RecurrentActorCriticPolicy
from feature_extractor import impala_gen
from stable_baselines.common.tf_util import batch_to_seq, seq_to_batch
from stable_baselines.common.tf_layers import conv, linear, conv_to_fc, lstm


class CustomPolicy(ActorCriticPolicy):
    initial_state=None
    
    def __init__(self, sess, ob_space, mes_space, hl_space, ac_space, n_env, 
                 n_steps, n_batch, reuse=False):
        super(CustomPolicy, self).__init__(sess, ob_space, mes_space, hl_space, ac_space, n_env, 
                 n_steps, n_batch, reuse=reuse)

        with tf.variable_scope("model", reuse=reuse):
            activ = tf.nn.relu

            extracted_features = impala_gen(self.processed_obs)
            extracted_features = tf.layers.flatten(extracted_features)
            
            mes=activ(tf.layers.dense(self.processed_mes,16,name='mes1'))
            mes=activ(tf.layers.dense(mes,16,name='mes2'))
            
            concat=tf.concat([extracted_features,mes,self.processed_hl],axis=1,name='concat1')
            concat=activ(tf.layers.dense(concat,128,name='concat2'))
            
            pi=activ(tf.layers.dense(concat,128,name='pi'))
            pi_latent=activ(tf.layers.dense(pi,128,name='pi_latent'))

            vf_latent=activ(tf.layers.dense(concat,64,name='vf_latent'))
            value_fn=linear(vf_latent, 'vf', 1)

            self.proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

        self.value_fn = value_fn
        self._setup_init()

    def step(self, obs, measures, hl_command, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.mes_ph:measures,
                                                    self.hl_ph:hl_command})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.mes_ph:measures,
                                                    self.hl_ph:hl_command})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, measures, hl_command, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.mes_ph:measures,
                                                    self.hl_ph:hl_command})

    def value(self, obs, measures, hl_command, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.mes_ph:measures,
                                                    self.hl_ph:hl_command})

    
    

class LstmPolicy(RecurrentActorCriticPolicy):
    recurrent = True

    def __init__(self, sess, ob_space, mes_space, hl_space,ac_space, n_env, n_steps, n_batch, n_lstm=256, 
                 reuse=False, layers=None,activ=tf.nn.relu, layer_norm=True):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(LstmPolicy, self).__init__(sess, ob_space, mes_space, hl_space, ac_space, n_env, n_steps, n_batch,
                                         state_shape=(2 * n_lstm, ), reuse=reuse)

        with tf.variable_scope("model", reuse=reuse):
            '''
            image part, impala
            '''
            extracted_features = impala_cnn(self.processed_obs)
            input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
            masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
            
            rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                         layer_norm=layer_norm)
            rnn_output = seq_to_batch(rnn_output)
            
            '''
            remaining part
            '''
            mes=activ(tf.layers.dense(self.processed_mes,16,name='mes1'))
            mes=activ(tf.layers.dense(mes,16,name='mes2'))
            
            concat=tf.concat([rnn_output,mes,self.processed_hl],axis=1,name='concat1')
            concat=activ(tf.layers.dense(concat,128,name='concat2'))
            
            pi=activ(tf.layers.dense(concat,128,name='pi'))
            pi_latent=activ(tf.layers.dense(pi,128,name='pi_latent'))

            vf_latent=activ(tf.layers.dense(concat,64,name='vf_latent'))
            value_fn=linear(vf_latent, 'vf', 1)
#
            self.proba_distribution, self._policy, self.q_value = \
                self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)
            print('%%%%%%%%^^^^^^^&^&&&&&&&&&',type(value_fn[:, 0]))
            self.value_fn = value_fn
        self._setup_init()

    def step(self, obs, measures, hl_command, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.mes_ph:measures,self.hl_ph:hl_command, 
                                  self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.mes_ph:measures,self.hl_ph:hl_command, 
                                  self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, measures, hl_command, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.mes_ph:measures,self.hl_ph:hl_command, 
                                                 self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, measures, hl_command, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.mes_ph:measures,self.hl_ph:hl_command, 
                                               self.states_ph: state, self.dones_ph: mask})
