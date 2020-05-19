from abc import ABC
import tensorflow as tf

from stable_baselines.common.distributions import make_proba_dist_type, DiagGaussianProbabilityDistribution
from stable_baselines.common.input import observation_input


class BasePolicy(ABC):

    def __init__(self, sess, ob_space, mes_space, hl_space, ac_space, n_env, n_steps, n_batch, reuse=False):
        self.n_env = n_env
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.sess = sess
        self.reuse = reuse
        self.ob_space = ob_space
        self.ac_space = ac_space

        with tf.variable_scope("input", reuse=False):
            self.obs_ph, self.processed_obs = observation_input(ob_space, n_batch)
            self.mes_ph, self.processed_mes = observation_input(mes_space, n_batch)
            self.hl_ph, self.processed_hl = observation_input(hl_space, n_batch)            
            self.action_ph = tf.placeholder(dtype=ac_space.dtype, shape=(n_batch,) + ac_space.shape,
                                                 name="action_ph")


class ActorCriticPolicy(BasePolicy):
    def __init__(self, sess, ob_space, mes_space, hl_space, ac_space, n_env, 
                 n_steps, n_batch, reuse=False):
        
        super(ActorCriticPolicy, self).__init__(sess, ob_space, mes_space, 
             hl_space, ac_space, n_env, n_steps, n_batch, reuse=reuse)
        
        self.pdtype = make_proba_dist_type(ac_space)
        self.policy = None
        self.proba_distribution = None
        self.value_fn = None
        self.action = None
        self.deterministic_action = None

    def _setup_init(self):
        """Sets up the distributions, actions, and value."""
        with tf.variable_scope("output", reuse=True):
#            assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None, "failed in base_policies.py line 44"
            self.action = self.proba_distribution.sample()
            self.deterministic_action = self.proba_distribution.mode()
            self.neglogp = self.proba_distribution.neglogp(self.action)
            
            if isinstance(self.proba_distribution, DiagGaussianProbabilityDistribution):
                self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]
            self.value_flat = self.value_fn[:, 0]