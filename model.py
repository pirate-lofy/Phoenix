import tensorflow as tf
import joblib

class Model:
    def __init__(self,policy,ob_img_space,ob_measure_space,ac_space,
                 n_batch_act,n_batch_critic,ent_coef,vf_coef,
                 max_grad_norm,frame_stack):
        
        sess=tf.get_default_session()
        
        actor = policy(sess, ob_img_space, ob_measure_space,ac_space)
        critic = policy(sess, ob_img_space, ob_measure_space,ac_space,reuse=True)
        
        '''-----------------------------------------'''
        A = critic.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = critic.pd.neglogp(A)
        entropy = tf.reduce_mean(critic.pd.entropy())
        
        vpred = critic.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(critic.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        vf_losses1 = tf.square(vpred - R)
        vf_losses2 = tf.square(vpredclipped - R)
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef
        '''-----------------------------------------'''
        
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)
        
        
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        
        def train(lr, cliprange, img_obs, measure_obs, returns, masks, actions, values, neglogpacs, states=None):

            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {critic.X:img_obs,critic.X_measurements:measure_obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[critic.S] = states
                td_map[critic.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
            
        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
        
            
        self.step = actor.step
        self.value = actor.value
        self.initial_state = actor.initial_state
        self.train = train
        self.critic = critic
        self.actor = actor
        self.save = save
        self.load = load
        
        tf.global_variables_initializer().run(session=sess)
        