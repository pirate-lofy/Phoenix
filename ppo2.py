import os
import time
import joblib
import numpy as np
import os.path as osp
import tensorflow as tf
from stable_baselines import logger
from collections import deque
from stable_baselines.common import explained_variance

class Model(object):
    def __init__(self, *, policy, ob_img_space, ob_measure_space, ac_space, nbatch_act, nbatch_train,
                nsteps, ent_coef, vf_coef, max_grad_norm, frame_stack=2):
        sess = tf.get_default_session()

        act_model = policy(sess, ob_img_space, ob_measure_space, frame_stack, ac_space, nbatch_act, 1, reuse=False)
        train_model = policy(sess, ob_img_space, ob_measure_space, frame_stack, ac_space, nbatch_train, nsteps, reuse=True)
        
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])
        CLIPRANGE = tf.placeholder(tf.float32, [])
        neglogpac = train_model.pd.neglogp(A)
        entropy = tf.reduce_mean(train_model.pd.entropy())

        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
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
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def train(lr, cliprange, img_obs, measure_obs, returns, masks, actions, values, neglogpacs, states=None):
            advs = returns - values
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            td_map = {train_model.X:img_obs,train_model.X_measurements:measure_obs, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']

        def save(save_path):
            ps = sess.run(params)
            joblib.dump(ps, save_path)

        def load(load_path):
            loaded_params = joblib.load(load_path)
            restores = []
            for p, loaded_p in zip(params, loaded_params):
                restores.append(p.assign(loaded_p))
            sess.run(restores)
            # If you want to load weights, also save/load observation scaling inside VecNormalize

        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.save = save
        self.load = load
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101

class Runner(object):

    def __init__(self, *, env, model, nsteps, gamma, lam, frame_stack):
        self.env = env
        self.model = model
        nenv = env.num_envs
        self.nc = env.observation_space.spaces[0].shape[-1]
        obs_image_shape = np.asarray(env.observation_space.spaces[0].shape)
        obs_image_shape[-1] = obs_image_shape[-1] * frame_stack
        obs_image_shape = tuple(obs_image_shape)
        self.obs_img = np.zeros((nenv,) + obs_image_shape, dtype=model.train_model.X.dtype.name)
        self.obs_measure = np.zeros((nenv,) + env.observation_space.spaces[1].shape, dtype=model.train_model.X_measurements.dtype.name)
        
        obs_img, obs_measure = env.reset()
        self.update_obs_image(obs_img, None)
        self.obs_measure[:] = obs_measure
        print(self.obs_img.shape,self.obs_measure.shape)
        self.gamma = gamma
        self.lam = lam
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]
     

    def update_obs_image(self, obs_img, dones):
        if dones is not None:
            self.obs_img *= (1 - dones.astype(np.uint8))
        
        self.obs_img = np.roll(self.obs_img, shift=-self.nc, axis=3)
        self.obs_img[:, :, :, -self.nc:] = obs_img[:, :, :]


    def run(self):
        mb_img_obs, mb_measure_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        for _ in range(self.nsteps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs_img, self.obs_measure, self.states, self.dones)
            mb_img_obs.append(self.obs_img.copy())
            mb_measure_obs.append(self.obs_measure.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            obs_img, obs_measure, rewards, self.dones, infos = self.env.step(actions)
            
            self.update_obs_image(obs_img, self.dones)
            self.obs_measure[:] = obs_measure
            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
            

            for done in self.dones:
                if done:
                    self.env.reset()
        #batch of steps to batch of rollouts
        mb_img_obs = np.asarray(mb_img_obs, dtype=self.obs_img.dtype)
        mb_measure_obs = np.asarray(mb_measure_obs, dtype=self.obs_measure.dtype)
        print(len(mb_rewards),mb_rewards)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs_img, self.obs_measure, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return (*map(sf01, (mb_img_obs, mb_measure_obs, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs)),
            mb_states, epinfos)
#        return (*map(sf01, (mb_img_obs, mb_measure_obs, mb_returns, mb_actions, mb_values, mb_neglogpacs)),
#            mb_states, epinfos)
# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    if s==(32,):return arr
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])

def constfn(val):
    def f(_):
        return val
    return f

def learn(*, policy, env, nsteps, total_timesteps, ent_coef, lr,
            vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
            log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
            save_interval=0):


    if isinstance(lr, float): lr = constfn(lr)
    else: assert callable(lr)
    if isinstance(cliprange, float): cliprange = constfn(cliprange)
    else: assert callable(cliprange)
    total_timesteps = int(total_timesteps)

    nenvs = 1
    ob_img_space = env.observation_space.spaces[0]
    ob_measure_space = env.observation_space.spaces[1]
    ac_space = env.action_space
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches

    make_model = lambda : Model(policy=policy, ob_img_space=ob_img_space, ob_measure_space=ob_measure_space, 
            ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train, nsteps=nsteps, 
            ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)

    if save_interval and logger.get_dir():
        import cloudpickle
        with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
            fh.write(cloudpickle.dumps(make_model))
    model = make_model()
    runner = Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam, frame_stack=2)

    epinfobuf = deque(maxlen=100)
    tfirststart = time.time()

    nupdates = total_timesteps//nbatch
    for update in range(1, nupdates+1):
        print('update no.', update)
        assert nbatch % nminibatches == 0
        nbatch_train = nbatch // nminibatches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / nupdates
        lrnow = lr(frac)
        cliprangenow = cliprange(frac)
        img_obs, measure_obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run() #pylint: disable=E0632
        epinfobuf.extend(epinfos)
        mblossvals = []
        if states is None: # nonrecurrent version
            print('nonrecurrent')
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (img_obs, measure_obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
#        else: # recurrent version
#            print('recurrent')
#            assert nenvs % nminibatches == 0
#            envsperbatch = nenvs // nminibatches
#            envinds = np.arange(nenvs)
#            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
#            envsperbatch = nbatch_train // nsteps
#            for _ in range(noptepochs):
#                np.random.shuffle(envinds)
#                for start in range(0, nenvs, envsperbatch):
#                    end = start + envsperbatch
#                    mbenvinds = envinds[start:end]
#                    mbflatinds = flatinds[mbenvinds].ravel()
#                    slices = (arr[mbflatinds] for arr in (img_obs, measure_obs, returns, masks, actions, values, neglogpacs))
#                    mbstates = states[mbenvinds]
#                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(nbatch / (tnow - tstart))
        if update % log_interval == 0 or update == 1:
#            ev = explained_variance(values, returns)
            logger.logkv("serial_timesteps", update*nsteps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*nbatch)
            logger.logkv("fps", fps)
#            logger.logkv("explained_variance", float(ev))
            logger.logkv('returnmean', safemean(returns))
            logger.logkv('time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir():
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i'%update)
            print('Saving to', savepath)
            model.save(savepath)
#    env.close()

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)
