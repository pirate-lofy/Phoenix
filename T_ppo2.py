
from stable_baselines import logger
import os.path as osp
import time
import numpy as np
import os


def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def learn(model,runner,n_epochs,n_steps,n_min_patches,n_opt_epochs,n_batch,
          clip_range,save_each,log_interval,lr):
            
#    time_first_start = time.time()
    time_first_start=time.time()
    n_updates = n_epochs//n_batch
    
    for update in range(1, n_updates+1):
        assert n_batch % n_min_patches == 0
        
        n_batch_critic = n_batch // n_min_patches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / n_updates
        
        lrnow = lr(frac)
        cliprangenow = clip_range(frac)
        
        img_obs, measure_obs, returns, masks, actions_list, values, neglogpacs = runner.run() #pylint: disable=E0632

        mblossvals = []
        
        # recurrent and non-recurrent
        inds = np.arange(n_batch)
        for opt in range(n_opt_epochs):
            np.random.shuffle(inds)
            
            for start in range(0, n_batch, n_batch_critic):
                end = start + n_batch_critic
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (img_obs, measure_obs, returns, masks, actions_list, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        
        
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps = int(n_batch / (tnow - tstart))
        
        # log to print results
        if update % log_interval == 0 or update == 1:
            logger.logkv("serial_timesteps", update*n_steps)
            logger.logkv("nupdates", update)
            logger.logkv("total_timesteps", update*n_batch)
            logger.logkv("fps", fps)
            logger.logkv('returnmean', safemean(returns))
            logger.logkv('time_elapsed', tnow - time_first_start)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv(lossname, lossval)
            logger.dumpkvs()
        
        # save model
        if save_each and (update % save_each == 0 or update == 1) and logger.get_dir():
            savepath=os.path.join(os.getcwd(),'checkpoints')
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            print('Saving to', savepath)
            model.save(savepath)
                