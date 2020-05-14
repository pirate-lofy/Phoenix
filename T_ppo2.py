from baselines import logger
import time
import numpy as np
import os
import shutil
from colorama import Fore
import subprocess as sp

def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)

def learn(model,runner,n_epochs,n_steps,n_min_patches,n_opt_epochs,n_batch,
          clip_range,save_each,log_interval,lr,i=0):
    
    _ = sp.call('clear',shell=True)
    
    print(Fore.GREEN+'PPO2 log: Training has started....'+Fore.WHITE)
                

    time_first_start=time.time()
    n_updates = n_epochs//n_batch
    e_time=50
    for update in range(i+1, n_updates+1):
        s=time.time()
        assert n_batch % n_min_patches == 0
        n_batch_critic = n_batch // n_min_patches
        tstart = time.time()
        frac = 1.0 - (update - 1.0) / n_updates
        
        lrnow = lr(frac)
        cliprangenow = clip_range(frac)

        # step 1 in the algorithm
        # collect N transactions
        print(Fore.GREEN+'PPO2 log: runner cycle'+Fore.WHITE)
        img_obs, measure_obs, returns, masks, actions_list, values, neglogpacs = runner.run() #pylint: disable=E0632

        mblossvals = []
        
        # recurrent and non-recurrent
        inds = np.arange(n_batch)
        
        # step 3,4
        # for epochs
        print(Fore.GREEN+'PPO2 log: training cycle'+Fore.WHITE)
        for opt in range(n_opt_epochs):
            np.random.shuffle(inds)
            
            
            # get mini patches
            for start in range(0, n_batch, n_batch_critic):
                end = start + n_batch_critic
                mbinds = inds[start:end]
                slices = (arr[mbinds] for arr in (img_obs, measure_obs, returns, masks, actions_list, values, neglogpacs))
                mblossvals.append(model.train(lrnow, cliprangenow, *slices))
                #print(Fore.GREEN+'PPO2 log: logging summaries'+Fore.WHITE)
                #model.log(rond)
                #rond+=1
            
            if time.time()>=s+e_time:
                print(Fore.GREEN+'PPO2 log: Exceeded the full episode time'+Fore.WHITE)
                s=time.time()
                runner.env.reset()
        lossvals = np.mean(mblossvals, axis=0)
        tnow = time.time()
        fps=n_batch/(tnow-tstart)
        
        # log to print results
        if update % log_interval == 0 or update == 1:
            logger.logkv("epoch", update)
            logger.logkv("serial_timesteps", update*n_steps)
            logger.logkv("total_timesteps", update*n_batch)
            logger.logkv("fps", fps)
            logger.logkv('returnmean', safemean(returns))
            logger.logkv('time_elapsed', tnow - time_first_start)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/'+lossname, lossval)
            logger.dumpkvs()
            
            
        # save model
        # if save_each and (update % save_each == 0 or update == 1):
        #     savepath=os.path.join(os.getcwd(),'checkpoints')
        #     if update==1 and os.path.isdir(savepath):
        #         shutil.rmtree(savepath)
        #     if not os.path.exists(savepath):
        #         os.mkdir(savepath)
        #     savepath = os.path.join(savepath,update)
        #     print('PPO2 log: Saving to', savepath)
        #     model.save(savepath)
            
        # save model, colab version
        if save_each and (update%save_each==0 or update==1):
            savepath='/home/colab/Desktop/checkpoints'
            if not os.path.exists(savepath):
                os.mkdir(savepath)
            filepath = savepath+'/'+str(update)
            print('PPO@ log: Saving to',filepath)
            model.save(filepath)
        
        
    print(Fore.Green+'PPO2 log: Training has ended.'+Fore.WHITE)