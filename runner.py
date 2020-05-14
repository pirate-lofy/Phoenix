import numpy as np
class Runner:
    def __init__(self,env,model,n_steps,gamma,lam):
        self.env = env
        self.model = model
        self.lam = lam
        self.gamma=gamma
        self.n_steps = n_steps
        self.done=False
        
        obs_img, obs_measure,obs_hl = env.reset()
        self.obs_image,self.obs_measure,self.obs_hl=self.make_data_ready(obs_img.copy(),
                                                             obs_measure.copy(),
                                                             obs_hl.copy())



    def sf01(self,arr):
        # swap and then flatten axes 0 and 1 
        s = arr.shape
        if len(s)==1:return arr
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
    
    def make_data_ready(self,imgs,measures,hl):
        imgs=np.expand_dims(imgs,0)
        imgs=imgs.astype(self.model.critic.X.dtype.name)
        measures=np.expand_dims(measures,0)
        measures=measures.astype(self.model.critic.X_measurements.dtype.name)
        hl=np.expand_dims(hl,0)
        hl=hl.astype('float32')
        return imgs,measures,hl

    def run(self):
        img_obs, measure_obs, hl_obs,rewards, actions_list, values, dones, neglogpacs = [],[],[],[],[],[],[],[]

        # collect trasactions
        for stp in range(self.n_steps):
            actions, value, neglogpac = self.model.step(self.obs_image, self.obs_measure,self.obs_hl)
            
            # append current state's data
            img_obs.append(self.obs_image.copy())
            measure_obs.append(self.obs_measure.copy())
            hl_obs.append(self.obs_hl.copy())
            actions_list.append(actions)
            values.append(value)
            neglogpacs.append(neglogpac)
            dones.append(self.done)
            obs_img, obs_measure, obs_hl,reward, self.done, _ = self.env.step(actions)
#            
            rewards.append(reward)
            
            self.obs_image,self.obs_measure,self.obs_hl=self.make_data_ready(obs_img.copy(),
                                                                 obs_measure.copy(),
                                                                 obs_hl.copy())

            if self.done:
                self.env.reset()
          
        self.env.dead_command()
        
        #convert them to numpy array
        img_obs = np.asarray(img_obs, dtype=self.obs_image.dtype)
        measure_obs = np.asarray(measure_obs, dtype=self.obs_measure.dtype)
        hl_obs = np.asarray(hl_obs, dtype=self.obs_hl.dtype)
        rewards = np.asarray(rewards, dtype=np.float32)
        actions_list = np.asarray(actions_list)
        values = np.asarray(values, dtype=np.float32)
        neglogpacs = np.asarray(neglogpacs, dtype=np.float32)
        dones = np.asarray(dones, dtype=np.bool)
        
        # getting the v' for the last state
        # for the reversed operation
        last_values = self.model.value(self.obs_image, self.obs_measure,self.obs_hl)
     
        returns = np.zeros_like(rewards)
        advs = np.zeros_like(rewards)
        lastgaelam = 0
        
        # GAE
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                nextnonterminal = 1.0 - self.done
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            
            delta =rewards[t] + self.gamma * nextvalues * nextnonterminal -values[t]
            advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
       
        returns =advs +values
        g=map(self.sf01, (img_obs,measure_obs,hl_obs,returns,dones,actions_list,values,neglogpacs))
        g=tuple(g)
        return g
    
    