from abc import ABC, abstractmethod
import typing
from typing import Union, Optional, Any

import gym
import numpy as np

from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.common.vec_env import VecEnv

if typing.TYPE_CHECKING:
    from stable_baselines.common.base_class import BaseRLModel  # pytype: disable=pyi-error

class AbstractEnvRunner(ABC):
    def __init__(self, *, env: Union[gym.Env, VecEnv], model: 'BaseRLModel', n_steps: int):
        """
        Collect experience by running `n_steps` in the environment.
        Note: if this is a `VecEnv`, the total number of steps will
        be `n_steps * n_envs`.

        :param env: (Union[gym.Env, VecEnv]) The environment to learn from
        :param model: (BaseRLModel) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        """
        self.env = env
        self.model = model
        n_envs = env.num_envs
        
        self.batch_ob_shape = (n_envs * n_steps,) + env.observation_space.shape
        self.obs = np.zeros((n_envs,) + env.observation_space.shape, dtype=env.observation_space.dtype.name)
        self.batch_measures_shape = (n_envs * n_steps,) + env.measures_space.shape
        self.measures = np.zeros((n_envs,) + env.measures_space.shape, dtype=env.measures_space.dtype.name)
        self.batch_hl_command_shape = (n_envs * n_steps,) + env.hl_command_space.shape
        self.hl_command = np.zeros((n_envs,) + env.hl_command_space.shape, dtype=env.hl_command_space.dtype.name)
        self.obs[:], self.measures[:], self.hl_command[:] = env.reset()
        
        self.n_steps = n_steps
        self.states = model.initial_state
        self.dones = [False for _ in range(n_envs)]
        self.callback = None  # type: Optional[BaseCallback]
        self.continue_training = True
        self.n_envs = n_envs

    def run(self, callback: Optional[BaseCallback] = None) -> Any:
        """
        Collect experience.

        :param callback: (Optional[BaseCallback]) The callback that will be called
            at each environment step.
        """
        self.callback = callback
        self.continue_training = True
        return self._run()

    @abstractmethod
    def _run(self) -> Any:
        """
        This method must be overwritten by child class.
        """
        raise NotImplementedError



class Runner(AbstractEnvRunner):
    def __init__(self, env, model, n_steps, gamma, lam):
        """
        A runner to learn the policy of an environment for a model

        :param env: (Gym environment) The environment to learn from
        :param model: (Model) The model to learn
        :param n_steps: (int) The number of steps to run for each environment
        :param gamma: (float) Discount factor
        :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        """
        super().__init__(env=env, model=model, n_steps=n_steps)
        self.lam = lam
        self.gamma = gamma

    def _run(self):
        """
        Run a learning step of the model

        :return:
            - observations: (np.ndarray) the observations
            - rewards: (np.ndarray) the rewards
            - masks: (numpy bool) whether an episode is over or not
            - actions: (np.ndarray) the actions
            - values: (np.ndarray) the value function output
            - negative log probabilities: (np.ndarray)
            - states: (np.ndarray) the internal states of the recurrent policies
            - infos: (dict) the extra information of the model
        """
        # mb stands for minibatch
        mb_obs, mb_measures, mb_hl_commands, mb_rewards, mb_actions, mb_values, \
            mb_dones, mb_neglogpacs = [], [], [], [], [], [], [], []
        mb_states = self.states
        for _ in range(self.n_steps):
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.measures,
                                                                       self.hl_command,
                                                                       self.states, self.dones)
            
            mb_obs.append(self.obs.copy())
            mb_measures.append(self.measures.copy())
            mb_hl_commands.append(self.hl_command.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.env.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
            
            self.obs[:], self.measures[:], self.hl_command[:], rewards, self.dones, infos =\
                    self.env.step(clipped_actions)

            self.model.num_timesteps += self.n_envs

            if self.callback is not None:
                # Abort training early
                if self.callback.on_step() is False:
                    self.continue_training = False
                    # Return dummy values
                    return [None] * 9

            mb_rewards.append(rewards)
        
        # dead command to force the car to stop
        self.env.dead_command()
        
        # batch of steps to batch of rollouts
        mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_measures = np.asarray(mb_measures, dtype=self.measures.dtype)
        mb_hl_commands = np.asarray(mb_hl_commands, dtype=self.hl_command.dtype)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        
        last_values = self.model.value(self.obs, self.measures, self.hl_command,
                                       self.states, self.dones)
        # discount/bootstrap off value fn
        mb_advs = np.zeros_like(mb_rewards)
        true_reward = np.copy(mb_rewards)
        last_gae_lam = 0
        for step in reversed(range(self.n_steps)):
            if step == self.n_steps - 1:
                nextnonterminal = 1.0 - self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 - mb_dones[step + 1]
                nextvalues = mb_values[step + 1]
            delta = mb_rewards[step] + self.gamma * nextvalues * nextnonterminal - mb_values[step]
            mb_advs[step] = last_gae_lam = delta + self.gamma * self.lam * nextnonterminal * last_gae_lam
        mb_returns = mb_advs + mb_values

        mb_obs, mb_measures, mb_hl_commands, mb_returns, mb_dones, mb_actions, mb_values, mb_neglogpacs, true_reward = \
            map(swap_and_flatten, (mb_obs, mb_measures, mb_hl_commands, mb_returns, mb_dones, mb_actions, 
                                   mb_values, mb_neglogpacs, true_reward))

        return mb_obs, mb_measures, mb_hl_commands, mb_returns, mb_dones, mb_actions, \
                mb_values, mb_neglogpacs, mb_states, true_reward


# obs, returns, masks, actions, values, neglogpacs, states = runner.run()
def swap_and_flatten(arr):
    """
    swap and then flatten axes 0 and 1

    :param arr: (np.ndarray)
    :return: (np.ndarray)
    """
    shape = arr.shape
    return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
