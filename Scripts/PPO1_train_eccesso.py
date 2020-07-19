import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1
from stable_baselines.common.policies import FeedForwardPolicy, register_policy


class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[32,64,64,dict(pi=[32,64,64,32],
                                                          vf=[32,64,64,32])],
                                           feature_extraction="mlp")
        

# il modello PPO1_squeeze e successivi sono allenati con rew -h e con ipotesi di u lineare (le azioni sono K)
env = gym.make('gym_squeeze:squeeze-v0') #la roba tra qui e del model compreso può essere commentata se si vuole solo testare il modello
#model=PPO1(CustomPolicy, env, verbose=1,entcoeff=0.02, optim_batchsize=256,optim_stepsize=1e-3, tensorboard_log='PPO1')
model=PPO1.load("PPO1_squeeze_custom6",tensorboard_log='PPO1')
model.set_env(env)
model.learn(total_timesteps=2500000,tb_log_name='ppo1__custom')
model.save("PPO1_squeeze_custom6")

