import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines import TD3,DDPG
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

env = gym.make('gym_squeeze:squeeze-v0')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.1, adoption_coefficient=1.01)
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.1) * np.ones(n_actions))

model = DDPG(LnMlpPolicy, env, verbose=1, param_noise=param_noise, action_noise=action_noise,tensorboard_log="./ppo1_squeeze_tensorboard_1/",full_tensorboard_log=True)
model.learn(total_timesteps=10000000)
model.save("ddpg_squeeze")

del model # remove to demonstrate saving and loading

model = DDPG.load("ddpg_squeeze")



obs = env.reset()
x=[]
rc=[]
sc=[]
rew=[]
done=[]

rc2=[]
sc2=[]
rew2=[]
done2=[]

dones=False
i=0

while dones==False:
    i=i+1
    x.append(i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #obs=np.array(obs)
    print(obs)
    rc.append(obs[0:2])
    A=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
    rew.append(rewards)
    done.append(dones)
    print(dones,info)
    #print(type(obs),obs.shape)
    sc.append(A)
    #env.render(mode='human')
    
    action2, _states2 = model.predict(obs)
    obs2, rewards2, dones2, info2 = env.step_agent()
    #obs=np.array(obs)
    print(obs2)
    rc2.append(obs2[0:2])
    A2=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
    rew2.append(rewards2)
    done2.append(dones2)
    print(dones2,info2)
    #print(type(obs),obs.shape)
    sc2.append(A2)
    #env.render(mode='human')

rc=np.array(rc)

plt.figure('prova')
plt.plot(x,rew)
plt.figure('provarc')
plt.plot(x,rc[:,0])

rc2=np.array(rc2)

plt.figure('prova')
plt.plot(x,rew2)
plt.figure('provarc')
plt.plot(x,rc2[:,0])



