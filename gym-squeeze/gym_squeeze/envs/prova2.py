import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import ACKTR

# multiprocess environment
env = make_vec_env('gym_squeeze:squeeze-v0', n_envs=1)


model = ACKTR.load("acktr_squeeze_2")

obs = env.reset()
x=[]
rc=[]
sc=[]
rew=[]
done=[]
dones=False
i=0

while dones==False:
    i=i+1
    x.append(i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #obs=np.array(obs)
    print(obs)
    rc.append(obs[0,0:2])
    A=np.array([[obs[0,2],obs[0,3]],[obs[0,4],obs[0,5]]])
    rew.append(rewards)
    done.append(dones)
    print(dones,info)
    #print(type(obs),obs.shape)
    sc.append(A)
    #env.render(mode='human')

rc=np.array(rc)

plt.figure('prova')
plt.plot(x,rew)
plt.figure('provarc')
plt.plot(x,rc[:,0])
