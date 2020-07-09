import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import TRPO

env = gym.make('gym_squeeze:squeeze-v0')

model = TRPO(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=10000)
model.save("trpo_squeeze")

del model # remove to demonstrate saving and loading

model = TRPO.load("trpo_squeeze")

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
    rc.append(obs[0:2])
    A=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
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
