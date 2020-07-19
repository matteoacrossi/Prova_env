import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import TRPO, ACKTR, A2C, SAC, PPO1
import math
# multiprocess environment
env = gym.make('gym_squeeze:squeeze-v0')
#model=PPO1(MlpPolicy,env,gamma=0.999,timesteps_per_actorbatch=256,verbose=1)
#model=PPO1.load("quello_quasi_buono")
#model.set_env(env)
#model.learn(total_timesteps=1000000)
model=PPO1.load("quello_quasi_buono_squeeze_2")
obs = env.reset()
x=[]
rc=[]
sc=[]
rew=[]
done=[]
dones=False
i=0

rc2=[]
sc2=[]
rew2=[]
done2=[]
obs2=obs
dones=False
i=0
diff=[]
#while dones==False:
for i in range(0,int(1e5)):
    i=i+1
    x.append(i)
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    #obs=np.array(obs)
    #print(obs)
    rc.append(obs[0:2])
    A=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
    rew.append(rewards)
    done.append(dones)
    #print(dones,info)
    #print(type(obs),obs.shape)
    sc.append(A)
    #env.render(mode='human')
    
    action2, _states2 = model.predict(obs2)
    obs2, rewards2, dones2, info2 = env.step_agent()
    #obs=np.array(obs)
    #print(obs2)
    rc2.append(obs2[0:2])
    A2=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
    rew2.append(rewards2)
    done2.append(dones2)
    #print(dones2,info2)
    #print(type(obs),obs.shape)
    sc2.append(A2)
    #env.render(mode='human')
    diff.append(-(rewards2-rewards))

rc=np.array(rc)

plt.figure('prova')
plt.plot(x,rew)
plt.figure('provarc')
plt.plot(x,rc[:,0])
plt.figure('rew_diff')
plt.scatter(x,diff,s=0.3,c='red')

rc2=np.array(rc2)

plt.figure('prova')
plt.plot(x,rew2)
plt.figure('provarc')
plt.plot(x,rc2[:,0])

