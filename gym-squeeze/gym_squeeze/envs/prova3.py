import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.ddpg.policies import MlpPolicy, LnMlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

model_class = SAC  # works also with SAC, DDPG and TD3

env = gym.make('gym_squeeze:squeeze-v0')

# Available strategies (cf paper): future, final, episode, random
goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

# Wrap the model
model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
                                                verbose=1)
# Train the model
model.learn(1000)

model.save("./her_bit_env")

# WARNING: you must pass an env
# or wrap your environment with HERGoalEnvWrapper to use the predict method
model = HER.load('./her_bit_env', env=env)

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

