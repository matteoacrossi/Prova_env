import gym
import numpy as np
from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy
from stable_baselines import PPO1
# il modello PPO1_squeeze e successivi sono allenati con rew -h e con ipotesi di u lineare (le azioni sono K)
env = gym.make('gym_squeeze:squeeze-v0') #la roba tra qui e del model compreso può essere commentata se si vuole solo testare il modello

model=PPO1.load("PPO1_squeeze_custom6")
#C:/Users/aleef/Documents/GitHub/Prova_env/modelli/PPO1_squeeze_custom6

def media_cammini(N,steps,env,model):
    obs = env.reset()
    x=[]
    rc=[]
    rew=[]
    
    rc2=[]
    rew2=[]
    eccesso2=[] 
    eccesso1=[]
    rtot=np.zeros((steps,2))
    rtot2=np.zeros((steps,2))
    rewtot=np.zeros(steps)
    rewtot2=np.zeros(steps)
    sctot=np.zeros((steps,2,2))
    sctot2=np.zeros((steps,2,2))
    sc=[]
    sc2=[]
    
    #calcolo della Ex con media su estrazioni
    
    for j in range(0,N): #cammini
    
      for k in range(0,steps): #steps di integrazione
      
        print(j,k)
        #faccio l'asse x al primo ciclo
        if j==0:
            x.append(k)
            #tiro fuori le azioni dal modello
        action, _states = model.predict(obs)
        #print(action)
        #tiro fuori osservazioni, reward e altro dopo ogni step dando in pasto le azioni del modello all'agente
        obs, rewards, dones, info = env.step(action)
        
        #print(info)
        
        #salvo i momenti primi
        rc.append(obs[0:2])
        #salvo le rewards
        rew.append(rewards)
        #salvo le covarianze
        if j==0:
            A=np.array([[obs[2],obs[3]],[obs[4],obs[5]]])
            sc.append(A)
      
        #qui faccio lo stesso per l'agente imparato (nota che sotto non do in pasto le azioni perchè non sono usate)
        #action2, _states2 = model.predict(obs2)
        obs2, rewards2, dones2, info2 = env.optimal_agent()
        
        #print(info2)
        
        rc2.append(obs2[0:2])
        rew2.append(rewards2)
        if j==0:
            A2=np.array([[obs2[2],obs2[3]],[obs2[4],obs2[5]]])
            sc2.append(A2)
        
      #faccio cose per il formato
      rc=np.array(rc)
      rc2=np.array(rc2)
      rew=np.array(rew)
      rew2=np.array(rew2)
  
      #faccio la somma per poi fare la media di tutto
      rtot=rtot+rc
      rtot2=rtot2+rc2
      rewtot=rewtot+rew
      rewtot2=rewtot2+rew2
      #sctot=sctot+sc
      #sctot2=sctot2+sc
    
      #reinizializzo
      rc=[]
      rew=[]
      rc2=[]
      rew2=[]
      obs=env.reset()
    
    
    #faccio la media sulle realizzazioni
    rmean=rtot/N
    rmean2=rtot2/N
    rewmean=rewtot/N
    rewmean2=rewtot2/N
    
    
    #calcolo gli eccessi nei due casi con la formula del prodotto esterno della media
    for i in range(0,steps):
        eccesso1.append((np.outer(rmean[i,:],rmean[i,:].T)+np.outer(rmean[i,:].T,rmean[i,:]))*0.5)
        
    for i in range(0,steps):
        eccesso2.append((np.outer(rmean2[i,:],rmean2[i,:].T)+np.outer(rmean2[i,:].T,rmean2[i,:]))*0.5)
    
    return x,rmean,rmean2,rewmean,rewmean2,sc,sc2,eccesso1,eccesso2

N=1000
steps=int(1e4)

x,rmean,rmean2,rewmean,rewmean2,sc,sc2,eccesso1,eccesso2=media_cammini(N,steps,env,model)

#roba da plot
sc=np.array(sc)
sc2=np.array(sc2)
eccesso1=np.array(eccesso1)
eccesso2=np.array(eccesso2)
su=sc+eccesso1
su2=sc2+eccesso2
su=np.array(su)
su2=np.array(su2)



plt.figure('prova')
fig, ax = plt.subplots()
ax.plot(x,rewmean,'-.b',label='rew_agent')
ax.plot(x,rewmean2,'-.m',label='rew_opt')
leg = ax.legend()

plt.figure('provarc')
fig, ax = plt.subplots()
ax.plot(x,rmean[:,0],'-b',label='rc0_agent')
ax.plot(x,rmean2[:,0],'-m',label='rc0_opt')
leg = ax.legend()

plt.figure('sigma00mean')
fig, ax = plt.subplots()
ax.plot(x,su[:,0,0],'-.b',label='su_agent')
ax.plot(x,su2[:,0,0],'-.m',label='su_opt')
leg = ax.legend()

plt.figure('excess')
fig, ax = plt.subplots()
ax.plot(x,eccesso1[:,0,0],'-.b',label='exc_agent')
ax.plot(x,eccesso2[:,0,0],'-.m',label='exc_opt')
leg = ax.legend()

plt.figure('phasesp')
fig, ax = plt.subplots()
ax.plot(rmean[:,0],rmean[:,1],'-.b',label='ph_space_agent')
ax.plot(rmean2[:,0],rmean2[:,1],'-.m',label='ph_space_opt')
leg = ax.legend()