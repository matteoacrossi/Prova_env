import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import numpy as np
from scipy.linalg import sqrtm
import math

#definizione dei parametri (ributto qui tutti i conti delle matrici che servono per il calcolo)
k=1 #loss rate
eta=1 #eta della misura
X=k*0.49999 #accoppiamento hamiltoniana del sistema qui l'ho impostato sul valore 'critico'
              
#setto sigma ambiente Sb e sigma misura Sm
Sb=np.identity(2)
z=1e300 #qui abbozzo il lim per z a infinito della sigma di squeezing
Sm=np.array([[1/z,0],[0,z]])
              
#canale rumoroso per la misura con efficenza eta
Xs=np.identity(2)*(eta**(-0.5))
Ys=(eta**(-1)-1)*np.identity(2)
Sm=Xs.dot(Sm.dot(Xs.T))+Ys
              
#matrice simplettica
Sy=np.array([[0,1],[-1,0]])
              
#matrice dell'hamiltoniana di Squeezing
Hs=-X*np.array([[0,1],[1,0]]) 
              
#imposto i blocchi per la matrice Hc anche se mi sa non serve
zero=np.zeros([2,2])
C=(k**0.5)*Sy.T
              
              
#matrice A dell'evoluzione
A=Sy.dot(Hs)+0.5*Sy.dot(C.dot(Sy.dot(C.T)))
                    
#matrice di drift
D=Sy.dot(C.dot(Sb.dot(C.T.dot(Sy.T))))
              
#imposto la matrice 1/(sigma ambiente+sigma misura)^0.5
SIGMA=sqrtm(np.linalg.inv(Sb+Sm))
              
#imposto le matrici di dinamica monitorata (ho lasciato sotto commento il caso perfetto, ovvero quello dove il limite per z a infinito è preso esatto)
B=C.dot(Sy.dot(SIGMA))#np.array([ [-((eta*k)**0.5),0],[0,0] ] ) #
E=Sy.dot(C.dot(Sb.dot(SIGMA)))#B#
      
#queste servono per il feedback
l=1 
F=np.array([[l,0],[0,l]])
q=1e-4
Q=q*np.identity(2)
             
#imposto la sigmac steady state, per definire uno stop
a=(k-2*X)/k
b=k/(k-2*X)
sigmacss=np.array([[a,0],[0,b]])
eps=1e-8 #serve tipo definizione di limite per dire quando siamo in steady state
dt=1e-4

class SqueezeEnv(gym.Env):
      
      metadata = {'render.modes': ['human']} #non so bene a che serva ma per ora lo tengo
      #provo a definire degli attributi che dovrebbero essere le cose che vanno tenute in memoria in esecuzione
      momento_primo=np.ones(2)
      matrice_covarianza=np.identity(2)
      current_reward=0
      Done=False
      
      
      

      def __init__(self):
              
              super(SqueezeEnv, self).__init__() #questo non ho capito a che serve ma lo tengo
              
              #tentativo di dare spazi di azioni e osservazioni, per come lo vorrei fare lo spazio delle azioni è un float da -1 a 1 per u[0] e un altro analogo per u[1]
              #mentre lo spazio delle osservazioni è una matrice 3x2 dove la prima riga rende il momento primo e le altre due righe sono la matrice di covarianza. Per ora non ho idea se ho capito come funziona la notazione
              self.action_space = spaces.Box(
                  low=-1 , high=1,shape=(1,2), dtype=np.float16)    
              
              self.observation_space = spaces.Box(
                  low=-np.inf, high=np.inf, shape=(3, 2), dtype=np.float16)
              
              
      
      def step(self, action):
            
              
          
              #aggiorno il momento primo dello stato d'ambiente 
              rbcm=np.zeros(2)
              rc=self.momento_primo
              rbcm=rbcm+Sy.dot(C.T.dot(rc))*(dt**0.5)
              
              rm2=np.random.multivariate_normal(rbcm, (Sb+Sm)/2)
              dwm=((SIGMA).dot(rm2-rbcm))*(dt**0.5)
              
              #momento primo con feedback
              sc=self.matrice_covarianza
              u=action
              drc=A.dot(rc)*dt+(2**(-0.5))*(E-sc.dot(B)).dot(dwm)+F.dot(u)*dt
              rc=rc+drc
              
              #matrice di covarianza
              sc=sc+dt*((A.dot(sc)+sc.dot(A.T)+D)-(E-sc.dot(B)).dot((E-sc.dot(B)).T))
              
              #funzione costo
              h=sc[0,0]+rc[0]**2 + u.T.dot(Q.dot(u)) #fatta a mente ma mi sembra venga così per la P 1,0,0,0
              self.current_reward=-h
                  
              #provo a dare un criterio per smettere dopo un po' che siamo abbastanza vicini allo steady state
              distance=0
              for i in range(0,2):
                  for j in range(0,2):
                      distance=distance+(sc[i,j]-sigmacss[i,j])**2
              if distance<=eps:
                  self.Done=True
                  
              self.momento_primo=rc
              self.matrice_covarianza=sc
              output=np.array([[rc[0],rc[1]],[sc[0,0],sc[0,1]],[sc[1,0],sc[1,1]]])
              return output , self.current_reward , self.Done 
                      
      def reset(self):
                            
              #reinizializzo delle cose
              #parto da punti diversi ogni volta e vediamo
              self.momento_primo=np.array([rand.uniform(-1,1),rand.uniform(-1,1)]) 
              d=rand.uniform(0,2) 
              self.matrice_covarianza=np.array([[d,0],[0,d]])
              self.Done=False
              rc=self.momento_primo
              sc=self.matrice_covarianza
              output=np.array([[rc[0],rc[1]],[sc[0,0],sc[0,1]],[sc[1,0],sc[1,1]]])
              return output, self.Done
          
      
      
      def render(self,mode='human'):
              print('r:'self.momento_primo,'-h:'self.current_reward)
    
