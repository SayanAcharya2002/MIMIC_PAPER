from sklearn.cluster import KMeans
import numpy as np
import pickle
import gymnasium as gym
from gymnasium import spaces
import random
from stable_baselines3 import PPO
import os,time,torch


# with open('bestpol.pkl', 'rb') as file:
#   modl = pickle.load(file)
#   Qon = pickle.load(file)
#   physpol = pickle.load(file)
#   transitionr = pickle.load(file) 
#   transitionr2 = pickle.load(file)
#   R = pickle.load(file)
#   C:KMeans = pickle.load(file)
#   train = pickle.load(file)
#   qldata3train = pickle.load(file)
#   qldata3test = pickle.load(file)





class SepsisEnv(gym.Env):

    def __init__(self, C:KMeans,T,R,ncl=10,N_DISCRETE_ACTIONS=25,N_ATTRS=47):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.C=C
        self.T=T.copy()
        self.R=R.copy()
        self.ncl=ncl
        self.N_DISCRETE_ACTIONS = N_DISCRETE_ACTIONS
        self.N_ATTRS=N_ATTRS
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-10, high=+10,
                                            shape=(N_ATTRS,), dtype=np.float32)
        self.state_index=random.randint(0,self.ncl-1)
        self.alive_state=np.ones((N_ATTRS),dtype=np.float32)
        self.dead_state=np.zeros((N_ATTRS),dtype=np.float32)

    def step(self, action):
        if np.sum(self.T[self.state_index,:,action])==0:
          new_state_index=np.random.choice(list(range(self.ncl+2)))
        else:
          new_state_index=np.random.choice(list(range(self.ncl+2)),p=self.T[self.state_index,:,action])

        if new_state_index==self.ncl:
            self.state_index=new_state_index
            return self.dead_state.copy(),-100,True,False,{}
        elif new_state_index==self.ncl+1:
            self.state_index=new_state_index
            return self.alive_state.copy(),100,True,False,{}
        else:
            self.state_index=new_state_index
            return self.C.cluster_centers_[new_state_index].copy(),0,False,False,{}
        
        # return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.state_index=random.randint(0,self.ncl-1)
        observation=self.C.cluster_centers_[self.state_index].copy()
        info={}
        return observation, info

    def close(self):
        pass
    
    def get_all_actions(self,model):
        all_states=np.concatenate([self.C.cluster_centers_,self.dead_state.reshape(1,-1),self.alive_state.reshape(1,-1)],axis=0)
        # print(all_states.shape)
        vals=model.predict(all_states,deterministic=True)[0]
        # print(vals)
        return np.array(vals)


# T=np.random.rand(12,12,25)
# for i in range(12):
#   for k in range(25):
#     T[i,:,k]/=sum(T[i,:,k])

# env=SepsisEnv(C,T,R)

# models_dir = f"models/{int(time.time())}/"
# logdir = f"logs/{int(time.time())}/"

# if not os.path.exists(models_dir):
# 	os.makedirs(models_dir)

# if not os.path.exists(logdir):
# 	os.makedirs(logdir)


# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# TIMESTEPS = 100
# iters = 0

# iters += 1
# model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
# model.save(f"{models_dir}/{TIMESTEPS*iters}")

# vals=C.cluster_centers_[0:5]
# ans=model.predict(vals,deterministic=True)
# print(model.policy.get_distribution(torch.tensor(vals,device="cuda:0")).distribution.probs.detach().cpu().numpy())
# print(env.get_all_actions(model))