# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 13:12:58 2021

@author: hugha
"""
# trying to incorporate maddpg_made with petting_zoo

import numpy as np
from maddpg_action_taking_test import MADDPG
#from maddpg_made.Agent import Agent
from buffer_made import MultiAgentReplayBuffer
from pettingzoo.mpe import simple_adversary_v2
#from pettingzoo.mpe import simple_v2


def obs_list_to_state_vector(observation):
    state = np.array([])
    for agent in env.agents:
        state = np.concatenate([state, observation[agent]])
    return state

# scenario is just name for file for saving
#scenario = 'simple_adversary'
env = simple_adversary_v2.parallel_env()
env.reset()

critic_obs_dims = 0
for i in env.agents:
    critic_obs_dims += env.observation_spaces[i].shape[0]

n_actions = []
for i in env.agents:
    n_actions.append(env.action_spaces[i].n)

actor_obs_dims = []
for i in env.agents:
    actor_obs_dims.append(env.observation_spaces[i].shape[0])

maddpg_agents = MADDPG(env, fc1=64, fc2=64, alpha=0.01, 
                       beta=0.01, chkpt_dir='testing_save/') #C:/Users/hugha/Documents/multiagent-particle-envs/multiagent/scenarios
 
memory = MultiAgentReplayBuffer(env, max_size=1000000, batch_size=1024)   

########################## parallel API ################################

PRINT_INTERVAL = 500
N_GAMES = 5000        
MAX_STEPS = 50
total_steps = 0
score_history = []
evaluate = False
best_score = 0

if evaluate:
    maddpg_agents.load_checkpoint()

for i in range(N_GAMES):
    obs = env.reset()
    #done = env.dones
    done_list = [False for agent in env.agents]
    
    score = 0
    episode_step = 0
    
    while not any(done_list):
        #env.render(mode='human')
        actions = maddpg_agents.choose_action(obs, False)
        #print("the actions:", actions)
        obs_, reward, done, infos = env.step(actions)
        done_list = list(done.values())
        if episode_step > MAX_STEPS:
            done_list = [True] #this might flag an error, might have to move it to after
            #store transition, but that has to move anyway

        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)
        
        memory.store_transition(obs, state, actions, reward, obs_, state_, done)
        
        obs = obs_
        
        episode_step +=1
        total_steps +=1
        score += sum(reward.values()) # I have a feeling it collects the rewards
        
        #got to try and get it to learn now
        if total_steps%100 ==0:
            maddpg_agents.learn(memory)

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
        #if not evaluate:
         #   if avg_score > best_score:
          #      maddpg_agents.save_checkpoint()
           #     best_score = avg_score
    if i% PRINT_INTERVAL == 0 and i > 0:
        print('episode', i, 'average score {:.1f}'.format(avg_score))
#env.close()
#print(score)

for i in range(N_GAMES):
    obs = env.reset()
    #done = env.dones
    done_list = [False for agent in env.agents]
    
    score = 0
    episode_step = 0
    
    while not any(done_list):
        #env.render(mode='human')
        actions = maddpg_agents.choose_action(obs, True)
        #print("the actions:", actions)
        obs_, reward, done, infos = env.step(actions)
        done_list = list(done.values())
        if episode_step > MAX_STEPS:
            done_list = [True] #this might flag an error, might have to move it to after
            #store transition, but that has to move anyway

        state = obs_list_to_state_vector(obs)
        state_ = obs_list_to_state_vector(obs_)
        
        memory.store_transition(obs, state, actions, reward, obs_, state_, done)
        
        obs = obs_
        
        episode_step +=1
        total_steps +=1
        score += sum(reward.values()) # I have a feeling it collects the rewards
        
        #got to try and get it to learn now
        if total_steps%100 ==0:
            maddpg_agents.learn(memory)

    score_history.append(score)
    avg_score = np.mean(score_history[-100:])
        #if not evaluate:
         #   if avg_score > best_score:
          #      maddpg_agents.save_checkpoint()
           #     best_score = avg_score
    if i% PRINT_INTERVAL == 0 and i > 0:
        print('episode', i, 'average score {:.1f}'.format(avg_score))