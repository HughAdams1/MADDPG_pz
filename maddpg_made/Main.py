# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 08:55:25 2021

@author: hugha
"""

from make_env import make_env
import numpy as np
from maddpg_made.Maddpg import MADDPG
from maddpg_made.Buffer import MultiAgentReplayBuffer


def obs_list_to_state_vector(observation):
    state = np.array([])
    for obs in observation:
        state = np.concatenate([state, obs])
    return state

if __name__ == '__main__':
    #scenario = 'simple'
    scenario = 'simple_adversary'
    env = make_env(scenario)
    n_agents = env.n
    actor_dims = []
    for i in range(n_agents):
        actor_dims.append(env.observation_space[i].shape[0])
    critic_dims = sum(actor_dims)
    
    n_actions = env.action_space[0].n
    maddpg_agents = MADDPG(actor_dims, critic_dims, n_agents, n_actions,
                           scenario=scenario, fc1=64, fc2=64, alpha=0.01, 
                           beta=0.01, chkpt_dir='test_ceckpoints/') #C:/Users/hugha/Documents/multiagent-particle-envs/multiagent/scenarios
     
    memory = MultiAgentReplayBuffer(1000000, critic_dims, actor_dims, n_actions,
                                    n_agents, batch_size=1024)     
    
    PRINT_INTERVAL = 500
    N_GAMES = 3000        
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0
    
    if evaluate:
        maddpg_agents.load_checkpoint()
    
    for i in range(N_GAMES):  
        obs = env.reset()
        score = 0
        done = [False]*n_agents

        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
            actions = maddpg_agents.choose_action(obs)
            obs_, reward, done, info = env.step(actions)
            
            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)
            
            if episode_step>MAX_STEPS:
                done = [True]*n_agents
                
            memory.store_transition(obs, state, actions, reward, obs_, state_, done)
            
            if total_steps%100 ==0 and not evaluate:
                maddpg_agents.learn(memory)
                #maddpg_agents.learn_test(memory)
                
            obs = obs_
            
            score +=sum(reward)
            total_steps += 1
            episode_step += 1
        
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save_checkpoint()
                best_score = avg_score
        if i% PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))
                
     