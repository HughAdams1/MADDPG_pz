# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 09:28:11 2021

@author: hugha
"""

import numpy as np

class MultiAgentReplayBuffer:
    def __init__(self, env, max_size, batch_size):
        
        self.mem_size = max_size
        self.mem_cntr = 0
        
        self.agents = env.agents
        self.n_agents = len(env.agents)
        self.batch_size = batch_size
        
        self.critic_obs_dims = 0
        for i in env.agents:
            self.critic_obs_dims += env.observation_spaces[i].shape[0]
        #difficulty in accomodating the different memory sizes, actually problem easy
        self.state_memory = np.zeros((self.mem_size, self.critic_obs_dims))
        self.new_state_memory = np.zeros((self.mem_size, self.critic_obs_dims))
        
        self.init_actor_memory(env)
        
    def init_actor_memory(self, env):
        #we want a seperate function for this bc we r working with lists of numpy arrays
        #and its hard to do memory counter with lists. Need a seperate function to zero out
        #max memory size for actor and critic
        self.actor_state_memory = {}
        self.actor_new_state_memory = {}
        self.actor_action_memory = {}
        self.reward_memory = {}
        self.terminal_memory = {}
        
        for agent in env.agents:
            self.actor_state_memory[agent] = np.zeros((self.mem_size, env.observation_spaces[agent].shape[0]))
            self.actor_new_state_memory[agent] = np.zeros((self.mem_size, env.observation_spaces[agent].shape[0]))
            self.actor_action_memory[agent] = np.zeros((self.mem_size, env.action_spaces[agent].n))
            self.reward_memory[agent] = np.zeros(self.mem_size)
            self.terminal_memory[agent] = np.zeros(self.mem_size, dtype = bool)

            
            
            
        
    def store_transition(self, raw_obs, state, action, reward,
                        raw_obs_, state_, done):
        
        #position of the first available memory
        index = self.mem_cntr % self.mem_size
        
        for agent in raw_obs.keys():
            self.actor_state_memory[agent][index] = raw_obs[agent]
            self.actor_new_state_memory[agent][index]  = raw_obs_[agent]
            self.actor_action_memory[agent][index] = action[agent]
            self.reward_memory[agent][index] = reward[agent]
            self.terminal_memory[agent][index] = done[agent]
            
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        
        self.mem_cntr += 1
    
    def sample_buffer(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        states = self.state_memory[batch]
        #rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        #terminal = self.terminal_memory[batch]
        
        actor_states = {}
        actor_new_states = {}
        actions = {}
        rewards = {}
        terminal = {}
        
        for agent in self.agents:
            actor_states[agent] = self.actor_state_memory[agent][batch]
            actor_new_states[agent] = self.actor_new_state_memory[agent][batch]
            actions[agent] = self.actor_action_memory[agent][batch]
            rewards[agent] = self.reward_memory[agent][batch]
            terminal[agent] = self.terminal_memory[agent][batch]
            
        return actor_states, states, actions, rewards, \
            actor_new_states, states_, terminal
            
    def ready(self):
        if self.mem_cntr >= self.batch_size:
            return True
        return False