# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:55:00 2021

@author: hugha
"""

import torch as T
import torch.nn.functional as F
from maddpg_made.Agent import Agent
import numpy as np

T.autograd.set_detect_anomaly(True)

class MADDPG:
    def __init__(self, env, alpha = 0.01, beta = 0.01, fc1 = 64, fc2 = 64,
               gamma = 0.99, tau = 0.01, chkpt_dir = 'testing_save/'):
        self.agents = []
        #self.n_agents = n_agents
        #self.n_actions = n_actions
        scenario = "adversary"
        chkpt_dir += scenario
        
        #self.agents = env.agents
        self.n_agents = len(env.agents)
        
        self.critic_obs_dims = 0
        for i in env.agents:
            self.critic_obs_dims += env.observation_spaces[i].shape[0]
        self.agents = []
        #for agent in env.agents:
         #   self.agents.append(Agent(env.observation_spaces[agent].shape[0], self.critic_obs_dims, 
          #                           env.action_spaces[agent].n, self.n_agents, agent, alpha = alpha, beta=beta,
           #                          chkpt_dir = chkpt_dir))
    
        self.agents = {}
        for agent in env.agents:
            self.agents[agent] = Agent(env.observation_spaces[agent].shape[0], self.critic_obs_dims, 
                                     env.action_spaces[agent].n, self.n_agents, agent, alpha = alpha, beta=beta,
                                     chkpt_dir = chkpt_dir)
    
    def save_checkpoint(self):
        print(' ... saving checkpoint ...')
        for agent in self.agents:
            agent.save_models()
            
    def load_checkpoint(self):
        print(' ... loading checkpoint ...')
        for agent in self.agents:
            agent.load_models()
        
    def choose_action(self, raw_obs, exploit):
        actions = {}
        for agent in raw_obs.keys():
            action = self.agents[agent].choose_action(raw_obs[agent])
            #print("action:",action)
            
            action_distribution = np.asarray(action)
            #print("action_dist:",action_distribution)
            
            if np.isnan(action_distribution).any() or exploit == True:
                #print("sum_action", sum(action))
                action = np.argmax(action)
                #print(action)
            else:
                action_options = [x for x in range(len(action))]
                #print("action_options:", action_options)
                action = np.random.choice(action_options, p = action)
                #print(action)
            
            actions[agent] = action
        
        return actions
    
    
    def learn(self, memory):
        if not memory.ready():
            return
        #sample them
        actor_states, states, actions, rewards, \
            actor_new_states, states_, dones = memory.sample_buffer()
            
        # turn them into tensors and incorporate cuda
        device = self.agents[list(self.agents.keys())[0]].actor.device
        for agent in self.agents:
            #states = T.tensor(states[agent], dtype=T.float).to(device)
            actions[agent] = T.tensor(actions[agent], dtype=T.float).to(device)
            actions[agent].requires_grad_(True)
            rewards[agent] = T.tensor(rewards[agent], dtype = T.double).to(device)
            #states_ = T.tensor(states_[agent], dtype=T.float).to(device)
            dones[agent] = T.tensor(dones[agent]).to(device)
        
        states = T.tensor(states, dtype=T.float).to(device)
        states.requires_grad_(True)
        states_ = T.tensor(states_, dtype=T.float).to(device)
        #organise them 
        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []
        
        for agent in self.agents:
            new_states = T.tensor(actor_new_states[agent],
                                  dtype=T.float).to(device)
            with T.no_grad():
                new_pi = self.agents[agent].target_actor.forward(new_states)
                all_agents_new_actions.append(new_pi)
            
                mu_states = T.tensor(actor_states[agent],
                                     dtype=T.float).to(device)
                pi = self.agents[agent].actor.forward(mu_states)
                all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent])
        
        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        
        for agent in self.agents:
            with T.no_grad():
                critic_value_ = self.agents[agent].target_critic.forward(states_, new_actions).flatten()
                #maybe because when the first agent is done its done?
                #a)
                #for agent in self.agents:
                    #critic_value_[dones[agent]] = 0.0
                #b)    
                critic_value_[dones[list(self.agents.keys())[0]]] = 0.0

                
                #I've added this line below
                critic_value_ = critic_value_.to(T.float64)
                target = rewards[agent] + self.agents[agent].gamma*critic_value_
            
            critic_value = self.agents[agent].critic.forward(states, old_actions).flatten()
            critic_value = critic_value.to(T.float64)
            
            #.detach() #:, agent_idx, detach has been added
            critic_loss = F.mse_loss(target, critic_value)
            self.agents[agent].critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph = True)
            self.agents[agent].critic.optimizer.step()
            
            self.agents[agent].actor.optimizer.zero_grad()
            actor_loss = -self.agents[agent].critic.forward(states, mu).flatten()
            actor_loss = actor_loss.mean()
            
            actor_loss.backward(retain_graph = True)
            self.agents[agent].actor.optimizer.step()
            

            
            self.agents[agent].update_network_parameters()

 