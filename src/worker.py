#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alain
"""
from __future__ import division
from __future__ import print_function

#lib
import torch
import vizdoom as vzd
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from copy import deepcopy
from util_functions.vizdoom_utils import ViZDoomMap
from util_functions.utils import *

from src.runner import Runner

class Worker():
    def __init__(self,
                worker_id,
                worker_end,
                num_parallel_envs,
                agent,
                available_actions,
                resolution=(42,42),
                max_episode_len=2100,
                use_stacked_frames=1,
                stack_size=4,
                stack_mode=0,
                nsteps = 100,
                runners=None,
                runners_ids=None,
                runners_parent_conn=None,
                use_gae=1,
                use_cuda=False,
                device_id=0,
                rnd_lr=0.0001,
                curiosity_type='centralized',
                use_centralized_critic=0,
                lmbda=.95,
                gamma=.99,
                gamma_int=.99,
                ext_coef=1.,
                int_coef=0.,
                exploration_exploitation_tradeoff_adaptative = 1,
                episode_counter = 0,
                intrinsic_rew_episodic = False,
                constant_std_for_normalization = 0,
                normalization_std = 1000,
                episodes_pre_train = 5,
                use_lstm = 0,
                centralized_curiosity_output_size = 5
                ):

        self.use_lstm = use_lstm

        self.agent = agent
        self.worker_id = worker_id
        self.worker_end = worker_end
        self.nsteps = nsteps
        self.num_envs = num_parallel_envs
        self.active_runners = np.ones(self.num_envs).astype('bool')

        self.use_gae = use_gae
        self.use_cuda = use_cuda
        self.device = torch.device('cuda:'+str(device_id) if use_cuda else 'cpu')

        # runners
        self.runners = runners
        self.runners_ids = runners_ids
        self.runner_parent_connections = runners_parent_conn

        # train params
        self.resolution = resolution
        self.stack_size = stack_size
        self.available_actions = available_actions
        self.use_stacked_frames = use_stacked_frames
        self.max_episode_len = max_episode_len
        self.episode_counter = episode_counter

        # GAE
        self.gamma = gamma # for extrinsic
        self.gamma_int = gamma_int # for intrinsic
        self.lmbda = lmbda
        self.intrinsic_rew_episodic = intrinsic_rew_episodic

        # tradeoff/balance exploration
        self.exploration_exploitation_tradeoff_adaptative = exploration_exploitation_tradeoff_adaptative
        self.ext_coef= ext_coef
        self.int_coef= int_coef
        self.offset = 0

        # intrinsic module related
        self.curiosity_type = curiosity_type
        self.obs_rms = RunningMeanStd(shape=(42,42))

        # boolean for using centralized critic
        self.use_centralized_critic = use_centralized_critic

        # number of times a room has been used as initialization
        self.room_init_freq = {k: 0 for k in range(10,28)}
        # number of steps required by agent when being initialized in that room
        self.room_steps_for_done = {k: [] for k in range(10,28)}
        self.trajectories = {k: [] for k in range(10,28)}
        self.score_window = deque([0],maxlen=100)

        # training mode/loop init
        self.train()

    def calculate_ext_int_coefs(self):
        if self.curiosity_type == 'none':
            self.int_coef = 0.0
            self.ext_coef = 1.0
            
        # if adaptive/decaying is desired... (example)
        elif self.exploration_exploitation_tradeoff_adaptative:
            if self.episode_counter <= 1000:
                self.ext_coef = 2.0
                self.int_coef = 1.0
            elif self.episode_counter <= 2000:
                self.ext_coef = 3.0
                self.int_coef = 1.0
            elif self.episode_counter <= 3000:
                self.ext_coef = 4.0
                self.int_coef = 1.0
            elif self.episode_counter <= 4000:
                self.ext_coef = 5.0
                self.int_coef = 1.0
            else:
                self.ext_coef = 10.0
                self.int_coef = 1.0

        # manual stop of curiosity
        else:
            if self.episode_counter > 3000:
                self.ext_coef = 100.0
                self.int_coef = 1.0


    def _pre_train(self,episodes_pre_train,action_based=False):
        """
            Pre-training for observation standarization parameters (mean,std)
        """

        print('Init pre-train...')

        for pre_train in range(episodes_pre_train):
            print('W{} episode {}'.format(self.worker_id,pre_train+1))
            for parent_conn in self.runner_parent_connections:
                parent_conn.send(('pre-train',0))
                # for parent_conn in self.runner_parent_connections:
                cmd, args = parent_conn.recv()
                if cmd == 'ack_pre-train':
                    s, a, r, ns, d = args
                    if action_based:
                        s = np.array(s)
                        self.obs_rms.update_from_moments(np.mean(s,axis=0),np.var(s,axis=0),s.shape[0])
                    else:
                        ns = np.array(ns)
                        self.obs_rms.update_from_moments(np.mean(ns,axis=0),np.var(ns,axis=0),ns.shape[0])

        print('\nFin pre-train')

    def train(self):
        while True:
            cmd, args = self.worker_end.recv()

            if cmd == 'pre-train':
                episodes_pre_train,action_based = args
                self._pre_train(episodes_pre_train,action_based)
                self.worker_end.send(('ack-pre-train',self.obs_rms)) # return to main; in the case of centralized curiosity it is needed

            elif cmd == 'new_episode':

                # update episode_counter
                self.episode_counter += 1

                #steps counter
                self.steps_in_episode = [0 for _ in range(len(self.runners))]

                # set environment episode status to False
                self.episode_finished = np.zeros(self.num_envs).astype('bool')
                self.active_runners = np.ones(self.num_envs).astype('bool')
                self.rooms = {}
                self.score = {}

                #at individual critic init the agents hidden state (in centralized setting, this is done at main)
                self.input_hidden = []
                if self.use_lstm:
                    for _ in range(len(self.runners)):
                        self.input_hidden.append(self.agent.init_lstm_hidden())
                else:
                    for _ in range(len(self.runners)):
                        self.input_hidden.append((torch.zeros(1),torch.zeros(1))) # avoid erros if lstm not used

                # send order --> init new episode
                for i,r,parent_conn in zip(self.runners_ids,self.runners,self.runner_parent_connections):
                    parent_conn.send((cmd,0))

                # get room
                for i,r,parent_conn in zip(self.runners_ids,self.runners,self.runner_parent_connections):
                    cmd, args = parent_conn.recv()
                    if cmd == 'ack_new_episode':
                        room = args
                        self.room_init_freq[room] += 1
                        self.rooms[i] = room
                        print('W{} runner {} initialized at room {}'.format(self.worker_id,i,room))
                self.worker_end.send(('ack',args))

            elif cmd == 'new_rollout':
                # obtain centralized params
                self.centralized_critic_params = args

                # monitorize train_steps
                self.num_samples_runner = [0]*len(self.runners)

                # init new rollout
                self.states = []
                self.actions = []
                self.next_states = []
                self.rewards_ext = []
                self.dones = []

                self.stacked_states = []
                self.trajectory = []
                self.angle = []
                self.h_in = []
                self.trajectory_at_runner=[]
                self.policy_values = []
                # V(s)
                self.value_ext = []
                self.value_int = []
                # V(s')
                self.last_value_ext = []
                self.last_value_int = []

                # get model params to send to runners
                if self.use_cuda: # it is not possible to pass whole model of cuda through pipe connection --> first to cpu and then send
                    actor_params = self.agent.actor.to('cpu').state_dict()
                    critic_params =  self.centralized_critic_params if self.use_centralized_critic else self.agent.critic.to('cpu').state_dict()
                    self.agent.actor.to(self.device)
                    self.agent.critic.to(self.device)
                else:
                    actor_params = self.agent.actor.state_dict()
                    critic_params = self.centralized_critic_params if self.use_centralized_critic else self.agent.critic.state_dict()

                # send model parameters
                for i,r,parent_conn in zip(self.runners_ids,self.runners,self.runner_parent_connections):
                    # only send to those runners that have still not finished their episode
                    if self.active_runners[i]:
                        args = (actor_params, critic_params, self.input_hidden[i])
                        parent_conn.send((cmd, args))

                # get batch of experiences collected at runner
                for i,r,parent_conn in zip(self.runners_ids,self.runners,self.runner_parent_connections):
                    # only runners that are running
                    if self.active_runners[i]:
                        cmd, args = parent_conn.recv()

                        # unpack whole experiences
                        states, actions,rewards,\
                        next_states,dones,stacked_states,\
                        value_int,value_ext,\
                        episode_finished,episode_steps,runner_trajectory,\
                        angle, last_value_ext,last_value_int, h_in, policy_values \
                        = args

                        self.steps_in_episode[i] += len(actions)

                        # monitorization of params
                        self.episode_finished[i] = episode_finished
                        self.score[i] = rewards

                        # monitorization of experiences
                        self.num_samples_runner[i] = len(states)
                        self.states.append(states)
                        self.actions.append(actions)
                        self.rewards_ext.append(rewards)
                        self.next_states.append(next_states)
                        self.dones.append(dones)
                        self.stacked_states.append(stacked_states)
                        self.value_int.append(value_int)
                        self.value_ext.append(value_ext)
                        # for V(s') estimation at optimization phase
                        self.last_value_ext.append(last_value_ext)
                        self.last_value_int.append(last_value_int)
                        self.policy_values.append(policy_values)
                        # hidden_state at collection
                        self.h_in.append(h_in)

                        # additional interesting parameters to analyze
                        self.trajectory.extend(runner_trajectory) # for monitorization purposes (does not distinguish between runners)
                        self.trajectory_at_runner.append(runner_trajectory) # used to filter samples of a given experience
                        self.angle.extend(angle)

                        # monitorization params update
                        if episode_finished:
                            self.trajectories[self.rooms[i]].append((self.episode_counter,runner_trajectory))
                            if episode_steps < self.max_episode_len:
                                self.room_steps_for_done[self.rooms[i]].append(episode_steps)

                # boolean to know whether a worker has completely finished an episode
                all_runners_finished = np.all(self.episode_finished)

                # score calculation from all runners of the same worker
                avg_score = 0
                if np.all(self.episode_finished):
                    avg_score = np.mean([v[-1] for v in self.score.values()])
                    self.score_window.append(avg_score)
                    print('*** W{} EP{} avg_score:{}; last100 avg: {}***\n'.format(self.worker_id,self.episode_counter,avg_score,np.mean(self.score_window)))

                args = ((self.states,self.actions,self.next_states,self.rewards_ext,self.dones, self.stacked_states,self.policy_values),\
                        avg_score, all_runners_finished, self.agent.train_steps,self.trajectory, self.angle, self.steps_in_episode,self.active_runners,self.trajectory_at_runner)

                self.worker_end.send(('rollout_finished',args))

            elif cmd == 'send_policy_values':
                stacked_states_all_runners = args

                policy_values = []
                with torch.no_grad():
                    for stacked_states in stacked_states_all_runners:
                        input_states = torch.tensor(stacked_states).to(self.device).float()
                        policy_val = self.agent.actor(input_states).detach().cpu().numpy()
                        policy_values.append(policy_val)
                self.worker_end.send(('base-policy_values',policy_values))

            elif cmd == 'train':

                rewards_int_normalized = args

                # calculation of advantages & discounted_returns
                advantages = []
                advantages_ext = []
                advantages_int = []
                disc_return_ext = []
                disc_return_int = []

                # determine ext_coef and inf_coef
                self.calculate_ext_int_coefs()

                j = 0
                for i in self.runners_ids:
                    if self.active_runners[i]:
                        # advantage-ext
                        adv_ext,return_ext,value_ext \
                        = get_advantages(rewards = self.rewards_ext[j],
                                        dones = self.dones[j],
                                        value = self.value_ext[j],
                                        next_v = self.last_value_ext[j],
                                        gamma = self.gamma,
                                        lmbda = self.lmbda,
                                        use_gae = self.use_gae,
                                        type = 1)

                        advantages_ext.append(adv_ext)
                        disc_return_ext.append(return_ext)

                        if self.curiosity_type != 'none':
                            # advantage-int
                            adv_int,return_int,value_int \
                            = get_advantages(rewards = rewards_int_normalized[j],
                                            dones = self.dones[j],
                                            value = self.value_int[j],
                                            next_v = self.last_value_int[j],
                                            gamma = self.gamma_int,
                                            lmbda = self.lmbda,
                                            use_gae = self.use_gae,
                                            type = self.intrinsic_rew_episodic)

                            advantages_int.append(adv_int)
                            disc_return_int.append(return_int)

                        # Combine EXTRINSIC & INTRINSIC STREAMS
                        if self.curiosity_type != 'none':
                            advantage_total = (adv_ext * self.ext_coef) + (adv_int * self.int_coef)
                            advantage_total /= (self.ext_coef + self.int_coef)
                        else:
                            advantage_total = adv_ext

                        advantages.append(advantage_total)

                        # update counter
                        j += 1


                # params required for training --> reshape of type(list)
                # for actor
                advantages = [rollout_at_runner for runner_level in advantages for rollout_at_runner in runner_level]
                # for critic
                disc_return_ext = [rollout_at_runner for runner_level in disc_return_ext for rollout_at_runner in runner_level]
                disc_return_int = [rollout_at_runner for runner_level in disc_return_int for rollout_at_runner in runner_level]
                self.h_in = [rollout_at_runner for runner_level in self.h_in for rollout_at_runner in runner_level]
                #for both
                self.stacked_states = [rollout_at_runner for runner_level in self.stacked_states for rollout_at_runner in runner_level]
                self.actions = [rollout_at_runner for runner_level in self.actions for rollout_at_runner in runner_level]

                # monitorization reshape
                advantages_ext = [rollout_at_runner for runner_level in advantages_ext for rollout_at_runner in runner_level]
                advantages_int = [rollout_at_runner for runner_level in advantages_int for rollout_at_runner in runner_level]
                self.value_int = [rollout_at_runner for runner_level in self.value_int for rollout_at_runner in runner_level]
                self.value_ext = [rollout_at_runner for runner_level in self.value_ext for rollout_at_runner in runner_level]

                # required --> get idx and hidden states for each runner
                idx_i,idx_f = 0,0
                idx_runner = [[] for _ in range(len(self.runners))]
                for i,spr in enumerate(self.num_samples_runner): #num_samples_runner at current rollout
                    idx_f += spr
                    idx_runner[i] =(idx_i,idx_f)
                    if spr != 0:
                        self.input_hidden[i] = self.h_in[idx_i] # get hidden state of that rollout/runner (first idx)
                    # update
                    idx_i = idx_f

                # TRAIN FUNCTION
                # -- (updates always the Actor)
                # -- (if centralized critic is used, that is done at main)
                self.agent.train_model_batch(stacked_states = np.asarray(self.stacked_states),
                                             actions= np.asarray(self.actions),
                                             advantages = np.asarray(advantages),
                                             disc_return_ext = np.asarray(disc_return_ext),
                                             disc_return_int = np.asarray(disc_return_int),
                                             worker_id = self.worker_id,
                                             idx_runner = idx_runner,
                                             h_in = self.input_hidden)

                # UPDATE HIDDEN STATES IF NECESSARY (if not centralized)
                if self.use_lstm and not self.use_centralized_critic:
                    for j,(i,f) in enumerate(idx_runner):
                        if i!=f:
                            input_stacked_states = torch.from_numpy(np.asarray(self.stacked_states[i:f])).float()
                            input_hidden = self.input_hidden[j]
                            input_actions = torch.from_numpy(np.asarray(self.actions[i:f]))
                            input_roles = torch.from_numpy(np.full(len(input_actions),self.worker_id))

                            self.input_hidden[j] = self.agent.calculate_hidden_state(
                                                                            stacked_states=input_stacked_states,
                                                                            hidden=input_hidden,
                                                                            actions=input_actions,
                                                                            roles=input_roles)

                # GET METRICS RELATED TO TRAIN
                actor_loss,critic_ext_loss,critic_int_loss,entropy,ppo_loss,\
                value_ext_preW, value_ext_postW, prob_move_up, prob_move_right, prob_move_left,\
                entropy_step,policy_probs,clipped_info, grad_actor, grad_critic \
                = self.agent.get_train_related_metrics()

                args = (advantages,\
                       advantages_ext,disc_return_ext,self.value_ext,\
                       advantages_int,disc_return_int,self.value_int,\
                       actor_loss,critic_ext_loss,critic_int_loss,entropy,ppo_loss,\
                       value_ext_preW, value_ext_postW, \
                       prob_move_up, prob_move_right, prob_move_left,entropy_step,\
                       policy_probs,clipped_info,\
                       grad_actor, grad_critic,self.input_hidden,idx_runner,)

                # send by pipe
                self.worker_end.send(('training_completed',args))

                #once training is made, set runner as inactive if episode finished before than expected (i.e goal achieved)
                for i in range(self.num_envs):
                    self.active_runners[i] = not self.episode_finished[i]

            elif cmd == 'update_centralized_critic':
                centralized_critic_params, hidden_state = args
                self.input_hidden = hidden_state
                self.centralized_critic_params = centralized_critic_params

            elif cmd == 'get_model':
                actor_params = self.agent.actor.to('cpu').state_dict()
                self.agent.actor.to(self.device)

                if self.use_centralized_critic:
                    critic_params = self.centralized_critic_params
                else:
                    critic_params = self.agent.critic.to('cpu').state_dict()
                    self.agent.critic.to(self.device)

                # compress and send
                args = ((actor_params,critic_params,0,0,0))
                self.worker_end.send(('model_sent',args))
                print('W{} all worker monitorization params sent!'.format(self.worker_id))

            elif cmd == 'close_connection':
                for parent_conn in self.runner_parent_connections:
                    parent_conn.send((cmd,0))
                print('Game {} closed'.format(self.worker_id))
                self.worker_end.send(('connection_closed',0))
                self.worker_end.close() # close worker after loop finishes
                break
