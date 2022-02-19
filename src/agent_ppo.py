#!/usr/bin/env python3

import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical
torch.set_printoptions(precision=12)

from collections import deque
from src.model import Actor,Critic_LSTM

class Agent_lstm():
    def __init__(self, input_size, output_size,
                epoch, learning_rate,
                ent_coef=.01, ppo_eps=.1,
                clip_grad_norm =.5,
                use_cuda=False, device_id=0,
                num_minibatches = 1,
                use_centralized_critic = 0,
                im_centralized_critic = 0,
                train_actor = True, train_critic = True,
                load_actor=False,load_critic=False,
                path_actor='', path_critic='',
                use_lstm = 0, use_action_as_input = 0,
                use_role_as_input=0,
                lstm_hidden_size = 128, lstm_bidirectional = 0,
                num_parallel_envs = 1,
                verbose_test = False):

        self.verbose = verbose_test

        self.use_lstm = use_lstm
        self.use_actions_as_input = use_action_as_input
        self.use_role_as_input = use_role_as_input
        self.num_parallel_envs = num_parallel_envs

        self.train_steps = 0
        self.action_size = output_size
        self.learning_rate = learning_rate
        self.clip_grad_norm = clip_grad_norm
        self.use_centralized_critic = use_centralized_critic
        self.im_centralized_critic = im_centralized_critic

        # GPU/CPU usage
        self.device = torch.device('cuda:'+str(device_id) if use_cuda else 'cpu')

        # ACTOR
        self.actor = Actor(input_size=input_size, output_size=output_size)

        # CRITIC
        self.input_hidden = [[] for _ in range(num_parallel_envs)]
        self.idx_runner = [[] for _ in range(num_parallel_envs)]
        self.lstm_hidden_size = lstm_hidden_size
        self.bidirectional = lstm_bidirectional

        output_size_critic =  output_size if use_centralized_critic else 1

        self.critic = Critic_LSTM(input_size=input_size, output_size=output_size_critic,
                                  use_lstm = self.use_lstm, hidden_size =lstm_hidden_size, bidirectional=bool(lstm_bidirectional),
                                  use_role_as_input=use_role_as_input, use_action_as_input=use_action_as_input,
                                  device = self.device)
        self.init_lstm_hidden()

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)


        # load params if specified
        if load_actor:
            PATH_ACTOR = path_actor
            self._load_actor(PATH_ACTOR)
            print('loaded actor from: ',PATH_ACTOR)
        if load_critic:
            PATH_CRITIC = path_critic
            self._load_critic(PATH_CRITIC)
            print('loaded critic from: ',PATH_CRITIC)

        # define optimizers
        self.optimizer_actor = optim.Adam(list(self.actor.parameters()), lr=learning_rate)
        self.optimizer_critic = optim.Adam(list(self.critic.parameters()), lr=learning_rate)

        # PP0 parameters
        self.ent_coef = ent_coef
        self.ppo_eps = ppo_eps
        self.num_minibatches = num_minibatches
        self.epoch = epoch

        # training step monitorization network losses
        self.total_loss = []
        self.bf_actor_loss = []
        self.bf_critic_ext_loss = []
        self.bf_critic_int_loss = []
        self.bf_entropy = []

        # set if neural networks are going to be trained (useful for testing)
        self.train_actor = train_actor
        self.train_critic = train_critic

    def _load_actor(self,actor_params):
        """
            load actor parameters
        """
        self.actor.load_state_dict(torch.load(actor_params))

    def _load_critic(self,critic_params):
        """
            load critic parameters
        """
        self.critic.load_state_dict(torch.load(critic_params))

    def get_critic(self):
        if isinstance(self.critic, torch.nn.DataParallel):
            return self.critic.module.to('cpu').state_dict()
        else:
            return self.critic.to('cpu').state_dict()

    def init_lstm_hidden(self):
        if isinstance(self.critic, torch.nn.DataParallel):
            return self.critic.module.init_hidden()
        else:
            return self.critic.init_hidden()

    def calculate_hidden_state(self,stacked_states,hidden,actions,roles):
        """
            Update the last output hidden state based on the current network weight updates
            (so that the next trajectories are collected with updated hidden state)
        """
        stacked_states = stacked_states.to(self.device)
        hidden = (hidden[0].to(self.device),hidden[1].to(self.device))
        actions = actions.to(self.device)
        roles = roles.to(self.device)

        with torch.no_grad():# the initial state has to be a constant --> https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/3
            _, _, new_hidden = self.critic(state=stacked_states,hidden=hidden,action=actions,role=roles)
        new_hidden = (new_hidden[0].to('cpu'),new_hidden[1].to('cpu'))
        return new_hidden

    def get_gradients(self,model):
        """
            Get gradients to monitorize updates
        """
        gradients = []
        for p in model.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear):
                try:
                    g = p.weight.grad.mean().cpu().detach().numpy()
                    gradients.append(g)
                except AttributeError:
                    pass

        return gradients

    def train_model_batch(self,stacked_states,actions,disc_return_ext,disc_return_int,h_in,idx_runner,advantages=0,worker_id=0):
        """
            Train the batch (collection of trajectories/rollouts) gathered by each worker
        """
        [print('\n***at ppo lstm class***') if (self.im_centralized_critic and self.verbose) else 0]

        # convert into torch
        stacked_states = torch.from_numpy(stacked_states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device)
        roles = torch.from_numpy(np.full(len(actions),worker_id)).to(self.device)

        # import hidden states and idx per runner
        self.input_hidden = []
        for h in h_in:
            hc_tuple = (h[0].to(self.device),h[1].to(self.device))
            self.input_hidden.append(hc_tuple)
        self.idx_runner = idx_runner

        self.value_ext_postW = torch.rand(1) #mierda para que no casque (actualizar codigo porque no sirve par anada este parametro)

        # set batch size --> default=size equal to all the rollouts of the worker's runners
        self.batch_size = len(disc_return_ext)

        # init training step monitorization params
        self.total_loss = []
        self.bf_actor_loss = []
        self.bf_critic_ext_loss = []
        self.bf_critic_int_loss = []
        self.bf_entropy = []
        self.gradients_actor = []
        self.gradients_critic = []
        self.bf_clip = [] # used to know when clipping (epislon) params is used

        # Compute old probabilities (for all the possible states in the batch)
        with torch.no_grad():

            self.old_value_ext = torch.tensor([],requires_grad = False).to(self.device)
            self.old_value_int = torch.tensor([],requires_grad = False).to(self.device)

            # get original values independently for each runner and then concat (due to different hidden states)
            for r_id,(h,(i,f)) in enumerate(zip(self.input_hidden,self.idx_runner)):
                if i!=f: # if it contains an active runner
                    ve, vi, h_print = self.critic(state=stacked_states[i:f],action=actions[i:f],role=roles[i:f],hidden=h)
                    self.old_value_ext = torch.cat((self.old_value_ext,ve),dim=0)
                    self.old_value_int = torch.cat((self.old_value_int,vi),dim=0)
                    if self.im_centralized_critic and self.verbose:
                        print('W{} R{} h_in:{}'.format(worker_id,r_id,h))
                        print('W{} R{} h_out:{}'.format(worker_id,r_id,h_print))
                        print('W{} R{} ve:{}'.format(worker_id,r_id,ve))
            self.old_policy = self.actor(state=stacked_states)
            self.m_old = Categorical(self.old_policy)
            log_prob_old = self.m_old.log_prob(actions)

        # Execute for X-Epochs
        for i in range(self.epoch):
            self._train_epoch(stacked_states,actions,log_prob_old,advantages,disc_return_ext,disc_return_int,roles)

        self.train_steps += 1

    def _train_epoch(self,stacked_states,actions,log_prob_old,advantages,disc_return_ext,disc_return_int,worker_type):
        """
            Trains one epoch the specified networks
            Calculates the size of each minibatch based on the number of minibatches and batch_size
        """
        #set minibatch size
        minibatch_size = int(self.batch_size / self.num_minibatches)

        # Set all-possible states as integer list
        sample_range = np.arange(self.batch_size)

        # randomize samples in batch
        np.random.shuffle(sample_range)

        for n in range(self.num_minibatches):
            first = n * minibatch_size
            last = first + minibatch_size
            sample_idx = sample_range[first:last]

            sample_idx = sample_idx.tolist()

            # targets
            targets_ext = torch.from_numpy(disc_return_ext)
            targets_int = torch.from_numpy(disc_return_int)

            # if it is just the centralized_critic
            if self.use_centralized_critic and self.im_centralized_critic:
                self._train_minibatch_critic(stacked_states, actions, targets_ext, targets_int, sample_idx, worker_type)

            # train the actors (this is used at worker.py)
            elif self.use_centralized_critic and not self.im_centralized_critic:
                if self.train_actor:
                    self._train_minibatch_actor(stacked_states, actions, log_prob_old,advantages, sample_idx)

            # trains both actor & critic
            elif not self.use_centralized_critic:
                if self.train_actor:
                    self._train_minibatch_actor(stacked_states, actions, log_prob_old, advantages, sample_idx)
                if self.train_critic:
                    self._train_minibatch_critic(stacked_states, actions, targets_ext, targets_int, sample_idx, worker_type)

    def _train_minibatch_actor(self,stacked_states, actions, log_prob_old, advantages, sample_idx):
        """
            PPO actor update for the current minibatch
        """
        # Get current policy values
        policy = self.actor(state=stacked_states)
        # Compute log_probabilities
        m = Categorical(policy)
        self.log_prob = m.log_prob(actions)

        # Define advantage function
        advantages = torch.from_numpy(advantages).to(self.device).float()

        # Calculate ratio between new and old policy
        ratio = torch.exp(self.log_prob - log_prob_old).to(self.device)

        # Clamp all elements in input (ratio) into the range [min, max ]
        clip = torch.clamp(
            ratio,
            1.0 - self.ppo_eps,
            1.0 + self.ppo_eps).to(self.device)

        # Calculate actor loss
        surr1 = ratio * advantages
        surr2 = clip * advantages
        actor_loss = -torch.min(surr1,surr2).mean()

        # Calculate entropy --> improve exploration
        entropy = m.entropy().mean()

        # final loss
        loss = actor_loss - self.ent_coef*entropy

        # Reset gradients
        self.optimizer_actor.zero_grad()
        # Backpropagation
        loss.backward()
        # gradient normalization
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad_norm)
        gradients = self.get_gradients(self.actor)
        self.optimizer_actor.step()

        # update monitorization params (ACTOR)
        self.bf_actor_loss.append(actor_loss.detach().cpu().numpy())
        self.bf_entropy.append(-self.ent_coef*entropy.detach().cpu().numpy())
        self.total_loss.append(loss.detach().cpu().numpy())
        aux = [True if (c > 1 + self.ppo_eps) or (c < 1 -self.ppo_eps) else False for c in clip ]
        self.bf_clip.extend(aux)
        self.gradients_actor.append(gradients)

    def _train_minibatch_critic(self,stacked_states, actions, targets_ext, targets_int, sample_idx ,worker_type):
        """
            Critic update for the current minibatch (with MSE)
        """

        value_ext = torch.tensor([]).to(self.device)
        value_int = torch.tensor([]).to(self.device)

        # Calculate values for each runner with their hiddens
        for h,(i,f) in zip(self.input_hidden,self.idx_runner):
            if i!=f:
                ve, vi, hprint = self.critic(state=stacked_states[i:f],action=actions[i:f],role=worker_type[i:f],hidden=h)
                value_ext = torch.cat((value_ext,ve),dim=0)
                value_int = torch.cat((value_int,vi),dim=0)

        if self.im_centralized_critic:
            actions = actions.unsqueeze(1) # from shape [num_actions, 1] to [num_actions]
            # select the specific Q(s,a) from Q(s,.)
            value_ext = value_ext.gather(1,actions).squeeze(1)
            value_int = value_int.gather(1,actions).squeeze(1)

        critic_ext_loss = F.mse_loss(value_ext, targets_ext.to(self.device).float())
        self.bf_critic_ext_loss.append(critic_ext_loss.detach().cpu().numpy())
        critic_loss = critic_ext_loss

        # if intrinsic_rewards not generated, that head does not have a loss to backpropagate
        if len(targets_int) > 0:
            critic_int_loss = F.mse_loss(value_int, targets_int.to(self.device).float())
            self.bf_critic_int_loss.append(critic_int_loss.detach().cpu().numpy())
            critic_loss += critic_int_loss

        # Reset gradients
        self.optimizer_critic.zero_grad()
        # Backpropagation
        critic_loss.backward()
        # gradient normalization
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip_grad_norm)
        gradients = self.get_gradients(self.critic)
        # optimization step
        self.optimizer_critic.step()

        # save value after workout (see progress)
        self.value_ext_postW = value_ext
        # save gradients
        self.gradients_critic.append(gradients)

    def get_train_related_metrics(self):
        if self.im_centralized_critic:
            return  self.bf_critic_ext_loss,\
                    self.bf_critic_int_loss,\
                    self.old_value_ext.detach().cpu().clone().numpy(),\
                    self.value_ext_postW.detach().cpu().clone().numpy(),\
                    self.gradients_actor
        elif self.use_centralized_critic and not self.im_centralized_critic:
            return self.bf_actor_loss,\
                    0,\
                    0,\
                    self.bf_entropy,\
                    self.total_loss,\
                    0,\
                    0,\
                    self.old_policy[:,0].detach().cpu().clone().numpy(),\
                    self.old_policy[:,2].detach().cpu().clone().numpy(),\
                    self.old_policy[:,1].detach().cpu().clone().numpy(),\
                    self.m_old.entropy().detach().cpu().clone().numpy(),\
                    self.old_policy.detach().cpu().clone().numpy(),\
                    self.bf_clip, \
                    self.gradients_actor, \
                    0
        else:
            return  self.bf_actor_loss,\
                    self.bf_critic_ext_loss,\
                    self.bf_critic_int_loss,\
                    self.bf_entropy,\
                    self.total_loss,\
                    self.old_value_ext.detach().cpu().clone().numpy(),\
                    self.value_ext_postW.detach().cpu().clone().numpy(),\
                    self.old_policy[:,0].detach().cpu().clone().numpy(),\
                    self.old_policy[:,2].detach().cpu().clone().numpy(),\
                    self.old_policy[:,1].detach().cpu().clone().numpy(),\
                    self.m_old.entropy().detach().cpu().clone().numpy(),\
                    self.old_policy.detach().cpu().clone().numpy(),\
                    self.bf_clip, \
                    self.gradients_actor, \
                    self.gradients_critic
