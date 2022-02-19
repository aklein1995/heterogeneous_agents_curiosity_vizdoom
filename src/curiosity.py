#!/usr/bin/env python3

import numpy as np
from src.model import RNDModel

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RNDModule():
    def __init__(self, input_size, output_size,
                     learning_rate, update_proportion,
                     use_cuda=False, device_id=0,
                     multi_head_action=0):

        self.num_neurons = 102
        self.multi_head_action = multi_head_action
        self.output_size = output_size
        self.input_size = input_size

        # Predictor-Curiosity
        if self.multi_head_action:
            print('Multi-head action curiosity RND')
            self.rnd = RNDModel_actions(self.input_size, self.output_size)
        else:
            print('Normal RND')
            self.rnd = RNDModel(self.input_size, self.output_size)
        # optimizer only update predictor parameters
        self.optimizer = optim.Adam(list(self.rnd.predictor.parameters()), lr=learning_rate)
        self.update_proportion = update_proportion
        self.device = torch.device('cuda:'+str(device_id) if use_cuda else 'cpu')
        self.rnd = self.rnd.to(self.device)
        print('output rnd size:',self.output_size)
        print('update proportion:',self.update_proportion)

    def compute_intrinsic_reward(self, next_obs, actions=[]):
        """
            Calculate intrinsic reward
        """
        # Observation transformation
        next_obs = np.array(next_obs)
        next_obs = torch.from_numpy(next_obs).to(self.device).float()
        next_obs = next_obs.unsqueeze(1)

        # Get Target and Prediction of next feature
        with torch.no_grad():
            target_next_feature = self.rnd.target(next_obs)
        predict_next_feature = self.rnd.predictor(next_obs)

        # Calculate instrinsic reward
        if self.multi_head_action:
            actions = torch.from_numpy(actions).unsqueeze(1)
            # get neurons relative to that action
            idx = []
            for a in actions:
                aux = list(np.arange(a*self.num_neurons,(a+1)*self.num_neurons))
                idx.append(aux)
            idx = torch.tensor(idx).to(self.device)
            error = (target_next_feature - predict_next_feature).gather(dim=-1,index=idx)
            # intrinsic_reward = (error.pow(2).mean(dim=1) / 2) # MEAN
            intrinsic_reward = self.output_size*(error.pow(2).sum(dim=1) / 2) #SUM

            """
                Originally we have 512 output neurons. For the action based approach, we have the same amount (510 as it is mod5=0, the numb of possible actions)
                510/5 = 102 neuros per action

                SUM operator
                --> multiply by all total number of actions, to get equal ponderation at the sum operator as if it would be when calculating with the total action space
                --> if we would have 510 output-neuron (instead of 102) for each action, then we would not need to multiplicate (this would require to have 510*5= 2550 output neurons)

                MEAN operator
                --> the mean across 102 or 510 neurons is indeed the mean; with more neurons we would have less variance
            """
        else:
            error = (target_next_feature - predict_next_feature)
            intrinsic_reward = error.pow(2).sum(dim=1) / 2

        return intrinsic_reward.data.cpu().clone().numpy()

    def train(self,allworkers_next_obs,actions=[]):
        """
            Compute loss of predictor network
        """
        # Observation transformation
        next_obs = np.array(allworkers_next_obs) #get [batch_size,42,42]
        next_obs = torch.from_numpy(next_obs).to(self.device).float() # transform to tensor
        next_obs = next_obs.unsqueeze(1) #  add required format (rgb=3 or grayscale=1 after batchsize) -->x [batch_size,1,42,42]

        # Get prediction and target for next_state
        with torch.no_grad():
            target_next_state_feature = self.rnd.target(next_obs)
        predict_next_state_feature = self.rnd.predictor(next_obs)

        forward_mse = nn.MSELoss(reduction='none')

        if self.multi_head_action:
            actions = np.array(actions)
            actions = torch.from_numpy(actions).unsqueeze(1)

            idx = []
            for a in actions:
                aux = list(np.arange(a*self.num_neurons,(a+1)*self.num_neurons))
                idx.append(aux)
            idx = torch.tensor(idx).to(self.device)

            forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).gather(dim=1,index=idx)
            # Proportion of exp used for predictor update
            mask = torch.rand(forward_loss.shape).to(self.device)
            mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)

            # apply mask into the neurons before making any sum (mask inside neurons of all actions instead of through actions)
            forward_loss = (forward_loss * mask).mean(dim=1) # neuron by neuron multiplication to apply mask, and then apply mean to get the loss for that step
            # loss calculation
            loss = (forward_loss).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device)) #as we have already make the mask operation before, here the numerator is only a SUM and the denominator divides to get equal weighted losses

        else:
            forward_loss = forward_mse(predict_next_state_feature, target_next_state_feature.detach()).mean(dim=1)
            # Proportion of exp used for predictor update
            mask = torch.rand(len(forward_loss)).to(self.device)
            mask = (mask < self.update_proportion).type(torch.FloatTensor).to(self.device)
            # loss calculation
            loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(self.device))
        # Reset gradients
        self.optimizer.zero_grad()

        # Backpropagation
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.rnd.parameters(), 0.5)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

class CountModule():
    def __init__(self, reduction_factor=32,action_based=False,num_parallel_envs=1,take_into_account_action_decay=False):
        # x-axis
        # 1120-160=960 max values at x-axis; 960//32 = 30 is the max index
        # 1120-160=960 max values at x-axis; 960//16 = 60 is the max index
        # y-axis
        # 992+128=1120 max values at x-axis; 1120//32 = 35 is the max index
        # 992+128=1120 max values at x-axis; 1120//16 = 70 is the max index

        self.reduction_factor = reduction_factor
        self.action_based = action_based
        self.num_parallel_envs = num_parallel_envs
        self.take_into_account_action_decay = take_into_account_action_decay

        if self.reduction_factor == 32:
            self.dim_x,self.dim_y = 30,35
        elif self.reduction_factor == 16:
            self.dim_x,self.dim_y = 60,70

        if action_based:
            self.counts = {k:{a:1 for a in range(5)} for k in range(self.dim_x*self.dim_y)}
            self.counts_action_dependant = {k:{a:1 for a in range(5)} for k in range(self.dim_x*self.dim_y)}
        else:
            self.counts = {k:1 for k in range(self.dim_x*self.dim_y)}
            self.counts_action_dependant = {k:1 for k in range(self.dim_x*self.dim_y)}

        # for temporal storage
        self.temporal_coordinates = []
        for _ in range(num_parallel_envs):
            self.temporal_coordinates.append([])

        self.temporal_actions = []
        for _ in range(num_parallel_envs):
            self.temporal_actions.append([])

    def update_action_dependant(self,worker_id):
        """
            Updates what is in quarantine storage if it has used the action that is different
            -Executed at the init of each episode
        """
        if worker_id == 0 and self.take_into_account_action_decay and len(self.temporal_coordinates[0]) > 0:
            print('\nChecking tree...')
            for rid in range(self.num_parallel_envs):

                action_dependant_experiences = False
                when_diverge = 0
                for counter,((x,y),a) in enumerate(zip(self.temporal_coordinates[rid],self.temporal_actions[rid])):
                    if a == 4:# check only when executing action USE = [0,0,0,1] with id 4
                        c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor # shortmap
                        # add that rollout to be trained on other RND
                        # if (c1 >= 34 and c1 <= 37) and (c2>=48 and c2<=54): # for reduction=16
                        if (c1 >= 17 and c1 <= 19) and (c2>=24 and c2<=26): # for reduction=32
                        # if (c1 >= 15 and c1 <= 16) and (c2>=0 and c2<=10): # init respawn for reduction=32
                        # if (c1 >= 9 and c1 <= 10) and (c2>=9 and c2<=10): # corner edge before room 13 for reduction=32
                            print('\nEntra en la zona peligrousa al cabo de {} steps!'.format(counter))
                            action_dependant_experiences = True
                            when_diverge = counter
                            break

                if action_dependant_experiences:
                    for counter,((x,y),a) in enumerate(zip(self.temporal_coordinates[rid],self.temporal_actions[rid])):
                        c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor
                        bin = (c2*self.dim_x) + c1
                        self.counts_action_dependant[bin][a] += 1
                        if counter == when_diverge:
                            print('trained until ',counter)
                            break

        self.temporal_coordinates = []
        for _ in range(self.num_parallel_envs):
            self.temporal_coordinates.append([])

        self.temporal_actions = []
        for _ in range(self.num_parallel_envs):
            self.temporal_actions.append([])

    def compute_intrinsic_reward(self,worker_id,coordinates,actions=[]):
        """
            Calculate intrinsic reward os given samples
        """
        intrinsic_rewards = []
        if self.action_based and len(actions) > 0:
            # action-based
            for (x,y),a in zip(coordinates,actions):
                c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor
                bin = (c2*self.dim_x) + c1

                count = self.counts[bin][a]
                if worker_id == 1 and self.take_into_account_action_decay:
                    count = self.counts[bin][a] - self.counts_action_dependant[bin][a]
                    count = max(1,count) # avoid count = 0

                intrinsic_rewards.append(1/np.sqrt(count))
        else:
            for x,y in coordinates:
                c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor
                # [numFila(y)*maxElemPorFila(dim_x)]+columna(x)
                bin = (c2*self.dim_x) + c1

                count = self.counts[bin]
                if worker_id == 1 and self.take_into_account_action_decay:
                    count = self.counts[bin] - self.counts_action_dependant[bin]
                    count = max(1,count) # avoid count = 0

                intrinsic_rewards.append(1/np.sqrt(count))

        return np.array(intrinsic_rewards)

    def train(self,coordinates,actions=[],worker_id=0,runner_id=0):
        """
            Add samples to the bins
        """

        if self.action_based and len(actions) > 0:
            # action-based
            for (x,y),a in zip(coordinates,actions):
                c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor
                bin = (c2*self.dim_x) + c1
                self.counts[bin][a] += 1
        else:
            for x,y in coordinates:
                c1,c2 = (int(x)-(160)) // self.reduction_factor, (int(y)+992)// self.reduction_factor
                bin = (c2*self.dim_x) + c1
                self.counts[bin] += 1

        # add samples to temporal storage
        if self.take_into_account_action_decay and worker_id == 0:
            self.temporal_coordinates[runner_id].extend(coordinates)
            self.temporal_actions[runner_id].extend(actions)
