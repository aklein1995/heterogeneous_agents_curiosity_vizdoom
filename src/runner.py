import torch
import torch.nn.functional as F
torch.set_printoptions(precision=12)

import numpy as np
from collections import deque
import vizdoom as vzd
import cv2

from src.model import Actor,Critic_LSTM
from util_functions.vizdoom_utils import ViZDoomMap


class Runner():
    """
        The runner is used to generate rollout of experiences.
            -Each worker has as many runners as parallel environments specified;
            once all of them have finished the collection, sync, it is generated
            a batch of experience for that worker.
    """
    def __init__(self, nsteps, input_size, output_size,
                    model_params, config_file_path,
                    worker_id, runner_id, runner_end,
                    available_actions, max_episode_len = 2100,
                    resolution=(42,42), stack_size=4,
                    use_stacked_frames=1, frame_repeat=4,
                    use_cuda=False, device_id=0,
                    rooms_in_setting=[], centralized_critic = 0,
                    centralized_output_size=1,
                    use_action_as_input=0, use_role_as_input=0,
                    use_lstm=0, lstm_hidden_size = 128, lstm_bidirectional = 0,
                    verbose_test = False
                    ):

        self.verbose = verbose_test

        # lstm_related
        self.use_lstm = use_lstm
        self.lstm_bidirectional = lstm_bidirectional

        #other input parameters
        self.use_actions_as_input = use_action_as_input
        self.use_role_as_input = use_role_as_input
        # other required parameters
        self.runner_end = runner_end
        self.action_size = output_size
        self.runner_id = runner_id
        self.worker_id = worker_id

        # GPU/CPU usage
        self.device = torch.device('cuda:'+str(device_id) if use_cuda else 'cpu')

        # Actor-critic modules
        self.actor = Actor(input_size=input_size, output_size=output_size)

        self.use_centralized_critic = centralized_critic
        self.centralized_critic_size = centralized_output_size
        output_critic = self.centralized_critic_size if self.use_centralized_critic else 1

        self.critic = Critic_LSTM(input_size=input_size, output_size=output_critic,
                                  use_lstm= self.use_lstm, hidden_size =lstm_hidden_size, bidirectional=bool(lstm_bidirectional),
                                  use_role_as_input=use_role_as_input, use_action_as_input=use_action_as_input,
                                  device = self.device)
        self.h_out = self.critic.getLSTMHiddenState()


        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        # train params
        self.nsteps = nsteps
        self.resolution = resolution
        self.stack_size = stack_size
        self.available_actions = available_actions
        self.frame_repeat = frame_repeat
        self.use_stacked_frames = use_stacked_frames
        self.max_episode_len = max_episode_len
        self.episode_steps = 0
        self.room = 10
        self.state_stacked_frames = deque([], maxlen=self.stack_size)

        # initialize environment
        self.env = vzd.DoomGame()
        self.env.load_config(config_file_path)
        self.env.init()
        self.worker_id = worker_id
        self.possible_rooms = rooms_in_setting
        print("Doom worker {} runner {} initialized".format(worker_id, runner_id))
        print('Possible rooms at thir runner:',self.possible_rooms)

        self.execution()

    def execution(self):
        """
            main control loop of a runner class --> executes what worker says
        """
        while True:
            cmd, args = self.runner_end.recv()
            if cmd == 'new_episode':
                self.runner_end.send(('ack_new_episode',self.restart_episode()))
            elif cmd == 'pre-train':
                self.runner_end.send(('ack_pre-train',self.new_episode_random()))
            elif cmd == 'new_rollout':
                actor_params, critic_params, critic_hidden = args
                self.runner_end.send(('ack_new_rollout',self.new_rollout(actor_params,critic_params,critic_hidden)))
            else:
                self.env.close()
                print('Environment of worker {}/runner{} closed'.format(self.worker_id,self.runner_id))
                self.runner_end.close()
                break

    def restart_episode(self):
        """
            Initializes the agent in the specified room
            -pre-poblates stacked_states and resets #steps and the total score of episode
        """
        self.episode_cumulative_reward = 0.0
        self.episode_steps = 0

        while True:
            self.env.new_episode()
            x = self.env.get_game_variable(vzd.GameVariable.POSITION_X)
            y = self.env.get_game_variable(vzd.GameVariable.POSITION_Y)
            self.room = ViZDoomMap().getRoom(x,y)
            # if self.room in [26]:#self.possible_rooms: # this is used if dense reward setting is wanted
            if True:
                break

        # stack frames
        state = self._preprocess(self.env.get_state().screen_buffer)
        self.state_stacked_frames = deque([], maxlen=self.stack_size)
        for _ in range(self.stack_size):
            self.state_stacked_frames.append(state)

        return self.room

    def new_rollout(self,actor,critic,h_out):
        """
            deploys the agent in the environment until a rollout/trajectory is gathered
            - finish if max episode steps already taken(done=False) or goal achieved(done=True)
            - returns collected info
        """
        # init/reset
        self._reset(actor,critic)

        # default init for h_in -- we pass to the input h_out which then would be assigned to h_in
        h_out = (h_out[0].to(self.device), h_out[1].to(self.device))
        init_hidden = h_out
        # get last visited state
        state = self.state_stacked_frames[-1]

        # For n in range number of steps
        for steps in range(self.nsteps):

            #update total episode steps
            self.episode_steps += 1

            # agent position
            x = self.env.get_game_variable(vzd.GameVariable.POSITION_X)
            y = self.env.get_game_variable(vzd.GameVariable.POSITION_Y)
            self.agent_trajectory.append([x,y])
            # angle
            angle = self.env.get_game_variable(vzd.GameVariable.ANGLE)
            self.angle.append(int(angle))

            # Action selection by agent based on current state
            if self.use_stacked_frames:
                h_in = h_out
                action, action_prob, value_ext, value_int, h_out = self.get_action(np.array(self.state_stacked_frames),h_in)
            else:
                action, action_prob, value_ext, value_int, h_out = self.get_action(np.array(state), h_in)


            # Take step at environment and persist the action self.frame_repeat times
            rew_ext = self.env.make_action(self.available_actions[action],self.frame_repeat)

            # Overwritte before obtained rew_ext (with frame skip may not be correct);
            rew_ext = self.env.get_total_reward()

            # ***WARNING ---> when having more intermediate rewards, it gives the cumulative along episode... thus we check  with the next funcitionality
            actual_reward = rew_ext - self.episode_cumulative_reward # get the actual reward of that step transition, comparing to the one of obtained in the step before

            if actual_reward > 0:
                self.episode_cumulative_reward += rew_ext
                rew_ext = actual_reward # now rew_ext would have the actual step reward value; not the cumulative
            else:
                rew_ext = actual_reward
            # ***

            # check if episode has finished
            if self.env.is_episode_finished() or rew_ext >= 1:
                done = True
                # if it has been success, screen_buffer does not already exists --> generate a next_state full of zeros to know that it has achieved
                if rew_ext > 0:
                    next_state = np.zeros(self.resolution).astype('float')
                else:
                    next_state = self._preprocess(self.env.get_state().screen_buffer)
            else:
                done = False
                next_state = self._preprocess(self.env.get_state().screen_buffer)

            # save experiences
            self.states.append(state)
            self.actions.append(action)
            self.rewards_ext.append(rew_ext)
            self.next_states.append(next_state)
            self.dones.append(done)

            self.value_int.append(value_int)
            self.value_ext.append(value_ext)

            self.stacked_states.append(np.array(self.state_stacked_frames))
            self.hidden_in.append((h_in[0].data.cpu(),h_in[1].data.cpu()))
            self.hidden_out.append(h_out)
            self.action_probability.append(action_prob)

            # update current state
            state = next_state
            self.state_stacked_frames.append(state)

            # break current rollout if episode has finished | max steps already taken
            if done or (self.episode_steps >= self.max_episode_len):
                x = self.env.get_game_variable(vzd.GameVariable.POSITION_X)
                y = self.env.get_game_variable(vzd.GameVariable.POSITION_Y)
                self.agent_trajectory[-1] =([x,y])
                self.episode_finished = True
                break

        # ***Rollout finished
        if done: # if goal achieved in that rollout
            print('W{} R{} (room {}) solved in {}/{} steps'.format(self.worker_id,self.runner_id,self.room,steps,self.episode_steps))



        # generate last V(s')
        hprint = h_in = h_out
        _,_, last_value_ext, last_value_int, _ = self.get_action(np.array(self.state_stacked_frames),h_in)

        if self.verbose:
            print('-----------------------------')
            print('W{} R{} final hidden values:{}'.format(self.worker_id,self.runner_id,hprint))
            print('W{} R{} final v(s) values:{}'.format(self.worker_id,self.runner_id,self.value_ext))
            print('W{} R{} lastvalue :{}'.format(self.worker_id,self.runner_id,last_value_ext))
            print('-----------------------------')


        # return collected experiences
        return self.states, self.actions, self.rewards_ext, \
                self.next_states, self.dones, self.stacked_states, \
                self.value_int, self.value_ext,\
                self.episode_finished, self.episode_steps, self.agent_trajectory,\
                self.angle, last_value_ext, last_value_int, self.hidden_in, self.action_probability

    def get_action(self, state, h_in):
        """
            get selected action by the current agent for a state observation (stacked or not)

            Inputs:
                -Observation
                -Hidden state (for the critic)
            Returns:
                -Action
                -Action distribution
                -Ve(s)
                -Vi(s)
                -Output hidden_state
        """
        h_out = h_in # just in case if we are not using lstm to pass with no error

        # insert dimensions to state in order to have [batch_size,chaneel,x_dim,y_dim]
        if self.use_stacked_frames:
            # here only necessary to add batch_size as we already have [chanel=4 (stacked frames),42,42]
            state = torch.from_numpy(state).to(self.device).float().unsqueeze(0)
        else:
            # here we have only a raw image, so we need to add 2 dims --> double unsqueeze
            state = torch.from_numpy(state).to(self.device).float().unsqueeze(0).unsqueeze(0)

        # Get action distribution and action
        with torch.no_grad():
            # Get policy and ext/int critic values
            policy = self.actor(state)
            # select action
            action_prob = policy.data.cpu().squeeze(0).detach().numpy()
            try:
                action = self.random_choice_prob_index(action_prob)
            except ValueError:
                # this may happen when exploiting gradients
                print('policy values:',policy)
                print('action_prob:',action_prob)
                print('actionsize:',self.action_size)
                print('state:',state)
                print('state:',state.shape)
                action = 0

        # prepare possible additional input data for the critic
        action_input_model = torch.tensor([]).to(self.device)
        role_input_model = torch.tensor([]).to(self.device)
        if self.use_actions_as_input:
            action_input_model = torch.from_numpy(np.array([action])).to(self.device)
        if self.use_role_as_input:
            role_input_model = torch.from_numpy(np.array([self.worker_id])).to(self.device)

        # Get Q(s,a) value estimates
        with torch.no_grad():
            value_ext, value_int, h_out = self.critic(state=state, hidden=h_in,action=action_input_model,role=role_input_model)

        # -- for centralized critic Q(s,a) --> V(s) calculation with actor probabilities
        if self.use_centralized_critic:
            value_ext,value_int = self.get_value_from_q(action_prob[np.newaxis,:],value_ext,value_int)

        return  action, action_prob, \
                value_ext.data.cpu().squeeze().clone().numpy(), \
                value_int.data.cpu().squeeze().clone().numpy(), \
                h_out

    def get_value_from_q(self,action_prob,value_ext,value_int):
        """
            Transform from Q(s,a) to V(s) taking into account action distribution
        """
        # check if mask is necessary for action probabilities length
        if self.centralized_critic_size !=  self.action_size:
            dif = self.centralized_critic_size - self.action_size
            append_zeros = np.zeros((action_prob.shape[0],dif))
            action_prob = np.append(arr=action_prob,values=append_zeros,axis=1)

        # calculations
        value_ext = value_ext.squeeze(0) # from[1,5] to [5]
        value_int = value_int.squeeze(0)
        value_ext = torch.sum(input=(value_ext * torch.from_numpy(action_prob).to(self.device)),dim=-1)
        value_int = torch.sum(input=(value_int * torch.from_numpy(action_prob).to(self.device)),dim=-1)
        return value_ext,value_int

    def random_choice_prob_index(self,probs):
        """
            get random action by generating a cumulative distributions and
            taking an action from it based on action selection probabilities
                -probs: action probabilities
        """
        action = np.random.choice(a=np.arange(self.action_size),p=probs)
        return action

    def new_episode_random(self):
        """
            Executes a episode with a random action distribution until
            goal or max num steps is achieved

            -returns all collected experiences
        """
        self.restart_episode()

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards_ext = []
        self.dones = []

        next_state = self._preprocess(self.env.get_state().screen_buffer)

        for steps in range(self.max_episode_len):
            state = next_state
            action = np.random.choice(np.arange(self.action_size))
            rew_ext = self.env.make_action(self.available_actions[action],self.frame_repeat)
            tot_rew = self.env.get_total_reward() # overwritte before obtained rew_ext (with frame skip may not be correct)

            if self.env.is_episode_finished() or tot_rew > 0:
                done = True
                if tot_rew > 0:
                    next_state = np.zeros(self.resolution).astype('float')
                else:
                    next_state = self._preprocess(self.env.get_state().screen_buffer)
            else:
                done = False
                next_state = self._preprocess(self.env.get_state().screen_buffer)

            # save experience at lists
            self.states.append(state)
            self.actions.append(action)
            self.rewards_ext.append(tot_rew)
            self.next_states.append(next_state)
            self.dones.append(done)

            if done:
                break

        # return collected experiences
        return self.states, self.actions, self.rewards_ext, \
                self.next_states, self.dones

    def _reset(self,updated_actor,updated_critic):
        """
            -reset all type of monitorization lists at rollout level
            -updates actor and critic networks too;
                new weights are received from worker.py
        """
        # update agent --> model has changed
        self.actor.load_state_dict(updated_actor)
        self.critic.load_state_dict(updated_critic)

        self.states = []
        self.actions = []
        self.next_states = []
        self.rewards_int = []
        self.rewards_ext = []
        self.rewards_tot = []
        self.dones = []

        self.value_ext = []
        self.value_int = []

        self.stacked_states = []
        self.episode_finished = False
        self.agent_trajectory = []
        self.angle = []

        self.hidden_in = [] # stores (hidden_state,cell_state)
        self.hidden_out = []
        self.action_probability = []

    def _preprocess(self,img):
        """
            Converts and down-samples the input image
                - it is down-sampled in two steps to minimize information loss
                by doing it direct interpolation
        """
        orig_img = img

        im84 = cv2.resize(orig_img,
                           ((84,84)),
                           interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(im84,
                   (self.resolution),
                   interpolation=cv2.INTER_LINEAR)
        # normalize 0-1
        img = (img / 255.)
        return img


"""
    def recalculate_hidden_outputs(self, stacked_states, actions, action_prob, h_in,verbose=False):
        # Forward pass to model
        stacked_states = torch.from_numpy(stacked_states).to(self.device).float()
        actions = torch.from_numpy(actions).to(self.device)
        roles = torch.from_numpy(np.full(len(actions),self.worker_id)).to(self.device)
        with torch.no_grad():
            if self.use_lstm:
                # attention with lstm or just bidirectional lstm
                ve,vi,hout,enc_image,vkl = self.critic(state=stacked_states, hidden=h_in,action=actions,role=roles,memory_attention=self.att_history)
            else:
                # attention without lstm
                ve,vi,enc_image = self.critic(state=stacked_states,action=actions,role=roles,memory_attention=self.att_history)
                hout = 0
            [print('Q(s,a) value_ext:',ve) if verbose else 0]
            # [print('Probabilities:',action_prob) if verbose else 0]
            if self.use_centralized_critic:
                ve,vi = self.get_value_from_q(action_prob,ve,vi)
            [print('V(s) value_ext:',ve) if verbose else 0]
        return ve.data.cpu().clone().numpy(), \
               vi.data.cpu().clone().numpy(), \
               hout
"""
