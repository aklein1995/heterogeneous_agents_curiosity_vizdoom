#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: alain
"""
from __future__ import division
from __future__ import print_function

#lib
import configparser
import numpy as np
from time import time
import torch
import pickle
from copy import deepcopy
import shutil

#multiprocessing
import torch.multiprocessing as mp
# torch.multiprocessing.set_sharing_strategy('file_system') # to be able to open more than

# my classes
from src.agent_ppo import Agent_lstm as Agent
from src.curiosity import RNDModule,CountModule
from src.worker import Worker
from src.runner import Runner
from util_functions.utils import *
from util_functions.vizdoom_utils import getRoomsFromSettings

#******************************************************************************
verbose_test = False
reduction_factor = 16

# CONFIGURATION PARAMETERS
config = configparser.ConfigParser()
config.read('config.conf')

deep_monitorization = int(config['DEFAULT']['DeepMonitorization'])
num_workers = int(config['DEFAULT']['NumWorkers']) #number of parallel workers/environments to be run
config_file_path = []
for i in range(num_workers):
    config_file_path.append(config['DEFAULT']['EnvPath'+str(i)])

nsteps = int(config['DEFAULT']['MaxStepsRollout'])
num_parallel_envs = int(config['DEFAULT']['NumParallelEnvs'])
n_episodes = int(config['DEFAULT']['NumEpisodes'])
max_rollout_len = int(config['DEFAULT']['MaxStepsRollout'])
max_episode_len = int(config['DEFAULT']['MaxStepsEpisode'])
frame_repeat = int(config['DEFAULT']['FrameSkip'])
stack_size = int(config['DEFAULT']['FrameStack'])
stack_mode = int(config['DEFAULT']['StackMode'])
resolution = (int(config['DEFAULT']['Resolution_H']), int(config['DEFAULT']['Resolution_W']))
dumpevery = int(config['DEFAULT']['DumpEvery'])
dump_models_every = int(config['DEFAULT']['DumpModelsEvery'])
room_setting = config['DEFAULT']['Room_setting']
use_centralized_critic = int(config['DEFAULT']['Use_centralized_critic'])

# Evaluation/Testing
train_actor = int(config['DEFAULT']['trainActor'])
train_critic = int(config['DEFAULT']['trainCritic'])
load_actor = int(config['DEFAULT']['loadActor'])
load_critic = int(config['DEFAULT']['loadCritic'])

path_actor,path_critic = [],[]
path_actor.append(config['DEFAULT']['pathActor0'])
path_critic.append(config['DEFAULT']['pathCritic0'])
path_actor.append(config['DEFAULT']['pathActor1'])
path_critic.append(config['DEFAULT']['pathCritic1'])

# Actor params
num_minibatches = int(config['DEFAULT']['NumMiniBatches'])
clip_grad_norm = float(config['DEFAULT']['ClipGradNorm'])
ac_learning_rate = float(config['DEFAULT']['ACLearningRate'])
epoch = int(config['DEFAULT']['Epoch'])
entropy = float(config['DEFAULT']['Entropy'])
ppo_eps = float(config['DEFAULT']['PPOEps'])
lmbda = float(config['DEFAULT']['Lambda'])
gamma = float(config['DEFAULT']['Gamma'])
gamma_int = float(config['DEFAULT']['INTGamma'])
use_gae = int(config['DEFAULT']['UseGAE'])
int_coef = float(config['DEFAULT']['IntCoef'])
ext_coef = float(config['DEFAULT']['ExtCoef'])
balance_exploration_adaptative = int(config['DEFAULT']['BalanceExplorationAdaptative'])

# GPU/CPU
use_gpu = int(config['DEFAULT']['UseGPU'])
gpu_device_id = int(config['DEFAULT']['GPUDeviceID'])
device = torch.device('cuda:'+str(gpu_device_id) if use_gpu else 'cpu')

# Input params
use_stacked_frames = int(config['DEFAULT']['UseStackedFrames'])
inp_frames = stack_size if use_stacked_frames else 1 # stack frames (4) or not (1) for input/obs

#Curiosity params
curiosity_type = config['DEFAULT']['Curiosity_type']
curiosity_subtype = config['DEFAULT']['Curiosity_subtype']
multi_head_action_curiosity = int(config['DEFAULT']['ActionBasedCuriosity'])
obs_norm_episodes = int(config['DEFAULT']['ObsNormEp'])
constant_std_for_normalization = int(config['DEFAULT']['NormalizationStdConstant'])
normalization_std = int(config['DEFAULT']['NormalizationStd'])
intrinsic_rew_episodic = int(config['DEFAULT']['EpisodicIntrinsicRew'])
rnd_learning_rate = float(config['DEFAULT']['RNDLearningRate'])
update_proportion = float(config['DEFAULT']['UpdateProportion'])
decaying_base_worker_curiosity = int(config['DEFAULT']['DecayBaseWorkerCuriosity'])
tree_filtering = int(config['DEFAULT']['Tree_filtering'])

# LSTM parameters
use_lstm = int(config['DEFAULT']['use_lstm'])
lstm_hidden_size = int(config['DEFAULT']['hidden_size'])
lstm_bidirectional = int(config['DEFAULT']['bidirectional'])

# Input space for critic
use_actions_as_input = int(config['DEFAULT']['Action_as_input'])
use_role_as_input = int(config['DEFAULT']['Role_as_input'])

# Available Actions --> #use ability is the 4th digit (based on .cfg file)
available_actions = [
                    [[0,0,0,0], # noop
                    [1,0,0,0], # forward
                    [0,1,0,0], # turn left
                    [0,0,1,0], # turn right
                    [0,0,0,1], # use
                    ],
                    [[0,0,0,0], # noop
                    [1,0,0,0], # forward
                    [0,1,0,0], # turn left
                    [0,0,1,0], # turn right
                    ],
                    ]
action_sizes = [len(av_act) for av_act in available_actions]
#******************************************************************************

if __name__ == '__main__':
    """
        https://docs.python.org/3.5/library/multiprocessing.html#contexts-and-start-methods
        https://docs.python.org/3.5/library/multiprocessing.html#multiprocessing.set_start_method
        https://discuss.pytorch.org/t/cuda-multiprocessing-training-multiple-model-in-different-processes-in-single-gpu/19921/2
    """
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        print('Error en la inicialización de multiprocessing')

    # generate MONITORIZATION: folder
    folder_path_results, models_folder_path = generateTestFolder()
    shutil.copyfile('config.conf',folder_path_results+'/config.conf')

    # =============================================================================
    # CENTRALIZED CRITIC
    # =============================================================================
    # for centralized_critic lstm
    # -- define it although not using centralized critic (will not be actually used but is required for no code errors)
    hidden_states = [[] for _ in range(num_workers)]
    index_runner = [[] for _ in range(num_workers)]

    centralized_critic_output_size = np.max(action_sizes)
    if use_centralized_critic:
        centralized_critic = Agent(input_size = inp_frames,
                                  output_size = centralized_critic_output_size,
                                  epoch = epoch,
                                  learning_rate = ac_learning_rate,
                                  ent_coef=entropy,
                                  ppo_eps=ppo_eps,
                                  clip_grad_norm = clip_grad_norm,
                                  use_cuda = use_gpu,
                                  device_id=gpu_device_id,
                                  num_minibatches = num_minibatches,
                                  use_centralized_critic = use_centralized_critic,
                                  im_centralized_critic = 1,
                                  train_actor = train_actor,
                                  load_actor = load_actor,
                                  path_actor = path_actor[0],
                                  train_critic = train_critic,
                                  load_critic = load_critic,
                                  path_critic = path_critic[0],
                                  use_lstm= use_lstm,
                                  use_action_as_input=use_actions_as_input,
                                  use_role_as_input = use_role_as_input,
                                  lstm_hidden_size=lstm_hidden_size,
                                  lstm_bidirectional=lstm_bidirectional,
                                  num_parallel_envs=num_parallel_envs,
                                  verbose_test=verbose_test
                                  )
    # **************************************************************************
    # CURIOSITY RELATED
    # **************************************************************************
    if curiosity_type == 'individual':
        # Decentralised
        decentralised_obs_rms = []
        decentralised_reward_rms = []
        decentralised_rff_int = []
        decentralised_curiosity_modules = []

        for wi in range(num_workers):
            # output_rnd = len(available_actions[wi])
            output_rnd = np.max(action_sizes)
            decentralised_obs_rms.append(RunningMeanStd(shape=(42,42)))
            decentralised_reward_rms.append(RunningMeanStd())
            decentralised_curiosity_modules.append(
                                        RNDModule(input_size = 1,
                                        output_size = output_rnd,
                                        learning_rate = rnd_learning_rate,
                                        update_proportion = update_proportion,
                                        use_cuda = use_gpu,
                                        device_id=gpu_device_id,
                                        multi_head_action=multi_head_action_curiosity)
                                        )
            print('{} Decentralised Curiosity Module defined'.format(wi))
            decentralised_rff_int.append([]) # Reward Forward Filter individual for each Runner
            for ri in range(num_parallel_envs):
                decentralised_rff_int[wi].append(RewardForwardFilter(gamma_int))

    centralized_curiosity_output_size = centralized_critic_output_size = np.max(action_sizes)
    if curiosity_type == 'centralized':
        # Centralised
        centralised_obs_rms = RunningMeanStd(shape=(42,42))
        centralised_reward_rms = RunningMeanStd()
        centralised_curiosity_module = RNDModule(input_size = 1,
                                    output_size = centralized_curiosity_output_size,
                                    learning_rate = rnd_learning_rate,
                                    update_proportion = update_proportion/num_workers,
                                    use_cuda = use_gpu,
                                    device_id=gpu_device_id,
                                    multi_head_action=multi_head_action_curiosity)
        print('Centralised Curiostiy Module defined')
        centralised_rff_int = []
        for wi in range(num_workers):
            centralised_rff_int.append([])
            for ri in range(num_parallel_envs):
                centralised_rff_int[wi].append(RewardForwardFilter(gamma_int))

    if curiosity_type == 'count_i':
        visitcount_module = []
        reward_rms = []
        rff_int = []
        for wi in range(num_workers):
            visitcount_module.append(CountModule(action_based=multi_head_action_curiosity,
                                                 num_parallel_envs=num_parallel_envs,
                                                 take_into_account_action_decay = False))
            reward_rms.append(RunningMeanStd())
            print('Visit Count Module {} defined'.format(wi))

            rff_int.append([])
            for ri in range(num_parallel_envs):
                rff_int[wi].append(RewardForwardFilter(gamma_int))


    if curiosity_type == 'count_c':
        visitcount_module = CountModule(action_based=multi_head_action_curiosity,
                                        num_parallel_envs=num_parallel_envs,
                                        take_into_account_action_decay = tree_filtering)
        reward_rms = RunningMeanStd()
        rff_int = []
        for wi in range(num_workers):
            rff_int.append([])
            for ri in range(num_parallel_envs):
                rff_int[wi].append(RewardForwardFilter(gamma_int))

        print('Visit Count Module defined')

    # =============================================================================
    # GENERATE WORKER AND RUNNERS
    # =============================================================================
    parent_connections = []
    child_connections = []
    workers = []
    closed_connections = []
    worker_ids = []

    episodes_counter = np.zeros(num_workers).astype(int)
    train_steps_counter = np.zeros(num_workers).astype(int)
    agents = []

    for worker_id in range(num_workers):
        actor_action_size = action_sizes[worker_id]
        agent = Agent(input_size = inp_frames,
                      output_size = actor_action_size,
                      epoch = epoch,
                      learning_rate = ac_learning_rate,
                      ent_coef = entropy,
                      ppo_eps = ppo_eps,
                      use_cuda = use_gpu,
                      clip_grad_norm = clip_grad_norm,
                      device_id=gpu_device_id,
                      num_minibatches = num_minibatches,
                      use_centralized_critic = use_centralized_critic,
                      im_centralized_critic = 0,
                      train_actor = train_actor,
                      load_actor = load_actor,
                      path_actor = path_actor[worker_id],
                      train_critic = train_critic,
                      load_critic = load_critic,
                      path_critic = path_critic[worker_id],
                      use_lstm=use_lstm,
                      use_action_as_input=use_actions_as_input,
                      use_role_as_input=use_role_as_input,
                      lstm_hidden_size=lstm_hidden_size,
                      lstm_bidirectional=lstm_bidirectional,
                      num_parallel_envs=num_parallel_envs,
                      verbose_test=verbose_test)
        agents.append(agent)

    for worker_id in range(num_workers):
        actor_action_size = action_sizes[worker_id]
        av_actions = available_actions[worker_id]
        # worker generation
        parent_conn, child_conn = mp.Pipe()
        theagent = agents[worker_id]
        # create parallel runners to collect rollout/trajectories
        runners = []
        runners_ids = []
        runner_parent_connections = []
        for runner_id in range(num_parallel_envs):
            pconn, chconn = mp.Pipe()
            runner = mp.Process(target = Runner,
                        args = (nsteps,
                                inp_frames,
                                actor_action_size,
                                (theagent.actor.state_dict(),theagent.critic.state_dict()),
                                config_file_path[worker_id],
                                worker_id,
                                runner_id,
                                chconn,
                                av_actions,
                                max_episode_len,
                                resolution,
                                inp_frames,
                                use_stacked_frames,
                                frame_repeat,
                                use_gpu,
                                gpu_device_id,
                                getRoomsFromSettings(room_setting),
                                use_centralized_critic,
                                centralized_critic_output_size,
                                use_actions_as_input,
                                use_role_as_input,
                                use_lstm,
                                lstm_hidden_size,
                                lstm_bidirectional,
                                verbose_test
                                )
                        )
            runners.append(runner)
            runners_ids.append(runner_id)
            runner_parent_connections.append(pconn)
        [r.start() for r in runners]
        print('Runners of worker {} initialized'.format(worker_id))

        worker = mp.Process(target = Worker,
                              args=(worker_id,
                                    child_conn,
                                    num_parallel_envs,
                                    theagent,
                                    av_actions,
                                    resolution,
                                    max_episode_len,
                                    use_stacked_frames,
                                    inp_frames,
                                    stack_mode,
                                    nsteps,
                                    runners,
                                    runners_ids,
                                    runner_parent_connections,
                                    use_gae,
                                    use_gpu,
                                    gpu_device_id,
                                    rnd_learning_rate,
                                    curiosity_type,
                                    use_centralized_critic,
                                    lmbda,
                                    gamma,
                                    gamma_int,
                                    ext_coef,
                                    int_coef,
                                    balance_exploration_adaptative,
                                    episodes_counter[worker_id],
                                    intrinsic_rew_episodic,
                                    constant_std_for_normalization,
                                    normalization_std,
                                    obs_norm_episodes,
                                    use_lstm,
                                    centralized_curiosity_output_size
                                    )
                              )

        worker_ids.append(worker_id)
        parent_connections.append(parent_conn)
        child_connections.append(child_conn)
        workers.append(worker)
        closed_connections.append(False)

    [w.start() for w in workers]

    print('\n*** INFO ***')
    print('Pipes & Workers generated!')
    print('Worker_ids:',worker_ids)
    print('Actions spaces:')
    for i in worker_ids:
        print('W{}: {}'.format(i,available_actions[i]))
    print('Curiosity-type:',curiosity_type)
    if curiosity_type == 'individual':
        print('Curiosity_subtype:',curiosity_subtype)
    print('Critic centralized: ',use_centralized_critic)


    # =============================================================================
    # MONITORIZATION VARIABLES
    # =============================================================================
    init_ep_time = [0 for _ in range(num_workers)]
    time_worker = [[] for _ in range(num_workers)]
    active_worker = [False] * num_workers
    active_parallel_envs = [num_parallel_envs*[True] for _ in range(num_workers)]
    episode_finished = [True] * num_workers
    rollout_finished = [False] * num_workers

    room_init = [[] for _ in range(num_workers)]
    steps_in_current_episode = [[] for _ in range(num_workers)]
    coordinates = [[] for _ in range(num_workers)]
    angle = [[] for _ in range(num_workers)]

    advantage_total = [[] for _ in range(num_workers)]
    advantage_ext = [[] for _ in range(num_workers)]
    disc_return_ext = [[] for _ in range(num_workers)]
    value_ext = [[] for _ in range(num_workers)]
    actor_loss = [[] for _ in range(num_workers)]
    critic_ext_loss = [[] for _ in range(num_workers)]
    critic_int_loss = [[] for _ in range(num_workers)]
    entropy  = [[] for _ in range(num_workers)]
    scores = [[] for _ in range(num_workers)]
    total_loss = [[] for _ in range(num_workers)]

    grad_actor = [[] for _ in range(num_workers)]
    grad_critic = [[] for _ in range(num_workers)]
    clip_info = [[] for _ in range(num_workers)]
    policy_probs = [[] for _ in range(num_workers)]
    probability_move_up = [[] for _ in range(num_workers)]
    probability_move_right = [[] for _ in range(num_workers)]
    probability_move_left = [[] for _ in range(num_workers)]
    entropy_step = [[] for _ in range(num_workers)]
    selected_action = [[] for _ in range(num_workers)]
    vext_pre_train = [[] for _ in range(num_workers)]
    vext_post_train = [[] for _ in range(num_workers)]

    # intrinsic motivation related
    curiosity_loss = [[] for _ in range(num_workers)]
    value_int = [[] for _ in range(num_workers)]
    advantage_int = [[] for _ in range(num_workers)]
    disc_return_int = [[] for _ in range(num_workers)]
    intrinsic_rewards = [[] for _ in range(num_workers)]
    intrinsic_rewards_normalized = [[] for _ in range(num_workers)]
    standard_deviation_intrinsic = [[] for _ in range(num_workers)]

    #KL RELATED
    value_kl = [[] for _ in range(num_workers)]
    rewards_kl = [[] for _ in range(num_workers)]
    advantage_kl = [[] for _ in range(num_workers)]
    disc_return_kl = [[] for _ in range(num_workers)]

    # =============================================================================
    # PRE-INIT REQUIREMENTS
    # =============================================================================
    # ***SAVE TARGET_RNDs
    # Decentralised
    if curiosity_type == 'individual':
        for worker_id in range(num_workers):
            target_rnd = decentralised_curiosity_modules[worker_id].rnd.target.state_dict()
            torch.save(target_rnd,models_folder_path + '/agent'+ str(worker_id) +'_decentralised_target_rnd.pth')
    #Centralised
    if curiosity_type == 'centralized':
        target_rnd = centralised_curiosity_module.rnd.target.state_dict()
        torch.save(target_rnd,models_folder_path + '/centralised_target_rnd.pth')

    # ***PRE-TRAINING PHASE FOR OBS NORMALIZATION (CURIOSITY)
    # Decentralised
    if curiosity_type == 'individual':
        for worker_id,parent_conn in zip(worker_ids,parent_connections):
            print('Pre-train initialization at main...')
            args = obs_norm_episodes,multi_head_action_curiosity
            parent_conn.send(('pre-train', args))
            cmd, args = parent_conn.recv()
            if cmd == 'ack-pre-train':
                obs_rms = args
                decentralised_obs_rms[worker_id] = obs_rms
    # Centralised --> we just get a copy of one of them
    if curiosity_type == 'centralized':
        parent_conn = parent_connections[0]
        print('Pre-train initialization at main...')
        args = obs_norm_episodes,multi_head_action_curiosity
        parent_conn.send(('pre-train', args))
        cmd, args = parent_conn.recv()
        if cmd == 'ack-pre-train':
            obs_rms = args
            centralised_obs_rms = obs_rms

    # =============================================================================
    # Training begins - LOOP INITIALIZATED!!!
    # =============================================================================
    print('Train begins!\n')
    time_start = time()

    while True:
        # Finish loop
        if np.all(closed_connections):
            print('All connections have been closed')
            break
        else:
            # Dump Results for the beginning of a new episode
            for worker_id,parent_conn in zip(worker_ids,parent_connections):
                if np.all(episode_finished): # all workers in idle mode --> all false
                    if episodes_counter[worker_id] > 0 and (episodes_counter[worker_id] % dumpevery==0):
                        doc_tail = dumpStringTail(episodes_counter[worker_id])

                        if deep_monitorization:
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'actor_gradients' + doc_tail,grad_actor[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'critic_gradients' + doc_tail,grad_critic[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'clip_info' + doc_tail,clip_info[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'advantages_total' + doc_tail,advantage_total[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'room_init' + doc_tail, room_init[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'returns_ext' + doc_tail,disc_return_ext[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'angle' + doc_tail,angle[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'critic_ext_loss' + doc_tail,critic_ext_loss[worker_id])

                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'entropy_loss' + doc_tail,entropy[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'total_loss' + doc_tail,total_loss[worker_id])
                            # intrinsic values related
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'curiosity_loss' + doc_tail,curiosity_loss[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'returns_int' + doc_tail,disc_return_int[worker_id])
                            DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'critic_int_loss' + doc_tail,critic_int_loss[worker_id])

                        print('Dumping...')
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'actor_loss' + doc_tail, actor_loss[worker_id])

                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'policy_probs' + doc_tail,policy_probs[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'time_in_episode' + doc_tail, time_worker[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'scores_in_episode' + doc_tail, scores[worker_id][-dumpevery:])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'steps_in_episode' + doc_tail, steps_in_current_episode[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'value_ext' + doc_tail,value_ext[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'coordinates' + doc_tail,coordinates[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'selected_action' + doc_tail,selected_action[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'advantages_ext' + doc_tail,advantage_ext[worker_id])

                        # intrinsic values related
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'value_int' + doc_tail,value_int[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'rewards_int' + doc_tail,intrinsic_rewards[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'std_intrinsic' + doc_tail,standard_deviation_intrinsic[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'advantages_int' + doc_tail,advantage_int[worker_id])
                        DumpToPickle(folder_path_results + '/w' + str(worker_id) + '_' + 'rewards_int_normalized' + doc_tail,intrinsic_rewards_normalized[worker_id])

                        # Models
                        if episodes_counter[worker_id] % dump_models_every == 0:
                            rnd_params = None
                            if curiosity_type == 'centralized':
                                rnd_params = centralised_curiosity_module.rnd.predictor.to('cpu').state_dict()
                                centralised_curiosity_module.rnd.predictor.to(device)
                            elif curiosity_type == 'individual':
                                rnd_params = decentralised_curiosity_modules[worker_id].rnd.predictor.to('cpu').state_dict()
                                decentralised_curiosity_modules[worker_id].rnd.predictor.to(device)
                            elif curiosity_type == 'count_i':
                                bin_dict = visitcount_module[worker_id]
                                with open(models_folder_path + '/count_bins_w' + str(worker_id) + '_' + str(episodes_counter[worker_id]),'wb+') as f:
                                    pickle.dump(bin_dict,f)
                            elif curiosity_type == 'count_c' and worker_id == 0:
                                bin_dict = visitcount_module
                                with open(models_folder_path + '/count_bins_w' + str(worker_id) + '_' + str(episodes_counter[worker_id]),'wb+') as f:
                                    pickle.dump(bin_dict,f)
                            dumpModels(parent_conn,episodes_counter[worker_id],worker_id,models_folder_path,dump_models_every,rnd_params)
                        print('Finished!')


                        # ***restart monitorization variables for new episode (every time we dump to file)
                        advantage_total[worker_id] = []

                        advantage_ext[worker_id] = []
                        disc_return_ext[worker_id] = []
                        value_ext[worker_id] = []

                        actor_loss[worker_id] = []
                        critic_ext_loss[worker_id] = []
                        entropy[worker_id] = []
                        total_loss[worker_id] = []
                        coordinates[worker_id] = []
                        angle[worker_id] = []
                        steps_in_current_episode[worker_id] = []
                        room_init[worker_id] = []

                        grad_critic[worker_id] = []
                        grad_actor[worker_id] = []
                        clip_info[worker_id] = []
                        policy_probs[worker_id] = []
                        probability_move_up[worker_id] = []
                        probability_move_right[worker_id] = []
                        probability_move_left[worker_id] = []
                        entropy_step[worker_id] = []
                        selected_action[worker_id] = []
                        vext_pre_train[worker_id] = []
                        vext_post_train[worker_id] = []
                        curiosity_loss[worker_id] = []

                        advantage_int[worker_id] = []
                        value_int[worker_id] = []
                        disc_return_int[worker_id] = []
                        critic_int_loss[worker_id] = []
                        intrinsic_rewards[worker_id] = []
                        intrinsic_rewards_normalized[worker_id] = []
                        standard_deviation_intrinsic[worker_id] = []

                        value_kl[worker_id] = []
                        advantage_kl[worker_id] = []
                        rewards_kl[worker_id] = []
                        disc_return_kl[worker_id] = []

                        #time related
                        time_worker[worker_id] = []

                    # time related
                    init_ep_time[worker_id] = time()

                    # active workers/runners state
                    active_worker[worker_id] = True
                    active_parallel_envs[worker_id] = num_parallel_envs*[True]

                    # hidden state for centralized critic
                    hidden_states[worker_id] = []
                    index_runner[worker_id] = []
                    if use_centralized_critic and use_lstm:
                        for _ in range(num_parallel_envs):
                            hidden_states[worker_id].append(centralized_critic.init_lstm_hidden())
                            # hidden_states[worker_id] = centralized_critic.init_lstm_hidden()
                    elif use_centralized_critic and not use_lstm:
                        for _ in range(num_parallel_envs):
                            hidden_states[worker_id].append((torch.zeros(1),torch.zeros(1)))

                    # Update counts centralized if mask used for different actions
                    if curiosity_type == 'count_c':
                        visitcount_module.update_action_dependant(worker_id)

                    # set episodic intrinsic reward (rff module for int returns calculations)
                    if False:
                        if curiosity_type == 'individual':
                            for ri in range(num_parallel_envs):
                                decentralised_rff_int[worker_id][ri] = RewardForwardFilter(gamma_int)
                        if curiosity_type == 'centralized':
                            for ri in range(num_parallel_envs):
                                centralised_rff_int[worker_id][ri] = RewardForwardFilter(gamma_int)
                        if curiosity_type == 'count_c' or curiosity_type == 'count_i':
                            for ri in range(num_parallel_envs):
                                rff_int[worker_id][ri] = RewardForwardFilter(gamma_int)
                    # send new episode signal
                    send_args = (episodes_counter[worker_id])
                    parent_conn.send(('new_episode', send_args))
                    cmd, args = parent_conn.recv()
                    if cmd == 'ack':
                        room = args
                        room_init[worker_id].append(room)
                        continue
            # =============================================================================
            # NEW ROLLOUT/MINIBATCH
            # =============================================================================
            intrinsic_returns_rollout = [] #we want to have each runner's (and workers) return separtely for normalization
            intrinsic_rewards_rollout = []
            intrinsic_rewards_norm_rollout = []
            # when using decentralised combination of rewards we need auxiliary buffers to store the values of the other predictor
            intrinsic_rewards_auxworker_rollout = []
            intrinsic_rewards_norm_auxworker_rollout = []

            for wi in range(num_workers):
                intrinsic_returns_rollout.append([])
                intrinsic_rewards_rollout.append([])
                intrinsic_rewards_norm_rollout.append([])
                intrinsic_rewards_auxworker_rollout.append([])
                intrinsic_rewards_norm_auxworker_rollout.append([])
                # generate only for ACTIVE RUNNERs because it can harm some parts of code -- not prepared to deal with empty arays in some cases
                for ri in range(np.sum(active_parallel_envs[wi])):
                    intrinsic_returns_rollout[wi].append([])
                    intrinsic_rewards_rollout[wi].append([])
                    intrinsic_rewards_norm_rollout[wi].append([])
                    intrinsic_rewards_auxworker_rollout[wi].append([])
                    intrinsic_rewards_norm_auxworker_rollout[wi].append([])

            experiences = [[] for _ in range(num_workers)]
            stacked_states = [[] for _ in range(num_workers)]
            actions = [[] for _ in range(num_workers)]
            next_obs = [[] for _ in range(num_workers)]
            current_obs = [[] for _ in range(num_workers)]
            rext = [[] for _ in range(num_workers)]
            rint = [[] for _ in range(num_workers)]
            rkl = [[] for _ in range(num_workers)]
            # keeping separated runner values
            next_observations_rollout_runner = [[] for _ in range(num_workers)]
            coordinates_rollout_runner = [[] for _ in range(num_workers)]
            actions_rollout_runner = [[] for _ in range(num_workers)]
            observations_rollout_runner = [[] for _ in range(num_workers)]


            for worker_id,parent_conn in zip(worker_ids,parent_connections):
                if active_worker[worker_id]:
                    rollout_finished[worker_id] = False

                    # SEND CENTRALIZED CRITIC
                    args = centralized_critic.critic.to('cpu').state_dict() if use_centralized_critic else 0
                    [centralized_critic.critic.to(device) if use_centralized_critic else 0]
                    parent_conn.send(('new_rollout',args))

                    # receive rollout experiences
                    cmd, args = parent_conn.recv()
                    if cmd == 'rollout_finished':
                        rollout_finished[worker_id] = True
                        # get worker params
                        rollout, score, episode_flag, train_steps, trajectory, ang, steps,active_runners, trajectory_at_runner = args

                        train_steps_counter[worker_id] = train_steps
                        episode_finished[worker_id] = episode_flag
                        experiences[worker_id].append(rollout)
                        coordinates[worker_id].extend(trajectory)
                        coordinates_rollout_runner[worker_id] = trajectory_at_runner #used for filtering at intrinsic calculations
                        angle[worker_id].extend(ang)
                        selected_action[worker_id].extend(np.asarray(rollout[1]).flatten(order='C'))
                        active_parallel_envs[worker_id] = active_runners

                        # Get experiences
                        next_observations_rollout_runner[worker_id] = next_observations = rollout[2] #next_observations
                        st_s = rollout[5] #stacked_states
                        actions_rollout_runner[worker_id] = act = rollout[1] #actions
                        observations_rollout_runner[worker_id] = observations = rollout[0] #observations

                        # Reshape to train critic -- from 3X50 TO 150 (assuming 3 parallel envs and 50 rollout size)
                        actions[worker_id].extend(np.asarray([rollout_at_runner for runner_level in act for rollout_at_runner in runner_level]))
                        current_obs[worker_id].extend(np.asarray([rollout_at_runner for runner_level in observations for rollout_at_runner in runner_level])) #1 runner: from [1,50,42,42] to [1*50,42,42]
                        next_obs[worker_id].extend(np.asarray([rollout_at_runner for runner_level in next_observations for rollout_at_runner in runner_level])) #1 runner: from [1,50,42,42] to [1*50,42,42]
                        stacked_states[worker_id].extend(np.asarray([rollout_at_runner for runner_level in st_s for rollout_at_runner in runner_level]))

                        # if episode finished
                        if episode_flag:
                            episodes_counter[worker_id] += 1
                            scores[worker_id].append(score)
                            steps_in_current_episode[worker_id].append(steps)

            # --> all type of workers have had to finish their rollout and calculate their intrinsic rewards
            while True:
                if np.all(rollout_finished):
                    break

            # =============================================================================
            # INTRINSIC REWARDS CALCULATION
            # =============================================================================
            # *****Decentralised*****
            if curiosity_type == 'individual':
                for worker_id in worker_ids:
                    if active_worker[worker_id]:
                        # *** 1. CALCULATE RAW INTRINSIC REWARDS - we need raw obs with separated stream of samples between runners
                        obs_cur = observations_rollout_runner[worker_id] if multi_head_action_curiosity else next_observations_rollout_runner[worker_id]
                        act = actions_rollout_runner[worker_id]
                        # get values for current worker
                        intrinsic_rewards_rollout[worker_id],intrinsic_returns_rollout[worker_id],decentralised_rff_int[worker_id] = \
                            intrinsic_reward_calculation(active_runners=active_parallel_envs[worker_id],\
                                                        curiosity_module=decentralised_curiosity_modules[worker_id],\
                                                        obs_rms=decentralised_obs_rms[worker_id],\
                                                        rff_int_runners=decentralised_rff_int[worker_id],\
                                                        obs_cur=obs_cur,\
                                                        acts=act)
                        # ***2. NORMALIZATION
                        # 2.1.update obs normalization
                        obs_for_curiosity = np.array(current_obs[worker_id]) if multi_head_action_curiosity else np.array(next_obs[worker_id])
                        actions_for_curiosity = np.array(actions[worker_id])
                        decentralised_obs_rms[worker_id].update_from_moments(np.mean(obs_for_curiosity,axis=0),
                                                                            np.var(obs_for_curiosity,axis=0),
                                                                            np.array(obs_for_curiosity).shape[0])

                        # 2.2.stats calculation --> intrinsic returns calculated per runner
                        for returns_runner in intrinsic_returns_rollout[worker_id]:
                            if len(returns_runner) > 0: #for security
                                batch_mean = np.mean(returns_runner)
                                batch_var = np.var(returns_runner)
                                batch_count = len(returns_runner)
                                decentralised_reward_rms[worker_id].update_from_moments(batch_mean=batch_mean,batch_var=batch_var,batch_count=batch_count)
                        std_normalization = np.sqrt(decentralised_reward_rms[worker_id].var)
                        standard_deviation_intrinsic[worker_id].append(std_normalization) #monitorization

                        # 2.3. Normalization of rewards
                        normalized_int_rewards = [] # rews per runner
                        for int_rews_runner in intrinsic_rewards_rollout[worker_id]:
                            normalized_int_rewards.append(np.asarray(int_rews_runner)/std_normalization)
                        intrinsic_rewards_norm_rollout[worker_id] = normalized_int_rewards # save them to send them next to worker

                        # 2.4. monitorization -- at episode level
                        intrinsic_rewards[worker_id].extend(np.asarray([rollout_at_runner for runner_level in intrinsic_rewards_rollout[worker_id] for rollout_at_runner in runner_level]))
                        intrinsic_rewards_normalized[worker_id].extend(np.asarray([rollout_at_runner for runner_level in normalized_int_rewards for rollout_at_runner in runner_level]))

                        # ***3. TRAIN CURIOSITY MODULE
                        rnd_loss = decentralised_curiosity_modules[worker_id].train(
                                    ((obs_for_curiosity-decentralised_obs_rms[worker_id].mean)/np.sqrt(decentralised_obs_rms[worker_id].var)).clip(-5, 5),
                                    actions_for_curiosity)
                        curiosity_loss[worker_id].append(rnd_loss)

                        # (OPTIONAL)*** 4. Get intrinsic reward values for the other predictor
                        # (This step only necessary when combining decentralised individual rewards)
                        if curiosity_subtype != 'independent':
                            auxworker = 1 if worker_id == 0 else 0
                            # take the other worker modules
                            intrinsic_rewards_auxworker_rollout[worker_id], _ ,_ = \
                                intrinsic_reward_calculation(active_runners=active_parallel_envs[worker_id],\
                                                            curiosity_module=decentralised_curiosity_modules[auxworker],\
                                                            obs_rms=decentralised_obs_rms[auxworker],\
                                                            rff_int_runners=decentralised_rff_int[auxworker],\
                                                            obs_cur=obs_cur,\
                                                            acts=act)

                        # (OPTIONAL)*** 5. Check if at rollout have been states related with a given state & action
                        # (This is to analyse how passing observations gathered by one agent (W1) to the other (W0) for training the RND Module, which is supposed decrease the novelty faster)
                        if decaying_base_worker_curiosity and worker_id == 0:
                            additional_obs_for_curiosity = []
                            additional_next_obs_for_curiosity = []
                            additional_actions_for_curiosity = []
                            # inside the scope of the rollout
                            for coords_r,actions_r,obs_r,next_obs_r in zip(coordinates_rollout_runner[worker_id],actions_rollout_runner[worker_id],observations_rollout_runner[worker_id],next_observations_rollout_runner[worker_id]):
                                # inside the scope of the runner
                                aux_share_experiences = False
                                for (x,y),a,o in zip(coords_r,actions_r,obs_r):
                                    if a == 4:# check only when executing action USE = [0,0,0,1] with id 4
                                        c1,c2 = (int(x)-(160)) // reduction_factor, (int(y)+992)// reduction_factor # shortmap
                                        # add that rollout to be trained on other RND
                                        if (c1 >= 34 and c1 <= 37) and (c2>=48 and c2<=54):
                                        # if (c1 >= 30 and c1 <= 31) and (c2>=0 and c2<=10):
                                            print('Entra!')
                                            aux_share_experiences = True
                                            break
                                if aux_share_experiences:
                                    additional_obs_for_curiosity.extend(obs_r)
                                    additional_next_obs_for_curiosity.extend(next_obs_r)
                                    additional_actions_for_curiosity.extend(actions_r)

                # for loop finished (each worker already computed everything)

                # (OPTIONAL) Continues step 4
                # print('\ninit norm values:',intrinsic_rewards_norm_rollout)
                for worker_id in worker_ids:
                    if active_worker[worker_id] and curiosity_subtype != 'independent':

                        auxworker = 1 if worker_id == 0 else 0

                        # 4.1 Normalize rewards of auxiliary-intrinsic rewards
                        std = standard_deviation_intrinsic[auxworker][-1]
                        normalized_int_rewards = [] # rews per runner
                        for int_rews_runner in intrinsic_rewards_auxworker_rollout[worker_id]:
                            normalized_int_rewards.append(np.asarray(int_rews_runner)/std)
                        intrinsic_rewards_norm_auxworker_rollout[worker_id] = normalized_int_rewards # save them to send them next to worker

                        # 4.2. Apply the specified combination of curiosity
                        combined_int_rewards_runner = []
                        for current_worker_norm_rews_runner, aux_worker_norm_rews_runner in zip(intrinsic_rewards_norm_rollout[worker_id],intrinsic_rewards_norm_auxworker_rollout[worker_id]):

                            combined_int_rewards = []

                            for idx in range(len(current_worker_norm_rews_runner)):
                                if curiosity_subtype == 'minimum':
                                    value = min(current_worker_norm_rews_runner[idx],aux_worker_norm_rews_runner[idx])
                                elif curiosity_subtype == 'covering': # more novel than avg
                                    avg = np.mean([current_worker_norm_rews_runner[idx],aux_worker_norm_rews_runner[idx]])
                                    value = current_worker_norm_rews_runner[idx] if current_worker_norm_rews_runner[idx] > avg else 0
                                elif curiosity_subtype == 'burrowing': # less novel than avg
                                    avg = np.mean([current_worker_norm_rews_runner[idx],aux_worker_norm_rews_runner[idx]])
                                    value = current_worker_norm_rews_runner[idx] if current_worker_norm_rews_runner[idx] < avg else 0
                                combined_int_rewards.append(value)
                            combined_int_rewards_runner.append(combined_int_rewards)

                        intrinsic_rewards_norm_rollout[worker_id] = combined_int_rewards_runner


                # (OPTIONAL) Continuation of step 5. -- ONCE EACH AGENT CURIOSITY HAS BEEN ALREADY TRAINED INDEPENDENTLY ... here, additional pass
                if num_workers > 1 and decaying_base_worker_curiosity:
                    if len(additional_actions_for_curiosity) > 0: # in 2 steps because this param may not exist without "decaying_base_worker_curiosity"
                        print(additional_actions_for_curiosity)
                        input_curiosity = additional_obs_for_curiosity if multi_head_action_curiosity else additional_next_obs_for_curiosity

                        rnd_loss = decentralised_curiosity_modules[1].train(
                                    ((input_curiosity-decentralised_obs_rms[1].mean)/np.sqrt(decentralised_obs_rms[1].var)).clip(-5, 5),
                                    additional_actions_for_curiosity)

            # *****Centralised*****
            elif curiosity_type == 'centralized':
                for worker_id in worker_ids:
                    if active_worker[worker_id]:
                        # *** 1. CALCULATE RAW INTRINSIC REWARDS - we need raw obs with separated stream of samples between runners
                        obs_cur = observations_rollout_runner[worker_id] if multi_head_action_curiosity else next_observations_rollout_runner[worker_id]
                        act = actions_rollout_runner[worker_id]
                        # get values for current worker
                        intrinsic_rewards_rollout[worker_id],intrinsic_returns_rollout[worker_id],centralised_rff_int[worker_id] = \
                            intrinsic_reward_calculation(active_runners=active_parallel_envs[worker_id],\
                                                        curiosity_module=centralised_curiosity_module,\
                                                        obs_rms=centralised_obs_rms,\
                                                        rff_int_runners=centralised_rff_int[worker_id],\
                                                        obs_cur=obs_cur,\
                                                        acts=act,
                                                        batchnormalization = True)
                        # ***2. NORMALIZATION
                        if False:
                            # 2.1.update obs normalization
                            obs_for_curiosity = np.array(current_obs[worker_id]) if multi_head_action_curiosity else np.array(next_obs[worker_id]) #[num_worker,rollout*num_runners,42,42]
                            actions_for_curiosity = np.array(actions[worker_id])
                            centralised_obs_rms.update_from_moments(np.mean(obs_for_curiosity,axis=0),
                                                                    np.var(obs_for_curiosity,axis=0),
                                                                    np.array(obs_for_curiosity).shape[0])

                        # 2.2.stats calculation --> intrinsic returns calculated per runner
                        for returns_runner in intrinsic_returns_rollout[worker_id]:
                            if len(returns_runner) > 0: #for security
                                batch_mean = np.mean(returns_runner)
                                batch_var = np.var(returns_runner)
                                batch_count = len(returns_runner)
                                centralised_reward_rms.update_from_moments(batch_mean=batch_mean,batch_var=batch_var,batch_count=batch_count)
                        std_normalization = np.sqrt(centralised_reward_rms.var)
                        standard_deviation_intrinsic[worker_id].append(std_normalization) #monitorization

                        # 2.3. Normalization of rewards
                        normalized_int_rewards = [] # rews per runner
                        for int_rews_runner in intrinsic_rewards_rollout[worker_id]:
                            normalized_int_rewards.append(np.asarray(int_rews_runner)/std_normalization)
                        intrinsic_rewards_norm_rollout[worker_id] = normalized_int_rewards # save them to send them next to worker

                        # monitorization -- at episode level
                        intrinsic_rewards[worker_id].extend(np.asarray([rollout_at_runner for runner_level in intrinsic_rewards_rollout[worker_id] for rollout_at_runner in runner_level]))
                        intrinsic_rewards_normalized[worker_id].extend(np.asarray([rollout_at_runner for runner_level in normalized_int_rewards for rollout_at_runner in runner_level]))

                # *** 3.TRAIN CURIOSITY MODULE
                # get a uniform vector with all the experiences collected by all workers and runners (we do one optimization step with a bigger batch size)
                if False:
                    obs_for_train = np.asarray([rollout_at_runner for runner_level in current_obs for rollout_at_runner in runner_level]) #1 runner: from [1,50,42,42] to [1*50,42,42] and the same at worker level
                    actions_for_train = np.asarray([rollout_at_runner for runner_level in actions for rollout_at_runner in runner_level])
                    rnd_loss = centralised_curiosity_module.train(
                                ((obs_for_train-centralised_obs_rms.mean)/np.sqrt(centralised_obs_rms.var)).clip(-5, 5),
                                actions_for_train)
                else:
                    obs_for_train = []
                    for worker_id in worker_ids:
                        if active_worker[worker_id]:
                            for obs in observations_rollout_runner[worker_id]:
                                batch_mean = np.mean(obs,axis=0)
                                batch_var = np.var(obs,axis=0)
                                if batch_var.all() == 0:
                                    print('MAIN batch_var zero')
                                    print(batch_var.shape)
                                    batch_var = 1e-4
                                obs_for_train.extend( ((np.array(obs) - batch_mean)/np.sqrt(batch_var)).clip(-1, 1) )

                    obs_for_train = np.asarray(obs_for_train)
                    actions_for_train = np.asarray([rollout_at_runner for runner_level in actions for rollout_at_runner in runner_level])
                    rnd_loss = centralised_curiosity_module.train(obs_for_train,actions_for_train)

            # COUNTS INDEPENDENT
            elif curiosity_type == 'count_i':
                for worker_id in worker_ids:
                    if active_worker[worker_id]:
                        # 1. Get rewards
                        coords = coordinates_rollout_runner[worker_id]
                        act = actions_rollout_runner[worker_id]
                        intrinsic_rewards_rollout[worker_id],intrinsic_returns_rollout[worker_id],rff_int[worker_id] = \
                            intrinsic_reward_calculation_countvisits(worker_id = worker_id,
                                                                     active_runners=active_parallel_envs[worker_id],
                                                                     visitcount_module=visitcount_module[worker_id],
                                                                     coords=coords,
                                                                     acts=act,
                                                                     rff_int_runners = rff_int[worker_id])

                        # ***2. NORMALIZATION
                        # 2.1.stats calculation --> intrinsic returns calculated per runner
                        for returns_runner in intrinsic_returns_rollout[worker_id]:
                            if len(returns_runner) > 0: #for security
                                batch_mean = np.mean(returns_runner)
                                batch_var = np.var(returns_runner)
                                batch_count = len(returns_runner)
                                reward_rms[worker_id].update_from_moments(batch_mean=batch_mean,batch_var=batch_var,batch_count=batch_count)
                        std_normalization = np.sqrt(reward_rms[worker_id].var)
                        standard_deviation_intrinsic[worker_id].append(std_normalization) #monitorization

                        # 2.2. Normalization of rewards
                        normalized_int_rewards = [] # rews per runner
                        for int_rews_runner in intrinsic_rewards_rollout[worker_id]:
                            normalized_int_rewards.append(np.asarray(int_rews_runner)/std_normalization)
                        intrinsic_rewards_norm_rollout[worker_id] = normalized_int_rewards # save them to send them next to worker

                        # monitorization -- at episode level
                        intrinsic_rewards[worker_id].extend(np.asarray([rollout_at_runner for runner_level in intrinsic_rewards_rollout[worker_id] for rollout_at_runner in runner_level]))
                        intrinsic_rewards_normalized[worker_id].extend(np.asarray([rollout_at_runner for runner_level in normalized_int_rewards for rollout_at_runner in runner_level]))

                        # 2. Update bins
                        coords_for_train = np.asarray([rollout_at_runner for runner_level in coordinates_rollout_runner[worker_id] for rollout_at_runner in runner_level])
                        actions_for_train = np.asarray([rollout_at_runner for runner_level in actions_rollout_runner[worker_id] for rollout_at_runner in runner_level])
                        visitcount_module[worker_id].train(coords_for_train,actions_for_train)

            # COUNTS CENTRALIZED
            elif curiosity_type == 'count_c':
                for worker_id in worker_ids:
                    if active_worker[worker_id]:
                        # 1. Get rewards
                        coords = coordinates_rollout_runner[worker_id]
                        act = actions_rollout_runner[worker_id]
                        intrinsic_rewards_rollout[worker_id],intrinsic_returns_rollout[worker_id],rff_int[worker_id] = \
                            intrinsic_reward_calculation_countvisits(worker_id = worker_id,
                                                                     active_runners=active_parallel_envs[worker_id],
                                                                     visitcount_module=visitcount_module,
                                                                     coords=coords,
                                                                     acts=act,
                                                                     rff_int_runners = rff_int[worker_id])

                        # ***2. NORMALIZATION
                        # 2.1.stats calculation --> intrinsic returns calculated per runner
                        for returns_runner in intrinsic_returns_rollout[worker_id]:
                            if len(returns_runner) > 0: #for security
                                batch_mean = np.mean(returns_runner)
                                batch_var = np.var(returns_runner)
                                batch_count = len(returns_runner)
                                reward_rms.update_from_moments(batch_mean=batch_mean,batch_var=batch_var,batch_count=batch_count)
                        std_normalization = np.sqrt(reward_rms.var)
                        standard_deviation_intrinsic[worker_id].append(std_normalization) #monitorization

                        # 2.2. Normalization of rewards
                        normalized_int_rewards = [] # rews per runner
                        for int_rews_runner in intrinsic_rewards_rollout[worker_id]:
                            normalized_int_rewards.append(np.asarray(int_rews_runner)/std_normalization)
                        intrinsic_rewards_norm_rollout[worker_id] = normalized_int_rewards # save them to send them next to worker

                        # monitorization -- at episode level
                        intrinsic_rewards[worker_id].extend(np.asarray([rollout_at_runner for runner_level in intrinsic_rewards_rollout[worker_id] for rollout_at_runner in runner_level]))
                        intrinsic_rewards_normalized[worker_id].extend(np.asarray([rollout_at_runner for runner_level in normalized_int_rewards for rollout_at_runner in runner_level]))

                        # 2. Update bins
                        j = 0
                        for rid,active in enumerate(active_parallel_envs[worker_id]):
                            if active:
                                visitcount_module.train(coordinates=coordinates_rollout_runner[worker_id][j],actions=actions_rollout_runner[worker_id][j],worker_id=worker_id,runner_id=rid)
                                j += 1

            # finished curiosity related calculation and updates

            # =============================================================================
            # ACTOR-CRITIC TRAIN PHASE
            # =============================================================================
            for worker_id,parent_conn in zip(worker_ids,parent_connections):
                if active_worker[worker_id]:

                    # send normalized int rewards for correct update
                    args = intrinsic_rewards_norm_rollout[worker_id]
                    parent_conn.send(('train', args))

                    # Wait until worker has trained its network
                    cmd, args = parent_conn.recv()
                    if cmd == 'training_completed':
                        # update monitorization params
                        adv_total,\
                        adv_ext,disc_ret_ext,vext,\
                        adv_int,disc_ret_int,vint,\
                        policy_loss,vext_loss,vint_loss,ent,ppo_loss,\
                        vext_preW, vext_postW, prob_move_up, prob_move_right, prob_move_left,\
                        ent_step, probs,\
                        clipped_info,gr_actor,gr_critic,\
                        inp_hidden, idx_runner = args

                        # extrinsic
                        advantage_total[worker_id].extend(adv_total)
                        advantage_ext[worker_id].extend(adv_ext)
                        disc_return_ext[worker_id].extend(disc_ret_ext)
                        value_ext[worker_id].extend(vext)
                        # intrinsic
                        advantage_int[worker_id].extend(adv_int)
                        disc_return_int[worker_id].extend(disc_ret_int)
                        value_int[worker_id].extend(vint)

                        actor_loss[worker_id].extend(policy_loss)
                        entropy[worker_id].extend(ent)
                        total_loss[worker_id].extend(ppo_loss)

                        clip_info[worker_id].extend(clipped_info)
                        policy_probs[worker_id].extend(probs)
                        probability_move_up[worker_id].extend(prob_move_up)
                        probability_move_right[worker_id].extend(prob_move_right)
                        probability_move_left[worker_id].extend(prob_move_left)
                        entropy_step[worker_id].extend(ent_step)
                        grad_actor[worker_id].append(gr_actor)

                        if use_centralized_critic:
                            # shared critic returns
                            rext[worker_id].extend(np.asarray(disc_ret_ext))
                            rint[worker_id].extend(np.asarray(disc_ret_int))
                            # get hidden states and idx of runners
                            hidden_states[worker_id] = inp_hidden
                            index_runner[worker_id] = idx_runner
                        else:
                            grad_critic[worker_id].append(gr_critic)
                            critic_ext_loss[worker_id].extend(vext_loss)
                            critic_int_loss[worker_id].extend(vint_loss)
                            vext_pre_train[worker_id].extend(vext_preW)
                            vext_post_train[worker_id].extend(vext_postW)

            # UPDATE CENTRALIZED CRITIC related
            for worker_id in worker_ids:
                if use_centralized_critic and active_worker[worker_id]:
                    # train
                    centralized_critic.train_model_batch(stacked_states=np.asarray(stacked_states[worker_id]),
                                                         actions=np.asarray(actions[worker_id]),
                                                         disc_return_ext=np.asarray(rext[worker_id]),
                                                         disc_return_int=np.asarray(rint[worker_id]),
                                                         worker_id = worker_id,
                                                         idx_runner=index_runner[worker_id],
                                                         h_in=hidden_states[worker_id])
                    # get monitorization metrics
                    vext_loss, vint_loss, vext_preW, vext_postW, gr_critic = centralized_critic.get_train_related_metrics()
                    grad_critic[worker_id].append(gr_critic)
                    critic_ext_loss[worker_id].extend(vext_loss)
                    critic_int_loss[worker_id].extend(vint_loss)
                    vext_pre_train[worker_id].extend(vext_preW)
                    vext_post_train[worker_id].extend(vext_postW)

            # once critic updated, update hidden state of both workers
            if use_lstm and use_centralized_critic:
                for worker_id in worker_ids:
                    if active_worker[worker_id]:
                        for j,(i,f) in enumerate(index_runner[worker_id]):
                            if i!=f:
                                input_stacked_states = torch.from_numpy(np.asarray(stacked_states[worker_id][i:f])).float()
                                input_hidden = hidden_states[worker_id][j]
                                input_actions = torch.from_numpy(np.asarray(actions[worker_id][i:f]))
                                input_roles = torch.from_numpy(np.full(len(actions[worker_id][i:f]),worker_id))

                                hidden_states[worker_id][j] = centralized_critic.calculate_hidden_state(
                                                                                stacked_states=input_stacked_states,
                                                                                hidden=input_hidden,
                                                                                actions=input_actions,
                                                                                roles=input_roles)

                                [print('\n***at main W{}*** \nhidden_in centralized {} \nnew hidden hidden:{}\n'.format(worker_id,input_hidden,hidden_states[worker_id])) if verbose_test else 0]

                # send updated critic weights to workers
                for worker_id,parent_conn in zip(worker_ids,parent_connections):
                    shared_critic_params = centralized_critic.critic.to('cpu').state_dict()
                    centralized_critic.critic.to(device)
                    if active_worker[worker_id]:
                        hidden_state_worker = hidden_states[worker_id]
                        args = shared_critic_params,hidden_state_worker
                        parent_conn.send(('update_centralized_critic', args))
            # (if episode has finished)
            for worker_id in worker_ids:
                    if active_worker[worker_id] and episode_finished[worker_id]:
                        elapsed_time = time() - init_ep_time[worker_id]
                        time_worker[worker_id].append(elapsed_time)
                        active_worker[worker_id] = not episode_finished[worker_id]

            # save every x-steps
            for worker_id,parent_conn in zip(worker_ids,parent_connections):
                # Close worker connection
                if episodes_counter[worker_id] >= n_episodes:
                    parent_conn.send(('close_connection',0))
                    cmd,args = parent_conn.recv()
                    print('close ack signal:',cmd)
                    if cmd == 'connection_closed':
                        closed_connections[worker_id] = True
                        print('Worker {} ended'.format(worker_id))
                        print('Closed-Connections status: ',closed_connections)
                        # update worker_id & parent connection arrays
                        index = parent_connections.index(parent_conn)
                        del parent_connections[index]
                        index = worker_ids.index(worker_id)
                        del worker_ids[index]

    # Results summary
    print('Training finished - Total number of episodes: {}'.format(episodes_counter))
    print("\nTotal elapsed time: %.2f minutes" % ((time() - time_start) / 60.0))
    for wID,worker_scores in enumerate(scores):
        train_scores = np.array(worker_scores)
        print("Worker %d, Results --> " % wID, \
            " mean: %.4f +/- %.4f," % (train_scores.mean(), train_scores.std()), \
            "min: %.1f," % train_scores.min(),\
            "max: %.1f," % train_scores.max()
            )
