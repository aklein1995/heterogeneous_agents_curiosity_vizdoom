
import numpy as np

import torch
from torch._six import inf
import pickle
import os
from collections import deque
import shutil

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = batch_count + self.count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (tot_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

def dumpModels(parent_conn,episode_counter,worker_id,folder_path_results,save_every_x=50,rnd_weights=None):
    """
        Dump Actor and Crtic Models every X episodes
        RND params every 1000 episodes
    """
    doc_tail = dumpStringTail(episode_counter)

    # Obtain trained model weights
    parent_conn.send(('get_model',0))
    cmd,args = parent_conn.recv()
    actor_params, critic_params, room_freq, room_steps, trajectories= args
    # get rnd params from either main(centralized) or worker(individual-worker)
    rnd_params = rnd_weights

    # determine every X episodes to save a new model
    if episode_counter % save_every_x == 0:
        doc_tail_ac = '_' + str(episode_counter)
    else:
        doc_tail_ac = doc_tail

    if cmd ==  'model_sent':
        #with doc_tail_ac --> more freq
        torch.save(actor_params,folder_path_results + '/agent' + str(worker_id) + '_actor' + doc_tail_ac + '.pth')
        torch.save(critic_params,folder_path_results + '/agent' + str(worker_id) + '_critc' + doc_tail_ac + '.pth')
        torch.save(rnd_params,folder_path_results + '/agent' + str(worker_id) + '_rnd_curiosity' + doc_tail_ac + '.pth')

def dumpStringTail(episode):
    if episode < 1000:
        return '_1000'
    elif episode < 2000:
        return '_2000'
    elif episode < 3000:
        return '_3000'
    elif episode < 4000:
        return '_4000'
    elif episode < 5000:
        return '_5000'
    elif episode < 6000:
        return '_6000'
    elif episode < 7000:
        return '_7000'
    elif episode < 8000:
        return '_8000'
    elif episode < 9000:
        return '_9000'
    else:
        return '_10000'

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def generateTestFolder():
    folders = os.listdir('runs')
    max_v = 0
    for f in folders:
        count = int(f.split('_')[1])
        if count > max_v:
            max_v = count
    new_v = max_v + 1
    result_folder_path = 'runs/experiment_' + str(new_v)
    models_folder_path = 'runs/experiment_' + str(new_v) + '/models'
    os.mkdir(result_folder_path)
    os.mkdir(models_folder_path)
    print('Generated monitorization folder: ',result_folder_path)
    print('Generated models folder: ',models_folder_path)
    # for code
    paths = ['/code','/code/src','/code/util_functions']
    for path in paths:
        os.mkdir(result_folder_path + path)
    print('Generated code related folders: ',models_folder_path)
    #copy files into created folders
    copytree(src='src',
            dst=result_folder_path + '/code/src')
    copytree(src='util_functions',
            dst=result_folder_path + '/code/util_functions')
    shutil.copyfile('config.conf',result_folder_path+'/code/config.conf')
    shutil.copyfile('main.py',result_folder_path+'/code/main.py')
    # copytree(src='main.py',dst=code_folder_path)
    return result_folder_path,models_folder_path

def DumpToPickle(path,value,type='list'):
    # check if file exists
    if os.path.exists(path):
        with open(path,'rb+') as f:
            a = pickle.load(f)
    else:
        a = []
    # append or extend based on attribute value
    if type == 'int':
        for _ in range(len(value)):
            a.append(value)
        with open(path,'wb+') as f:
            pickle.dump(a,f)
    else:
        a.extend(value)
        with open(path,'wb+') as f:
            pickle.dump(a,f)

def calculate_returns(rewards,gamma=.999):
    rewards_discounted = []

    idx = range(0,len(rewards))
    Gt = sum([(gamma**ii)*rewards[ii] for ii in idx])

    rewards_discounted.append(Gt) # for the first visit, R0
    for i in range(1,len(rewards)):
        Gt = (Gt - rewards[i-1])/gamma
        rewards_discounted.append(Gt)
    return np.asarray(rewards_discounted)

def get_state_value(critic,stacked_frames=[],next_state=[],device='cpu'):
    next_states = deque(stacked_frames,maxlen=4)
    # Calculate V(st+1) for the last sample
    additional = 1
    if len(next_states) < 4:
        additional = 4 - len(next_states)
    for i in range(additional):
        print(i)
        next_states.append(next_state)
    # transform to numpy and feed network
    next_states = np.asarray(next_states)
    vext, vint = critic(torch.from_numpy(next_states).to(device).float().unsqueeze(0))

    vext = vext.data.cpu().squeeze().numpy()
    vint = vint.data.cpu().squeeze().numpy()
    return vext, vint

def get_advantages_dual(rewards_ext,rewards_int,
                    dones,value_ext,value_int,
                    next_vext, next_vint,
                    gamma = .99,gamma_int = .99,
                    lmbda = .95,use_gae = 1,
                    ext_coef=1., int_coef=0,
                    device='cpu'):
    """
        A(s,a) = Q(s,a) - V(s) -- > advantage = actual return - network value estimate
        GAE calculation: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/
    """
    masks = 1 - np.asarray(dones)
    masks_int = np.ones(len(dones)) # masks_int = 1 - np.asarray(dones)

    # Calculate for extrinsic values
    discounted_return = []
    advantage_ext = []
    gae = 0
    running_add = min(next_vext,1)
    if use_gae:
        for i in reversed(range(len(rewards_ext))):
            if i == len(rewards_ext) - 1:
                delta = rewards_ext[i] + gamma * next_vext * masks[i] - value_ext[i]
            else:
                delta = rewards_ext[i] + gamma * value_ext[i + 1] * masks[i] - value_ext[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            discounted_return.insert(0, gae + value_ext[i])
    else:
        for i in reversed(range(len(rewards_ext))):
            running_add = rewards_ext[i] + gamma * running_add * masks[i]
            discounted_return.insert(0, running_add)

    disc_return_ext = np.asarray(discounted_return)
    advantage_ext = np.asarray(discounted_return) - np.asarray(value_ext)

    # Calculate for intrinsic values
    discounted_return_int = []
    advantage_int = []
    gae = 0
    running_add = next_vint
    if use_gae:
        for i in reversed(range(len(rewards_int))):
            if i == len(rewards_int) - 1:
                delta = rewards_int[i] + gamma_int * next_vint * masks_int[i] - value_int[i]
            else:
                delta = rewards_int[i] + gamma_int * value_int[i + 1] * masks_int[i] - value_int[i]
            gae = delta + gamma_int * lmbda * masks_int[i] * gae
            discounted_return_int.insert(0, gae + value_int[i])
    else:
        running_add = vint
        for i in reversed(range(len(rewards_int))):
            running_add = rewards_int[i] + gamma_int * running_add * masks[i]
            discounted_return_int.insert(0, running_add)

    disc_return_int = np.asarray(discounted_return_int)
    advantage_int = np.asarray(discounted_return_int) - np.asarray(value_int)

    # Calculate Total advantage
    advantages = (advantage_ext * ext_coef) + (advantage_int * int_coef)
    advantages /= (ext_coef + int_coef)

    return np.asarray(advantages),\
           np.asarray(advantage_ext),\
           np.asarray(disc_return_ext),\
           np.asarray(value_ext),\
           np.asarray(advantage_int),\
           np.asarray(disc_return_int),\
           np.asarray(value_int)

def get_advantages(rewards,dones,value,next_v,
                   gamma = .99,lmbda = .95,use_gae = 1,
                   type=1):
    """
        A(s,a) = Q(s,a) - V(s) -- > advantage = actual return - network value estimate
        GAE calculation: https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8
        https://danieltakeshi.github.io/2017/04/02/notes-on-the-generalized-advantage-estimation-paper/

        -type: determines if it is non-episodic or yes (also related to stationarity)
            -non-episodic = 0 (intrinsic values)
            -episodic = 1 (extrinsic values)
    """
    if type:
        masks = 1 - np.asarray(dones)
    else:
        masks = np.ones(len(dones))

    # Calculate
    discounted_return = []
    advantage = []
    gae = 0
    if use_gae:
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                delta = rewards[i] + gamma * next_v * masks[i] - value[i]
            else:
                delta = rewards[i] + gamma * value[i + 1] * masks[i] - value[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            discounted_return.insert(0, gae + value[i])
    else:
        running_add = min(next_v,1)
        for i in reversed(range(len(rewards))):
            running_add = rewards[i] + gamma * running_add * masks[i]
            discounted_return.insert(0, running_add)

    advantage = np.asarray(discounted_return) - np.asarray(value)

    return np.asarray(advantage),\
           np.asarray(discounted_return),\
           np.asarray(value)

def intrinsic_reward_calculation_deprecated(rnd_module,obs_rms,rff_int,next_states,actions):
    """
        -calculates intrinsic rewards from rnd_module
        -calculates the intrisic returns in a non-episodic and continuing way with a RewardForwardFilter
        (used for posterior normalization of intrinsic rewards)
    """
    # calculate reward intrinsic
    nobs = ((np.array(next_states)-obs_rms.mean)/np.sqrt(obs_rms.var)).clip(-5, 5)
    acts = np.array(actions)
    int_rews = rnd_module.compute_intrinsic_reward(nobs,acts)

    # calculate returns intrinsecos
    # int_rets = np.asarray([self.rff_int[runnerid].update(r) for r in int_rews[::-1]])
    int_rets = np.asarray([rff_int.update(r) for r in int_rews[::-1]])
    return int_rews, int_rets[::-1]



def intrinsic_reward_calculation(active_runners,curiosity_module,obs_rms,rff_int_runners,obs_cur,acts,returns_mask=True, batchnormalization=False):
    int_rewards_buffer = []
    int_returns_buffer = []
    j = 0 # used to count the id of buffers; INDICATES THE CURRENT RUNNER's ID
    for i,mask_runner in enumerate(active_runners):
        # only if runner was active (get that runners int rews)
        if mask_runner:

            # 1.calculate reward intrinsic
            if batchnormalization:
                # 1.1 batchnormalization
                batch_mean = np.mean(obs_cur[j],axis=0)
                batch_var = np.var(obs_cur[j],axis=0)
                if batch_var.all() == 0:
                    print('batch_var zero runner {}'.format(j))
                    print('batch_shape:',batch_var.shape)
                    print('obs shape:',np.array(obs_cur[j]).shape)
                    batch_var = 1e-4
                obs_cur_norm_clip = ((np.array(obs_cur[j]) - batch_mean)/np.sqrt(batch_var)).clip(-1, 1)
            else:
                # 1.1.2.  with obs_rms
                obs_cur_norm_clip = ((np.array(obs_cur[j])-obs_rms.mean)/np.sqrt(obs_rms.var)).clip(-5, 5)

            # 1.2. get actions if necessary
            a = np.array(acts[j])
            # 1.3. calculate int rewards
            int_rewards = curiosity_module.compute_intrinsic_reward(obs_cur_norm_clip,a)

            # 2.calculate returns
            if returns_mask:
                int_rets = np.asarray([rff_int_runners[i].update(r) for r in int_rewards[::-1]])
                int_returns = int_rets[::-1]
            else:
                int_returns = 0

            # 3. save values of that rollout for each runner
            int_rewards_buffer.append(int_rewards)
            int_returns_buffer.append(int_returns)
            j += 1

    return int_rewards_buffer,int_returns_buffer,rff_int_runners

def intrinsic_reward_calculation_countvisits(worker_id,active_runners,visitcount_module,coords,acts,rff_int_runners,returns_mask=True):
    int_rewards_buffer = []
    int_returns_buffer = []
    j = 0 # used to count the id of buffers; INDICATES THE CURRENT RUNNER's ID
    for i,mask_runner in enumerate(active_runners):
        # only if runner was active (get that runners int rews)
        if mask_runner:
            # 1.calculate reward intrinsic
            c = np.array(coords[j])
            a = np.array(acts[j])
            int_rewards = visitcount_module.compute_intrinsic_reward(worker_id,c,a)

            # 2.calculate returns
            if returns_mask:
                int_rets = np.asarray([rff_int_runners[i].update(r) for r in int_rewards[::-1]])
                int_returns = int_rets[::-1]
            else:
                int_returns = 0
            # 2. save values of that rollout for each runner
            int_rewards_buffer.append(int_rewards)
            int_returns_buffer.append(int_returns)
            j += 1

    return int_rewards_buffer,int_returns_buffer,rff_int_runners

def KL_divergence(p,q):
    """
        Calculates the KL-div for a batch of samples (i.e. 1-runner stream experiences)
        we know for the present work that:
         - p size = 5 (actions) - skilled-agent
         - q size = 4 (actions) - base-policy
    """
    results = []
    p = np.asarray(p)
    q = np.asarray(q)
    num_distributions = p.shape[0] #BatchSize
    size_distribution = p.shape[1] #ActionSpaceSize

    for i in range(num_distributions):
        score = 0
        for j in range(size_distribution-1): #make only for size of 4
            score += p[i,j] * np.log(p[i,j]/q[i,j])
        j += 1
        score += p[i,j] * np.log(p[i,j]/q[i,0]) #map to NOOP action
        results.append(score)
    return results
