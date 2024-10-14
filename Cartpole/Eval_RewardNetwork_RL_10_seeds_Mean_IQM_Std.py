__author__ = "akansha_kalra"
import os
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor,VecVideoRecorder
from copy import deepcopy
from gymnasium.envs.registration import register

register(
    id="FH-CartPole",
    entry_point="Fixed_Horizon_CP_Env:FH_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_CP_Env:FH_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)

env_id = "FH-CartPole"
env = make_vec_env(env_id, n_envs=1, seed=0)


def eval_policy(env,GT_RL=False,DDT_Soft='0',NN=False,num_episodes=100,write=False):

    ls = ['Seed0','Seed1','Seed2','Seed3','Seed4','Seed5','Seed6','Seed7','Seed8','Seed9']
    print(f"Using seeds for eval{ls}")
    episodic_rewards=[]
    base_dir='RL_using_Trained_Reward_Models/'
    eval_dir=base_dir+'eval/'
    if not os.path.exists(eval_dir):
        print('Creating  directory to save Eval PPO policy rollouts : ' + eval_dir)
        os.makedirs(eval_dir)

    if (DDT_Soft=='0' or DDT_Soft=='1') and NN is False and GT_RL is False:
        model_dir =base_dir + '/DDT/saved_models/'
    elif NN==True and GT_RL is False and DDT_Soft==None:
        model_dir=base_dir + '/NN/saved_models/'


    for s in ls:
        #RL using learned NN reward
        if NN and GT_RL is False and DDT_Soft==None:
            model_path = model_dir + 'RL_NN_' + s + '_ts_500k_batchsize1024_numenvs5.zip'
        # RL using GT reward
        if GT_RL and NN is False and DDT_Soft==None:
            model_path = 'GT_RL_FH/For_table_matched_params_to_DDT+NN/'+ 'GT_RL_' + s + '_ts_500000.0_batch_size1024_numenvs5'
        # RL using learned DDT reward - using soft reward
        if DDT_Soft=='0' and NN is False:
            model_path = model_dir + 'RL_DDT_Soft_' + s + '_ts_500k_batch_size1024_numenvs5.zip'
        # RL using learned DDT reward - using argmax reward
        elif DDT_Soft=='1' and NN is False:
            model_path = model_dir + 'RL_DDT_Argmax_' + s + '_ts_500k_batch_size1024_numenvs5.zip'

        if os.path.exists(model_path):
            print(model_path)
            curr_model=PPO.load(model_path)
            curr_rewards,_=evaluate_policy(curr_model,env, n_eval_episodes=num_episodes, deterministic=True,render=True, return_episode_rewards=True, warn=True)
            episodic_rewards.extend(reward for reward in curr_rewards)

    mean=np.mean(episodic_rewards)
    std=np.std(np.array(episodic_rewards))
    if write==True:
        if DDT_Soft=='0' and NN is False:
            policy_eval_file = open(eval_dir + 'DDT_Soft' +  str(num_episodes) + '.txt', 'w')
        elif DDT_Soft=='1' and NN is False:
            policy_eval_file = open(eval_dir + 'DDT_Argmax'+ str(num_episodes)+ '.txt','w')
        if NN==True and GT_RL is False and DDT_Soft==None:
            policy_eval_file = open(eval_dir + 'RL_using_NN'+ str(num_episodes)+ '.txt','w')
        if GT_RL and NN is False and DDT_Soft==None:
            policy_eval_file = open(eval_dir + 'GT_RL_' + str(num_episodes) + '.txt', 'w')


        policy_eval_file.write("Mean " + "Std " + "Episodic rewards")
        policy_eval_file.write('\n')
        policy_eval_file.write(f'{mean}, {std}, {episodic_rewards}')
        policy_eval_file.write('\n')
    print(f"Mean {mean}, Std {std} for ddt{DDT_Soft} ,RL_using_NN {NN}, GT RL {GT_RL} for {num_episodes} episodes with rewards{episodic_rewards}")
    return mean,std,episodic_rewards

def IQM(episodic_rewards_ls):
    reward_array = np.array(episodic_rewards_ls)
    sorted_reward_ls=sorted(episodic_rewards_ls, reverse=False)

    copy_sorted_rewards_ls=deepcopy(sorted_reward_ls)
    print(sorted_reward_ls)
    len_rewards=len(sorted_reward_ls)
    whole_nos = len_rewards // 4
    if len_rewards%4==0:
        for _ in range(whole_nos):
            del copy_sorted_rewards_ls[0]
        print(copy_sorted_rewards_ls)

        for _ in range(whole_nos):
            del copy_sorted_rewards_ls[-1]
        print(copy_sorted_rewards_ls)

        iqm=np.mean(copy_sorted_rewards_ls)

    elif len_rewards%4!=0:
        float_quotient=len_rewards/4

        partial_nos=float_quotient-whole_nos
        for _ in range(whole_nos):
            del copy_sorted_rewards_ls[0]
        print(copy_sorted_rewards_ls)

        for _ in range(whole_nos):
            del copy_sorted_rewards_ls[-1]
        print(copy_sorted_rewards_ls)

        partial_weight=1-partial_nos

        partial_obs_1=copy_sorted_rewards_ls[0]
        partial_obs_2=copy_sorted_rewards_ls[-1]

        del copy_sorted_rewards_ls[0]
        del copy_sorted_rewards_ls[-1]

        final_weight=len(copy_sorted_rewards_ls)+2*partial_weight
        print(final_weight)

        iqm=(np.sum(copy_sorted_rewards_ls)+partial_weight*(partial_obs_1+partial_obs_2))/final_weight

    print(f"IQM is {iqm}")
    return iqm

if __name__=="__main__":
    mean,std,episodic_rewards=eval_policy(env,GT_RL=True,DDT_Soft=None,NN=False,num_episodes=100,write=True)
    # mean,std,episodic_rewards=eval_policy(env,DDT_Soft='1',write=True)
    '''nn'''
    # mean, std, episodic_rewards = eval_policy(env, DDT_Soft=None,RL_using_NN=True, write=True)
    iqm=IQM(episodic_rewards)