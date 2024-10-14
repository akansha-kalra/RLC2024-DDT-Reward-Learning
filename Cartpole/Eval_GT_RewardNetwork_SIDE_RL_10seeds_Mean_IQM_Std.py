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
    id="FH-SS-CartPole",
    entry_point="Fixed_Horizon_Side_Start_CP_Env:FH_SideStart_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_Side_Start_CP_Env:FH_SideStart_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)

env_id = "FH-SS-CartPole"
start_side=2.4
env_kwargs = dict(side=start_side)
env = make_vec_env(env_id, n_envs=1, seed=0,env_kwargs=env_kwargs)


def eval_policy(env,GT_RL=False,DDT_Soft='0',NN=False,num_episodes=100,write=False):

    ls = ['seed0','seed1','seed2','seed3','seed4','seed5','seed6','seed7','seed8','seed9']
    print(f"Using seeds for eval{ls}")
    episodic_rewards=[]
    base_dir='Misaligned_side/'
    eval_dir=base_dir+'eval/'
    if not os.path.exists(eval_dir):
        print(' Creating  directory to save Eval PPO policy rollouts : ' + eval_dir)
        os.makedirs(eval_dir)
    # RL using learned DDT reward
    if (DDT_Soft=='0' or DDT_Soft=='1') and NN is False and GT_RL is False:
        model_dir =base_dir + '/RL_using_DDT/'
    # RL using learned NN reward
    elif NN is True and GT_RL is False:
        model_dir=base_dir + '/RL_using_NN/'
    # RL using GT reward
    elif NN is False and GT_RL is True:
        model_dir=base_dir + '/GT_RL/'


    for s in ls:
        if GT_RL is True and NN is False:
            assert DDT_Soft==None
            model_path=model_dir + 'GT_RL_Side_'+s + '_start_side2.4_ts_500000.0'
        if NN and GT_RL is False:
            assert DDT_Soft == None
            model_path = model_dir + 'NN_RL_Side_' + s + '_start_side2.4_ts_500000.0'
        if DDT_Soft=='0' and NN is False and GT_RL is False:
            model_path = model_dir + 'Soft_DDT_RL_Side_' + s + '_start_side2.4_ts_500000.0'
        elif DDT_Soft=='1' and NN is False and GT_RL is False:
            model_path = model_dir + 'Argmax_DDT_RL_Side_' + s + '_start_side2.4_ts_500000.0'

        if os.path.exists(model_path):
            print(model_path)
            curr_model=PPO.load(model_path)
            curr_rewards,_=evaluate_policy(curr_model,env, n_eval_episodes=num_episodes, deterministic=True,render=True, return_episode_rewards=True, warn=True)
            episodic_rewards.extend(reward for reward in curr_rewards)

    mean=np.mean(episodic_rewards)
    std=np.std(np.array(episodic_rewards))
    if write==True:
        if DDT_Soft=='0' and NN is False and GT_RL is False:
            policy_eval_file = open(eval_dir + 'DDT_Soft_SideStart' + str(start_side) +'_num_episodes'+  str(num_episodes) + '.txt', 'w')
        elif DDT_Soft=='1' and NN is False and GT_RL is False:
            policy_eval_file = open(eval_dir + 'DDT_Argmax_SideStart'+str(start_side) +'_num_episodes'+  str(num_episodes)+ '.txt','w')
        elif NN is True and GT_RL is False:
            policy_eval_file = open(eval_dir + 'RL_using_NN_SideStart'+str(start_side) +'_num_episodes'+ str(num_episodes)+ '.txt','w')
        elif NN is False and GT_RL is True:
            policy_eval_file = open(eval_dir + 'GT_RL_SideStart' +str(start_side) +'_num_episodes'+ str(num_episodes) + '.txt', 'w')

        policy_eval_file.write("Mean " + "Std " + "Episodic rewards")
        policy_eval_file.write('\n')
        policy_eval_file.write(f'{mean}, {std}, {episodic_rewards}')
        policy_eval_file.write('\n')
    print(f"Mean {mean}, Std {std} for ddt{DDT_Soft} ,RL_using_NN {NN} for {num_episodes} episodes with rewards{episodic_rewards}")
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
    '''GT RL'''
    # mean,std,episodic_rewards=eval_policy(env,GT_RL=True,DDT_Soft=None,NN=False,num_episodes=100,write=True)
    '''ddt-soft'''
    # mean,std,episodic_rewards=eval_policy(env,DDT_Soft='0',write=True)
    '''ddt-Argmax'''
    mean,std,episodic_rewards=eval_policy(env,DDT_Soft='1',NN=False,GT_RL=False,write=True)
    '''nn'''
    # mean, std, episodic_rewards = eval_policy(env, DDT_Soft=None,NN=True,GT_RL=False, write=True)
    iqm=IQM(episodic_rewards)