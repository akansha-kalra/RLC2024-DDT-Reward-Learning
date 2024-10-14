__author__ = "akansha_kalra"
import numpy as np
from copy import deepcopy
# import scipy.stats as stats
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--RL_Penalty_Arg_Type', default="NPNA", help="RL type- has to be one of NPNA, NPA, PNA, PA")
parser.add_argument('--TREX', default=False, type=bool, help="Whether to use RL policy trained with DDT or TREX. If false, then DDT")

args = parser.parse_args()
env_name=args.env_name
Penalty_Arg_Type=args.RL_Penalty_Arg_Type
TREX=args.TREX

def get_episodic_rewards(env_name, Penalty_Arg_Type = None, TREX = False):
    if env_name == "breakout":
        env_i = "Breakout"
    else:
        env_i = "BeamRider"
    eval_dir = 'Final_A2C_DDT_RL/Eval_RL/10_Seeds/'
    if Penalty_Arg_Type != None:

        policy_eval_file_path = eval_dir + env_i + Penalty_Arg_Type + '100_10Seeds.txt'
    if TREX == True:
        policy_eval_file_path = eval_dir + env_i + 'TREX100_10Seeds.txt'
    print(f"Using file {policy_eval_file_path} to get rewards array")
    with open(policy_eval_file_path, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if i != 0:
            mean, std, episodic_rewards = line.strip('\n').split(',')[0], line.strip('\n').split(',')[1], line.strip('\n').split( ',')[2:]

    episodic_rewards_0 = episodic_rewards[0].replace('[', '')
    episodic_rewards_499 = episodic_rewards[499].replace(']', '')
    del episodic_rewards[0]
    del episodic_rewards[-1]
    print(len(episodic_rewards))
    episodic_rewards.insert(0, episodic_rewards_0)
    episodic_rewards.insert(499, episodic_rewards_499)
    print(len(episodic_rewards))

    episodic_rewards_ls = [float(x) for x in episodic_rewards]
    return episodic_rewards_ls




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









if __name__ == "__main__":
    '''DDT+TREX'''
    episodic_rewards_ls=get_episodic_rewards(env_name,Penalty_Arg_Type,TREX)
    '''uncomment for TREX'''
    # episodic_rewards_ls = get_episodic_rewards(env_name, Penalty_Arg_Type=None,TREX=True)
    iqm=IQM(episodic_rewards_ls)
