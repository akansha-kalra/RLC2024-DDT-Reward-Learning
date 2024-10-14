__author__ = "akansha_kalra"
import os
import warnings
warnings.filterwarnings('ignore')
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack,VecEnvWrapper
import numpy as np
import argparse
parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--RL_Penalty_Arg_Type', default="NPNA", help="RL type- has to be one of NPNA, NPA, PNA, PA")
args = parser.parse_args()
env_name=args.env_name
Penalty_Arg_Type=args.RL_Penalty_Arg_Type


def create_env(env_name,num_envs=1):
    if env_name == "beamrider":
        env_id = "BeamRider" + "NoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(f"Env is {env_id} and we have {num_envs} envs")
    assert isinstance(env_id, str)

    env_config = {"env_name": env_id, "num_envs": num_envs, "env_seed": 0}
    env = make_atari_env(env_config["env_name"], n_envs=env_config["num_envs"], seed=env_config["env_seed"])
    print(env)

    env = VecFrameStack(env, n_stack=4)

    return env


def eval_policy(env_name,env,Penalty_Arg_Type,num_episodes=100,write=False,ten_seeds=True):
    if ten_seeds==True:
        ls = ['seed0', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5', 'seed6','seed7', 'seed8', 'seed9']
    else:
        ls = ['seed0', 'seed1', 'seed2', 'seed3', 'seed4']

    print(f"Using seeds for eval{ls}")

    episodic_rewards=[]
    eval_dir= 'Final_A2C_DDT_RL/Eval_RL/10_Seeds/'
    if not os.path.exists(eval_dir):
        print(' Creating  directory to save Eval A2C policy rollouts : ' + eval_dir)
        os.makedirs(eval_dir)

    if env_name == "breakout":
        env_i="Breakout"
        model_dir = "Final_A2C_DDT_RL/Breakout/"
    else:
        env_i="BeamRider"
        model_dir = "Final_A2C_DDT_RL/BeamRider/"
    if Penalty_Arg_Type == "NPNA" or Penalty_Arg_Type == "NPA" or Penalty_Arg_Type == "PNA" or Penalty_Arg_Type == "PA":

        for s in ls:
            model_path = model_dir + Penalty_Arg_Type + "/" + env_i + "_" + Penalty_Arg_Type + "_" + s + ".zip"
            if os.path.exists(model_path):
                print(model_path)
                curr_model=A2C.load(model_path)
                curr_rewards,_=evaluate_policy(curr_model,env, n_eval_episodes=num_episodes, deterministic=True,render=True, return_episode_rewards=True, warn=True)
                episodic_rewards.extend(reward for reward in curr_rewards)

    mean=np.mean(episodic_rewards)
    std=np.std(np.array(episodic_rewards))
    if write==True:
        if ten_seeds==True:
            policy_eval_file = open(eval_dir + env_i + Penalty_Arg_Type + str(num_episodes) + '_10Seeds.txt', 'w')
        else:
            policy_eval_file = open(eval_dir + env_i + Penalty_Arg_Type + str(num_episodes) + '.txt', 'w')
        policy_eval_file.write("Mean " + "Std " + "Episodic rewards")
        policy_eval_file.write('\n')
        policy_eval_file.write(f'{mean}, {std}, {episodic_rewards}')
        policy_eval_file.write('\n')
    print(f"Mean {mean}, Std {std} for {env_name} under {Penalty_Arg_Type} for {num_episodes} episodes with rewards{episodic_rewards}")
    return mean,std,episodic_rewards

if __name__ == "__main__":

    env=create_env(env_name)
    print(env)
    eval_policy(env_name,env,Penalty_Arg_Type,num_episodes=100,write=True,ten_seeds=True)