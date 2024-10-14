__author__ = "akansha_kalra"
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

args = parser.parse_args()
env_name=args.env_name

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


def eval_TREX_policy(env_name,env,num_episodes=100,ten_seeds=True):
    if ten_seeds==True:
        ls = ['seed0', 'seed1', 'seed2', 'seed3', 'seed4', 'seed5', 'seed6','seed7', 'seed8', 'seed9']
    else:
        ls=['seed0','seed1','seed2','seed3','seed4']
    print(ls)
    episodic_rewards=[]
    # eval_dir='/home/ak/Documents/AtariRL_SD/A2C_DDT_RL/Eval_RL/'
    if env_name == "breakout":
        env_i="Breakout"
        model_dir = "Final_A2C_DDT_RL/TREX/Breakout/"
    else:
        env_i="BeamRider"
        model_dir = "Final_A2C_DDT_RL/TREX/BeamRider/"

    eval_dir = 'Final_A2C_DDT_RL/Eval_RL/10_Seeds/'
    if ten_seeds==True:
        policy_eval_file = open(eval_dir + env_i + 'TREX' + str(num_episodes) + '_10Seeds.txt', 'w')
    else:
        policy_eval_file = open(eval_dir+env_i+'TREX'+str(num_episodes)+'.txt', 'w')
    policy_eval_file.write( "Mean " + "Std " + "Episodic rewards")
    policy_eval_file.write('\n')
    for s in ls:
        model_path=model_dir +env_i+"_"+"TREX_"+s+".zip"
        if os.path.exists(model_path):
            print(model_path)
            curr_model=A2C.load(model_path)
            curr_rewards,_=evaluate_policy(curr_model,env, n_eval_episodes=num_episodes, deterministic=True,render=True, return_episode_rewards=True, warn=True)
            episodic_rewards.extend(reward for reward in curr_rewards)

    mean=np.mean(episodic_rewards)
    std=np.std(np.array(episodic_rewards))
    policy_eval_file.write(f'{mean}, {std}, {episodic_rewards}')
    policy_eval_file.write('\n')
    print(f"Mean {mean}, Std {std} for {env_name} TREX NN for {num_episodes} episodes with rewards{episodic_rewards}")
    return mean,std,episodic_rewards

if __name__ == "__main__":
    # env_name="breakout"
    # env_name="beamrider"
    env=create_env(env_name)
    print(env)
    eval_TREX_policy(env_name,env,num_episodes=100,ten_seeds=True)

