__author__ = "akansha_kalra"
import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecFrameStack,VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import numpy as np
import h5py
import random
import os

seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")

register(
    id="FH-CartPole",
    entry_point="Fixed_Horizon_CP_Env:FH_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_CP_Env:FH_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)

'''checking vector env'''
# vec_env = make_vec_env("FH-CartPole", n_envs=2)
# print(vec_env.observation_space)
def reset_data():
    data=dict(
        obs=[],
        reward=[],)
    return data

def create_dataset(env,num_episodes,timesteps_per_episode=200,save=False):
    assert timesteps_per_episode ==200
    demos = reset_data()
    traj = reset_data()

    for episode in range(num_episodes):
        print(f"On episode no {episode}")
        obs,_ = env.reset()
        reward_traj=0
        truncated = False
        while not truncated:
            next_obs, reward, terminated,truncated, info = env.step(env.action_space.sample())
            assert terminated==False
            traj['reward'].append(reward)
            traj['obs'].append(obs)
            reward_traj+=reward
            obs = next_obs
        if truncated:
            print(f"Cum reward of this episode {reward_traj}")
            for k in demos:
                demos[k].extend(traj[k])
            obs,_= env.reset()
            traj=reset_data()


    new_data = dict(
        obs=np.array(demos['obs']).astype(np.float64),
        reward=np.array(demos['reward']).astype(np.float64))
    print(new_data['reward'])
    # print(new_data['obs'])
    if save:
        current_directory = os.getcwd()
        output_file =  current_directory +'/Dataset_epsiodelen200_num_episodes_' + str(num_episodes) + '.zip'
        hfile = h5py.File(output_file, 'w')
        for k in new_data:
            hfile.create_dataset(k, data=new_data[k], compression='gzip')
    return new_data


def create_pref_dataset(dataset,num_prefs=1,traj_snippet_len=20,save=False):
    assert num_prefs >=1 and traj_snippet_len >=2
    reward_list=dataset['reward']
    torch_x_theta_obs_list=torch.index_select(torch.from_numpy(np.array(dataset['obs'])), dim=1, index=torch.tensor([0, 2]))
    pref_counter = 0
    num_demos = len(reward_list)
    pref_labels = []
    pref_indices = []
    pref_demos = []
    while pref_counter < num_prefs:
        # print(f"No of prefs created {pref_counter}")
        ti = np.random.randint(num_demos)
        tj = np.random.randint(num_demos)
        print(ti,tj)
        if ti <=num_demos-traj_snippet_len and tj <=num_demos-traj_snippet_len:
            reward_ti = np.sum(reward_list[ti:ti + traj_snippet_len])
            reward_tj = np.sum(reward_list[tj:tj + traj_snippet_len])
            print(f"Reward of  traj_i {reward_ti} and reward traj_j  {reward_tj}")
            if reward_ti != reward_tj and(ti, tj) not in pref_indices:
                obs_ti=torch_x_theta_obs_list[ti:ti+traj_snippet_len]
                assert len(obs_ti)==traj_snippet_len
                obs_tj=torch_x_theta_obs_list[tj:tj+traj_snippet_len]
                assert len(obs_tj)==traj_snippet_len
                if reward_ti > reward_tj:
                    pref_label=0
                else:
                    pref_label=1
                traj_pair = torch.stack((obs_ti, obs_tj))

                pref_demos.append(traj_pair)
                pref_labels.append(pref_label)
                pref_counter+=1
    # print(pref_demos)
    # print(pref_labels)
    if save:
        current_directory = os.getcwd()
        file_name=  current_directory +'/Pref_Dataset_num_prefs_'+str(num_prefs)+'_traj_snippet_len_'+str(traj_snippet_len)
        torch.save({'pref_demos': pref_demos, 'pref_labels': pref_labels,'pref_indices':pref_indices}, file_name)
    return pref_demos, pref_labels, pref_indices


if __name__ == '__main__':
    env=gym.make("FH-CartPole")
    num_episodes=100
    timesteps_per_episode=200
    # save=False
    # create_dataset(env,num_episodes,timesteps_per_episode,save)
    dataset=hfile = h5py.File('/home/ak/Documents/Clean_Cartpole_DDT/Dataset_epsiodelen200_num_episodes_' + str(num_episodes) + '.zip')
    num_prefs=2200
    traj_snippet_len=20
    pref_dataset_path='/home/ak/Documents/Clean_Cartpole_DDT/Pref_Dataset_num_prefs_'+str(num_prefs)+'_traj_snippet_len_'+str(traj_snippet_len)
    # x=torch.load(pref_dataset_path)
    # print(x['pref_demos'])
    # print(x['pref_labels'])
    pref_demos,pref_labels,pref_indices=create_pref_dataset(dataset,num_prefs,traj_snippet_len=traj_snippet_len,save=False)