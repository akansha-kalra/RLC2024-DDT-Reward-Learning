__author__ = "akansha_kalra"
from gymnasium.envs.registration import register
import gymnasium as gym
from stable_baselines3.common.vec_env import VecFrameStack,VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import random
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
import os,random ,argparse
import matplotlib.pylab as plt

register(
    id="FH-SS-CartPole",
    entry_point="Fixed_Horizon_Side_Start_CP_Env:FH_SideStart_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_Side_Start_CP_Env:FH_SideStart_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)


parser = argparse.ArgumentParser(description=None)

parser.add_argument('--RL_seed', default=0, help="RL/PPO seed for experiments")
parser.add_argument('--exp_no', default="Side-RL_using_NN-1", help="which experiment number are you on")
parser.add_argument('--save_model_dir', default="Misalignment/RL_using_NN/", help="where to save trained model")
parser.add_argument('--pth', default="Misalignment/RL_runs_TB/RL_using_NN/", help="path where tensorboard events are stored")
parser.add_argument('--start_side',default=0.0, help ="x position to start RL at ")
args = parser.parse_args()
RL_seed = int(args.RL_seed)
save_model_dir = args.save_model_dir
Exp_name = args.exp_no



torch.manual_seed(RL_seed)
random.seed(RL_seed)
np.random.seed(RL_seed)
print(f"seed is {RL_seed}")

if not os.path.exists(save_model_dir):
    print(' Creating  directory to save final RL model : ' + save_model_dir)
    os.makedirs(save_model_dir)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, traj):
        input_traj = traj.to(self.device)
        # sum_rewards=0
        x = F.leaky_relu(self.fc1(input_traj))
        x = F.leaky_relu(self.fc2(x))
        r = self.fc3(x)
        return r


class NN_Reward(VecEnvWrapper):
    def __init__(self, venv,trained_nn_net_path):
        super(NN_Reward, self).__init__(venv)
        self.reward_net= SimpleNet()
        print(self.reward_net)
        self.reward_net=torch.load(trained_nn_net_path)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()
        input_to_net =torch.index_select(torch.from_numpy(obs).unsqueeze(dim=1),dim=2,index=torch.tensor([0,2]))
        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.as_tensor(input_to_net).to(self.device)).cpu().numpy().squeeze()
        # print(f"Reward by RL_using_NN is {rews_network}")
        return obs, rews_network, done, infos

    def reset(self, seed=None, options=None):
        return self.venv.reset()



def reward_wrapping_env(env,trained_nn_net_path):

    env = NN_Reward(env, trained_nn_net_path)
    return env

if __name__=="__main__":
    env_id = "FH-SS-CartPole"
    start_side = float(args.start_side)

    trained_nn_net_path= '/home/ak/Documents/Clean_Cartpole_DDT/Reward_Models/NN/saved_models/CP-NN-1_38'

    total_timesteps = 5e5
    # Parallel environments
    env_kwargs = dict(side=start_side)
    vec_env = make_vec_env(env_id, n_envs=5, seed=0,env_kwargs=env_kwargs)
    env=reward_wrapping_env(vec_env,trained_nn_net_path)

    Exp_name = Exp_name + '_start_side' + str(start_side) + '_ts_' + str(total_timesteps)
    tensorboard_pth = args.pth + Exp_name

    model = PPO("MlpPolicy", env, batch_size=1024, gae_lambda=0.8, gamma=0.98, learning_rate=0.001, n_epochs=20, n_steps=2048, verbose=1,seed=RL_seed,tensorboard_log=tensorboard_pth)
    model.learn(total_timesteps=total_timesteps,progress_bar=True)
    model.save(save_model_dir + Exp_name)
