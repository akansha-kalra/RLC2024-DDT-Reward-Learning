__author__ = "akansha_kalra"
import warnings
import gc
warnings.filterwarnings('ignore')
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack,VecEnvWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike
import torch
import torch.nn as nn
import torch.nn.functional as F
import os,random ,argparse
import matplotlib.pylab as plt
import numpy as np
gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--RL_seed', default=0, help="RL/PPO seed for experiments")
parser.add_argument('--timesteps', default=1e7, help="timesteps to run PPO for-setting equal to TREX")
parser.add_argument('--pth', default=".", help="path where tensorboard events are stored")
parser.add_argument('--save_model_dir', default=".", help="where to save trained model")
parser.add_argument('--checkpointing_dir', default=".", help="where to save RL checkpoints")
parser.add_argument('--exp_no', default="XX", help="which experiment number are you on")
parser.add_argument('--checkpointing_freq', default=10000, help="how often to checkpoint the PPO policy")
parser.add_argument('--num_envs', default=16, help="number of vectorized envs to run RL")

args = parser.parse_args()
env_name=args.env_name
RL_seed = int(args.RL_seed)
total_timesteps = int(args.timesteps)
checkpointing_freq = int(args.checkpointing_freq)
checkpointing_path = args.checkpointing_dir
save_model_dir = args.save_model_dir

# lr = float(args.lr)
Exp_name = args.exp_no
num_envs = int(args.num_envs)


tensorboard_pth = args.pth+Exp_name
torch.manual_seed(RL_seed)
random.seed(RL_seed)
np.random.seed(RL_seed)
print(f"seed is {RL_seed} and nof of envs {num_envs}")



class AtariNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, traj):
        '''calculate cumulative return of trajectory'''
        x = traj.permute(0,3,1,2) #get into NCHW format
        #compute forward pass of reward network
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.reshape(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = torch.sigmoid(self.fc2(x))
        return r




def normalize_state(obs):
    return obs / 255.0


# custom masking function for covering up the score/life portions of atari games
def mask_score(obs, env_name):
    obs_copy = obs.copy()
    if env_name == "spaceinvaders" or env_name == "breakout" or env_name == "pong":

        # takes a stack of four observations and blacks out (sets to zero) top n rows
        n = 10
        # no_score_obs = copy.deepcopy(obs)
        obs_copy[:, :n, :, :] = 0
    elif env_name == "beamrider":
        n_top = 16
        n_bottom = 11
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "enduro":
        n_top = 0
        n_bottom = 14
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
        # cuts out place in race, but keeps odometer
    elif env_name == "hero":
        n_top = 0
        n_bottom = 30
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "qbert":
        n_top = 12
        # n_bottom = 0
        obs_copy[:, :n_top, :, :] = 0
        # obs_copy[:,-n_bottom:,:,:] = 0
    elif env_name == "seaquest":
        n_top = 12
        n_bottom = 16
        obs_copy[:, :n_top, :, :] = 0
        obs_copy[:, -n_bottom:, :, :] = 0
        # cuts out divers and oxygen
    elif env_name == "mspacman":
        n_bottom = 15  # mask score and number lives left
        obs_copy[:, -n_bottom:, :, :] = 0
    elif env_name == "videopinball":
        n_top = 15
        obs_copy[:, :n_top, :, :] = 0
    elif env_name == "montezumarevenge":
        n_top = 10
        obs_copy[:, :n_top, :, :] = 0
    else:
        print("NOT MASKING SCORE FOR GAME: " + env_name)
        pass
        # n = 20
        # obs_copy[:,-n:,:,:] = 0
    return obs_copy


def preprocess(ob, env_name):
    # print("masking on env", env_name)
    return mask_score(normalize_state(ob), env_name)


class Trex_Reward(VecEnvWrapper):
    def __init__(self, venv,env_name,Trex_path):
        super(Trex_Reward, self).__init__(venv)
        self.reward_net = AtariNet()
        print(self.reward_net.fc2.bias)
        self.reward_net.load_state_dict(torch.load(Trex_path))
        print(self.reward_net.fc2.bias)


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)
        # self.v_env=v_env

        self.env_name = env_name



    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)

        with torch.no_grad():
            rews_network = self.reward_net.forward(torch.as_tensor(normed_obs).float().to(self.device)).cpu().numpy().squeeze()
        return obs, rews_network, done, infos

    def reset(self, seed=None, options=None):
        return self.venv.reset()



def reward_wrapping_env(env, env_name,Trex_path):

    env = Trex_Reward(env, env_name, Trex_path)
    return env



if __name__ == '__main__':
    # env_name = "breakout"
    if env_name == "breakout":
        Trex_path = 'Trained_TREX/breakout_seed1.params'
    elif env_name== "beamrider":
        Trex_path = 'Trained_Trex/beamrider_seed2.params'
    print(f" Env is {env_name}Trex path: {Trex_path}")

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
    env = reward_wrapping_env(env, env_name, Trex_path)
    print(env)

    checkpointing_callback = CheckpointCallback(save_freq=checkpointing_freq, save_path=checkpointing_path,name_prefix=Exp_name)

    A2C_model = A2C("CnnPolicy", env, seed=RL_seed, tensorboard_log=tensorboard_pth, verbose=1,
                  policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), ent_coef=0.01, vf_coef=0.25)
    A2C_model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=checkpointing_callback)
    A2C_model.save(save_model_dir + Exp_name)