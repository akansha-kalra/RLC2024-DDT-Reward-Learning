__author__ = "akansha_kalra"

import gymnasium as gym
from gymnasium.envs.registration import register
from stable_baselines3.common.vec_env import VecFrameStack,VecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch
import random
import torch.nn as nn
import numpy as np
from collections import defaultdict
import os,random ,argparse
import matplotlib.pylab as plt
import numpy as np

register(
    id="FH-CartPole",
    entry_point="Fixed_Horizon_CP_Env:FH_CartPoleEnv",
    vector_entry_point="Fixed_Horizon_CP_Env:FH_CartPoleVectorEnv",
    max_episode_steps=200,
    reward_threshold=200,
)

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--soft_routing_argmax',default=1,help="If 0 then it soft routes if it's 1 then it does argmax")
parser.add_argument('--RL_seed', default=0, help="RL/PPO seed for experiments")
parser.add_argument('--exp_no', default="XX", help="which experiment number are you on")
parser.add_argument('--save_model_dir', default="/home/ak/Documents/Clean_Cartpole_DDT/RL_using_Reward_Models/DDT/saved_models/", help="where to save trained model")
parser.add_argument('--pth', default="RL_using_Reward_Models/DDT/TB/", help="path where tensorboard events are stored")

args = parser.parse_args()
RL_seed = int(args.RL_seed)
save_model_dir = args.save_model_dir
Exp_name = args.exp_no
soft_routing_argmax=int(args.soft_routing_argmax)
tensorboard_pth = args.pth+Exp_name
print(f"You are starting RL with setting parameter soft_routing_argmax {soft_routing_argmax} MAKE SURE RESULTS BELOW MATCH INTENDED VALUE")

torch.manual_seed(RL_seed)
random.seed(RL_seed)
np.random.seed(RL_seed)
print(f"seed is {RL_seed}")

if not os.path.exists(save_model_dir):
    print(' Creating  directory to save final RL model : ' + save_model_dir)
    os.makedirs(save_model_dir)

class Leaf():
    def __init__(self, nb_classes):
        # device = torch.device('cpu')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.distribution = nn.Parameter(torch.rand(nb_classes).to(device))
        #         print("leaf distribution", self.distribution)
        self.softmax = nn.Softmax(dim=1)
        self.path_prob = 0

    def forward(self):
        # simply softmax of the learned distribution vector
        return (self.softmax(self.distribution.view(1, -1)))


class Node():
    def __init__(self, depth, nb_classes, input_size, lmbda):
        self.input_size = input_size
        self.nb_classes = nb_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc = nn.Linear(self.input_size, 1).to(device)
        self.beta = nn.Parameter(torch.rand(1).to(device))  # inverse temperature
        # to compute penalty
        self.root_lmbda = lmbda
        self.lmbda = lmbda * 2 ** (-depth)
        self.alpha = 0  # will be set according to inputs

        if depth > 0:
            self.children = self.build_children(depth)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth):
        return [Node(depth - 1, self.nb_classes, self.input_size, self.root_lmbda),
                Node(depth - 1, self.nb_classes, self.input_size, self.root_lmbda)]

    def forward(self, x):
        return (torch.sigmoid(self.beta * self.fc(x)))


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, nb_classes, input_size, class_reward_vector, seed):

        super(SoftDecisionTree, self).__init__()

        torch.manual_seed(seed)
        self.nb_classes = nb_classes  # output_dim
        self.input_size = input_size  # input_dim
        self.depth = depth
        self.class_reward = class_reward_vector

        # build tree
        self.root = Node(self.depth - 1, self.nb_classes, self.input_size, lmbda=0.1)
        self.nodes = []
        self.leaves = []
        self.collect_parameters()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def collect_parameters(self):
        nodes = [self.root]
        self.param_list = nn.ParameterList()
        self.module_list = nn.ModuleList()
        node_counter = 0
        leaf_counter = 0

        while nodes:
            node = nodes.pop(0)
            if isinstance(node, Leaf):
                leaf_counter += 1
                self.param_list.append(node.distribution)
                self.leaves.append(node)
            #                 print(self.param_list)
            else:
                '''if node==self.root:
                    self.tree_indexing(node,0)'''
                node_counter += 1
                nodes.append(node.children[0])
                nodes.append(node.children[1])
                self.module_list.append(node.fc)
                self.nodes.append(node)
        print(f"total no of leaf nodes are {leaf_counter} for depth {self.depth}")
        print(f"total no of non-leaf nodes are {node_counter} for depth {self.depth}")

    def forward(self, current_node, inputs, path_prob):

        if isinstance(current_node, Leaf):
            current_node.path_prob = path_prob
            return
        prob = current_node.forward(inputs)

        # Left Children -> prob = activation
        self.forward(current_node.children[0], inputs, prob * path_prob)
        # Right children -> prob = 1 - activation
        self.forward(current_node.children[1], inputs, (1 - prob) * path_prob)

    def get_loss(self):
        class_reward = torch.tensor(self.class_reward).float().to(self.device)
        class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).to(self.device)
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob), dim=1)
            loss_tree += loss

        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree

    def fwd_input(self, input_traj):
        traj_reward = 0
        for input_state in input_traj:
            # print(input_state)
            input_state = input_state.to(self.device)
            input_state = torch.reshape(input_state, (1, -1))
            ones = torch.ones(len(input_state), 1).to(self.device)
            self.forward(self.root, input_state, ones)
            state_reward = self.get_loss()
            # print("state reward",state_reward)
            traj_reward += torch.sum(state_reward)

        final_traj_reward = traj_reward

        #     print("state reward",state_reward)
        # print("final traj reward",final_traj_reward)
        return final_traj_reward

    def soft_forward(self, input):
        input = torch.reshape(input, (-1, 2)).to(self.device)
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree

    def argmax_forward(self, input):
        input = torch.reshape(input, (-1, 2)).to(self.device)
        class_reward_vec = torch.tensor(self.class_reward).float().detach().cpu()
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward(self.root, input, ones)
        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        # print(chosen_predictors)
        max_Q = list(predictor.forward().detach().cpu() for predictor in chosen_predictors)
        # print(max_Q)
        # max_Q_tensor=torch.from_numpy(np.array(max_Q))
        '''for multiple vectorized envs'''
        max_reward=torch.argmax(torch.stack(max_Q), dim=2)
        # max_reward = torch.argmax(max_Q[0])

        return max_reward

class DDT_Reward(VecEnvWrapper):
    def __init__(self, venv,trained_DDT,soft_routing):
        super(DDT_Reward, self).__init__(venv)
        input_dim = 1 * 2
        depth = 2
        class_reward_vector = [0, 1]
        nb_classes = len(class_reward_vector)
        self.reward_net= SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector,seed=0)
        print(self.reward_net)
        self.reward_net=torch.load(trained_DDT)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)
        self.soft_routing = soft_routing
        if self.soft_routing==0:
            print("RL on Soft DDT output")
        elif self.soft_routing==1:
            print("RL on Argmax DDT output")

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()
        # input_to_ddt = torch.take(torch.from_numpy(obs), torch.tensor([0, 2])).float().reshape(1, 2)
        input_to_ddt = torch.index_select(torch.from_numpy(obs).unsqueeze(dim=1),dim=2,index=torch.tensor([0,2]))
        with torch.no_grad():

            if self.soft_routing == 0:
                rews_network = self.reward_net.soft_forward(torch.as_tensor(input_to_ddt).to(self.device)).cpu().numpy().squeeze()
            elif self.soft_routing==1:
                rews_network = self.reward_net.argmax_forward(torch.as_tensor(input_to_ddt).to(self.device)).cpu().numpy().squeeze()
        # done=terminated or truncated

        return obs, rews_network, done, infos

    def reset(self, seed=None, options=None):
        return self.venv.reset()



def reward_wrapping_env(env,trained_DDT, soft_routing):

    env = DDT_Reward(env, trained_DDT, soft_routing)
    return env

if __name__=="__main__":


    env_id="FH-CartPole"

    trained_DDT_path= 'Trained_Reward_Models/DDT/saved_models/CP-1_50'

    vec_env = make_vec_env(env_id, n_envs=5,seed=0)
    env=reward_wrapping_env(vec_env,trained_DDT_path,soft_routing=soft_routing_argmax)

    model = PPO("MlpPolicy", env, batch_size=1024, gae_lambda=0.8, gamma=0.98, learning_rate=0.001, n_epochs=20, n_steps=2048, verbose=1, seed=RL_seed, tensorboard_log=tensorboard_pth)
    model.learn(total_timesteps=5e5,progress_bar=True)
    model.save(save_model_dir + Exp_name)








