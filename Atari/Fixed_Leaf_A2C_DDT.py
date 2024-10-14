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
# parser.add_argument('--lr', default=3e-4, help="lr for RL")
parser.add_argument('--Leaf_path', default=".", help="path of Trained Leaf params")
parser.add_argument('--Node_path', default=".", help="path of Trained Node params")
parser.add_argument('--soft_routing_argmax',default=0,help="If 0 then it soft routes if it's 1 then it does argmax")

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
Leaf_path=args.Leaf_path
Node_path=args.Node_path
soft_routing_argmax=int(args.soft_routing_argmax)
print(f"You are starting RL with setting parameter soft_routing_argmax {soft_routing_argmax} MAKE SURE RESULTS BELOW MATCH INTENDED VALUE")
tensorboard_pth = args.pth
torch.manual_seed(RL_seed)
random.seed(RL_seed)
np.random.seed(RL_seed)
print(f"seed is {RL_seed}")

if not os.path.exists(checkpointing_path):
    print(' Creating  Checkpointing directory: ' + checkpointing_path)
    os.makedirs(checkpointing_path)

if not os.path.exists(save_model_dir):
    print(' Creating  directory to save final RL model : ' + save_model_dir)
    os.makedirs(save_model_dir)

class Leaf():
    def __init__(self, nb_classes):
        # device = torch.device('cpu')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # print(device)
        self.distribution = nn.Parameter(torch.rand(nb_classes))
        #         print("leaf distribution", self.distribution)
        self.softmax = nn.Softmax(dim=1)
        self.path_prob = 0

    def forward(self):
        # simply softmax of the learned distribution vector
        return (self.softmax(self.distribution.view(1, -1)))


class Node():
    def __init__(self, depth, nb_classes, module_list):

        self.nb_classes = nb_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7, stride=2).to(device)
        module_list.append(self.conv1)

        self.fc1 = nn.Linear(6084, 1).to(device)
        module_list.append(self.fc1)

        # self.root_lmbda = lmbda
        # self.lmbda = lmbda * 2 ** (-depth)

        if depth > 0:
            self.children = self.build_children(depth, module_list)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth, module_list):
        return [Node(depth - 1, self.nb_classes, module_list),
                Node(depth - 1, self.nb_classes, module_list)]

    def forward(self, x):
        x = x.double()
        x = x.permute(0, 3, 1, 2)
        out = F.leaky_relu(self.conv1(x))
        # print("output after applying Conv layer", out)
        # reshape output for Linear Layer
        if out.dim() == 4:
            input_linear = out.reshape((out.size(dim=0), -1))
        # elif out.dim()==3:
        #     input_linear=torch.flatten(out)

        out = self.fc1(input_linear)
        out = torch.sigmoid(out)

        return out


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, class_reward_vector):

        super(SoftDecisionTree, self).__init__()
        self.class_reward = class_reward_vector
        self.nb_classes = len(self.class_reward)  # output_dim
        # self.input_size = input_size  # input_dim
        self.depth = depth

        # build tree
        self.module_list = nn.ModuleList()
        self.root = Node(self.depth - 1, self.nb_classes, self.module_list)
        self.nodes = []
        self.leaves = []

        global node_name
        node_name = {}
        # set Torch optimizer's parameters
        self.collect_parameters()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def collect_parameters(self):
        nodes = [self.root]
        self.param_list = nn.ParameterList()
        # self.module_list = nn.ModuleList()
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
                node_counter += 1
                nodes.append(node.children[0])
                nodes.append(node.children[1])
                # self.module_list.append(node.fc)
                self.nodes.append(node)
        print(f"total no of leaf nodes are {leaf_counter} for depth {self.depth}")
        print(f"total no of non-leaf nodes are {node_counter} for depth {self.depth}")

    def forward_set_prob(self, current_node, inputs, path_prob):
        if current_node == self.root:
            node_name[current_node] = 0
        elif current_node == self.root.children[0]:
            node_name[current_node] = 1
        elif current_node == self.root.children[1]:
            node_name[current_node] = 2
        elif current_node == self.root.children[0].children[0]:
            node_name[current_node] = 3
        elif current_node == self.root.children[0].children[1]:
            node_name[current_node] = 4
        elif current_node == self.root.children[1].children[0]:
            node_name[current_node] = 5
        elif current_node == self.root.children[1].children[1]:
            node_name[current_node] = 6
        elif current_node == self.root.children[0].children[0].children[0]:
            node_name[current_node] = 7
        elif current_node == self.root.children[0].children[0].children[1]:
            node_name[current_node] = 8
        elif current_node == self.root.children[0].children[1].children[0]:
            node_name[current_node] = 9
        if isinstance(current_node, Leaf):
            current_node.path_prob = path_prob
            #             print(f"Current node: {current_node}  has path probability: {path_prob}")
            return  # end of recursion at a leaf

        prob = current_node.forward(inputs)
        '''if torch.any(path_prob == 0) or torch.any(torch.isnan(prob)) == True:
            print("Tree is dying, one cause: heavy penalty")'''

        # Left Children -> prob = activation
        self.forward_set_prob(current_node.children[0], inputs, prob * path_prob)
        # # Right children -> prob = 1 - activation
        self.forward_set_prob(current_node.children[1], inputs, (1 - prob) * path_prob)

    def get_loss(self):
        class_reward_vector = torch.tensor(self.class_reward).double().to(self.device)
        loss = 0
        loss_tree = 0
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            loss_l = torch.dot(class_reward_vector, Q.reshape(2, ))
            loss = torch.sum((loss_l * leaf.path_prob), dim=1)
            loss_tree += loss
        return loss_tree

    def leaf_prob(self):
        prob_list = {}
        for leaf in self.leaves:
            prob_list[leaf] = leaf.path_prob.cpu()
        # print(prob_list)
        return prob_list, node_name

    def argmax_forward(self, input):
        class_reward_vec = torch.tensor(self.class_reward).double().detach().cpu()
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        # leaf_object_prob, _ = self.leaf_prob()
        # print(leaf_object_prob)
        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        # print(chosen_predictors)
        max_Q = [predictor.forward().detach().cpu() for predictor in chosen_predictors]
        max_Q_tensor = torch.cat(max_Q, dim=0)
        maxLeaf_QR = torch.sum((torch.mul(max_Q_tensor, class_reward_vec.reshape(-1, 2))), dim=1)
        # prod=max_Q_tensor*class_reward[None,:]
        # maxLeaf_QR=torch.sum(prod,dim=1,keepdim=True)
        # print(maxLeaf_QR)
        return maxLeaf_QR

    def soft_forward(self, input):
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree

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

class DDT_Reward(VecEnvWrapper):
    def __init__(self, venv,env_name,trained_leaf_params,trained_node_params,soft_routing):
        super(DDT_Reward, self).__init__(venv)
        self.reward_net = SoftDecisionTree(depth=2, class_reward_vector=[0, 1])
        print(self.reward_net)
        # print([param for param in self.reward_net.param_list])
        # print(f"Trained leaf params are {trained_leaf_params}")

        for leaf, leaf_distribution in trained_leaf_params:
            leaf_index = int(leaf)
            self.reward_net.leaves[leaf_index].distribution = leaf_distribution


        for node, node_param in trained_node_params:
            node_num, node_param_type = node.split(".")
            node_num = int(node_num)
            if node_param_type == "weight":
                self.reward_net.module_list[node_num].weight = node_param
            else:
                self.reward_net.module_list[node_num].bias = node_param


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.reward_net.to(self.device)


        self.env_name = env_name
        self.soft_routing = soft_routing
        if self.soft_routing==0:
            print("RL on Soft DDT output")
        elif self.soft_routing==1:
            print("RL on Argmax DDT output")

    def step_wait(self):
        obs, rews, done, infos = self.venv.step_wait()

        # mask and normalize for input to network
        normed_obs = preprocess(obs, self.env_name)

        with torch.no_grad():

            if self.soft_routing == 0:
                rews_network = self.reward_net.soft_forward(torch.as_tensor(normed_obs).float().to(self.device)).cpu().numpy().squeeze()
            elif self.soft_routing==1:
                rews_network = self.reward_net.argmax_forward(torch.as_tensor(normed_obs).float().to(self.device)).cpu().numpy().squeeze()
        # done=terminated or truncated

        return obs, rews_network, done, infos

    def reset(self, seed=None, options=None):
        return self.venv.reset()



def reward_wrapping_env(env, env_name,Leaf_path,Node_path, soft_routing):
    trained_leaf_params = np.load(Leaf_path, allow_pickle=True)
    trained_leaf_params = trained_leaf_params.tolist()

    trained_node_params = np.load(Node_path, allow_pickle=True)
    trained_node_params = trained_node_params.tolist()

    env = DDT_Reward(env, env_name, trained_leaf_params,trained_node_params, soft_routing)
    return env


if __name__ == '__main__':
    # env_name = "breakout"
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

    env = reward_wrapping_env(env,env_name,Leaf_path,Node_path,soft_routing=soft_routing_argmax)
    print(env)

    checkpointing_callback = CheckpointCallback(save_freq=checkpointing_freq, save_path=checkpointing_path,
                                                name_prefix=Exp_name)

    A2C_model = A2C("CnnPolicy", env, seed=RL_seed, tensorboard_log=tensorboard_pth, verbose=1,
                    policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)), ent_coef=0.01,
                    vf_coef=0.25)
    A2C_model.learn(total_timesteps=total_timesteps, progress_bar=True, callback=checkpointing_callback)
    A2C_model.save( save_model_dir+ Exp_name)