__author__ = "akansha_kalra"
import torch

# from stable_baselines3.c
import warnings
warnings.filterwarnings('ignore')
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--DDT_with_penalty', default=False, type=bool, help="traces for DDT trained-with or without penalty")
parser.add_argument('--save_params_dir',default="Trained_Extracted_Params/", help="where to store the extracted params")

args = parser.parse_args()
env_name=args.env_name
DDT_with_penalty=args.DDT_with_penalty
save_params_dir=args.save_params_dir
if not os.path.exists(save_params_dir):
    os.makedirs(save_params_dir)

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
class Leaf():
    def __init__(self, nb_classes):
        # device = torch.device('cpu')
        device = torch.device("cpu")
        # print(device)
        self.distribution = nn.Parameter(torch.rand(nb_classes))
#         print("leaf distribution", self.distribution)
        self.softmax = nn.Softmax(dim=1)
        self.path_prob = 0


    def forward(self):
      # simply softmax of the learned distribution vector
        return(self.softmax(self.distribution.view(1,-1)))

class Node():
    def __init__(self, depth, nb_classes,module_list):

        self.nb_classes = nb_classes
        device = torch.device("cpu")

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7, stride=2).to(device)
        module_list.append(self.conv1)

        self.fc1 = nn.Linear(6084, 1).to(device)
        module_list.append(self.fc1)

        if depth > 0:
            self.children = self.build_children(depth,module_list)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth,module_list):
        return [Node(depth - 1, self.nb_classes,module_list),
                Node(depth - 1, self.nb_classes,module_list)]
    def forward(self, x):
        x = x.double()
        x=x.permute(0,3,1,2)
        out = F.leaky_relu(self.conv1(x))
        if out.dim()==4:
            input_linear=out.reshape((out.size(dim=0),-1))
        out = self.fc1(input_linear)
        # print("out after applying the linear layer ", out)
        out = torch.sigmoid(out)
        # print("out after sigmoid activation ", out)
        return out
        # return (torch.sigmoid(self.beta*self.fc(x)))
class SoftDecisionTree(nn.Module):
    def __init__(self, depth,class_reward_vector):

        super(SoftDecisionTree, self).__init__()
        self.class_reward = class_reward_vector
        self.nb_classes = len(self.class_reward) # output_dim
        # self.input_size = input_size  # input_dim
        self.depth=depth

        # build tree
        self.module_list = nn.ModuleList()
        self.root = Node(self.depth - 1, self.nb_classes,self.module_list)
        self.nodes = []
        self.leaves = []


        global node_name
        node_name={}
      # set Torch optimizer's parameters
        self.collect_parameters()
        self.device = torch.device("cpu")


    def collect_parameters(self):
        nodes = [self.root]
        self.param_list = nn.ParameterList()
        # self.module_list = nn.ModuleList()
        node_counter=0
        leaf_counter=0

        while nodes:
            node = nodes.pop(0)
            if isinstance(node, Leaf):
                leaf_counter += 1
                self.param_list.append(node.distribution)
                self.leaves.append(node)
            #                 print(self.param_list)
            else:
                node_counter+=1
                nodes.append(node.children[0])
                nodes.append(node.children[1])
                # self.module_list.append(node.fc)
                self.nodes.append(node)
        print(f"total no of leaf nodes are {leaf_counter} for depth {self.depth}")
        print(f"total no of non-leaf nodes are {node_counter} for depth {self.depth}")


    def forward_set_prob(self, current_node, inputs,path_prob):
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
            loss_l = torch.dot(class_reward_vector, Q.reshape(2,))
            loss = torch.sum((loss_l * leaf.path_prob), dim=1)
            loss_tree += loss
        return loss_tree

    def leaf_prob(self):
        prob_list = {}
        for leaf in self.leaves:
            prob_list[leaf] = leaf.path_prob.cpu()
        # print(prob_list)
        return prob_list ,node_name

    def argmax_forward(self,input):
        class_reward_vec = torch.tensor(self.class_reward).double().detach().cpu()
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)

        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        # print(chosen_predictors)
        max_Q = [predictor.forward().detach().cpu() for predictor in chosen_predictors]
        max_Q_tensor = torch.cat(max_Q,dim=0)
        maxLeaf_QR = torch.sum((torch.mul(max_Q_tensor, class_reward_vec.reshape(-1,2))), dim=1)
        return maxLeaf_QR

    def forward(self, input):
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree




if __name__ == '__main__':
    reward_net = SoftDecisionTree(depth=2, class_reward_vector=[0, 1])
    assert env_name is not None
    if env_name == 'breakout':
        if DDT_with_penalty:
            reward_model_pth = 'Trained_DDT_unextracted/Breakout_Penalty-Run-seed2-PB10-NoES_100'
        else:
            reward_model_pth = 'Trained_DDT_unextracted/Breakout_NoPenalty-Run-seed0-PB10-NoES_100'

    elif env_name == 'beamrider':
        if DDT_with_penalty:
            reward_model_pth = 'Trained_DDT_unextracted/BeamRider_Penalty-Run-seed1-PB10-NoES_100'
        else:
            reward_model_pth ='Trained_DDT_unextracted/BeamRider_NoPenalty-Run-1-seed0-PB10_MD_100epochs_withoutES_100'
    print(f"Extracting reward DDT stored at{reward_model_pth}")
    print(reward_net,reward_net.module_list[0].bias)
    print(reward_net.param_list)

    print(f" Trained parameters loading now")
    reward_net = torch.load(reward_model_pth)

    print(reward_net.module_list[0].bias)
    print(reward_net.param_list)
    print(reward_net)
    Leaf_params = np.array(list(param for param in reward_net.param_list.named_parameters()))
    Node_params = np.array(list(reward_net.module_list.named_parameters()))

    np.save(save_params_dir+f"{env_name}_DDT_penalty{DDT_with_penalty}_LEAF", Leaf_params)
    np.save(save_params_dir+f"{env_name}_DDT_penalty{DDT_with_penalty}_NODE", Node_params)
