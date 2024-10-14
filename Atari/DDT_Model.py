__author__ = "akansha_kalra"
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os,re
import matplotlib.pylab as plt
torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True)
from collections import defaultdict


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
        return(self.softmax(self.distribution.view(1,-1)))

class Node():
    def __init__(self, depth, nb_classes,module_list):

        self.nb_classes = nb_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=7, stride=2).to(device)
        module_list.append(self.conv1)

        self.fc1 = nn.Linear(6084, 1).to(device)
        module_list.append(self.fc1)

        # self.root_lmbda = lmbda
        # self.lmbda = lmbda * 2 ** (-depth)

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
        # print("output after applying Conv layer", out)
        #reshape output for Linear Layer
        if out.dim()==4:
            input_linear=out.reshape((out.size(dim=0),-1))
        # elif out.dim()==3:
        #     input_linear=torch.flatten(out)

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

        global node_p
        node_p = defaultdict(list)
        #
        global node_name
        node_name={}
      # set Torch optimizer's parameters
        self.collect_parameters()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        pr = prob.data.cpu()[0].detach().numpy()

        if current_node not in node_p:
            node_p[current_node] = (pr)
            # print("f appending {pr}")

        #
        # num_bound = 1e-8
        # alpha = torch.sum((prob * path_prob), dim=1) / (torch.sum(path_prob, dim=1) + num_bound)
        # alpha = torch.clamp(alpha, num_bound, 1 - num_bound)
        # current_node.alpha = alpha

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
        # leaf_object_prob, _ = self.leaf_prob()
        # print(leaf_object_prob)
        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        # print(chosen_predictors)
        max_Q = [predictor.forward().detach().cpu() for predictor in chosen_predictors]
        max_Q_tensor = torch.cat(max_Q,dim=0)
        maxLeaf_QR = torch.sum((torch.mul(max_Q_tensor, class_reward_vec.reshape(-1,2))), dim=1)
        # prod=max_Q_tensor*class_reward[None,:]
        # maxLeaf_QR=torch.sum(prod,dim=1,keepdim=True)
        # print(maxLeaf_QR)
        return maxLeaf_QR

    def forward(self, input):
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree

    def get_leaf_distribution(self):
        global get_Q
        get_Q = defaultdict(list)
        for leaf in self.leaves:
            # print(f"Q before transporsing {leaf.forward(),leaf.forward().size()}")
            Q = torch.transpose(leaf.forward(), 0, 1).double()
            get_Q[leaf] = Q
        return get_Q

    def leaf_vis_tree(self, inputs):
        self.eval()
        inputs = inputs.to(self.device)
        # inputs = inputs.view(len(inputs), -1)
        ones = torch.ones((len(inputs), 1)).to(self.device)
        # self.forward(self.root, inputs, ones)
        self.forward_set_prob(self.root, inputs, ones)
        get_Q = self.get_leaf_distribution()
        return node_name, node_p, get_Q

    def vis_evaluate_tree(self, inputs):

        self.eval()
        node_name.clear()
        node_p.clear()
        inputs = inputs.to(self.device)
        # inputs = inputs.view(len(inputs), -1)
        ones = torch.ones((len(inputs), 1)).to(self.device)
        '''UNcomment if plotting heatmaps- and add r,c to be given as input'''
        # self.forward_with_rc(self.root, inputs, ones, r, c)
        self.forward_set_prob(self.root, inputs, ones)

        return node_name, node_p

    def fwd_input_state(self, input_state):
        traj_reward = 0
        input_state = input_state.to(self.device)
        input_state = input_state.unsqueeze(0)
        ones = torch.ones(len(input_state), 1).to(self.device)
        self.forward_set_prob(self.root, input_state, ones)
        state_reward = self.get_loss()
        # print("state reward",state_reward)
        traj_reward += torch.sum(state_reward)

        final_traj_reward = traj_reward
        #
        # #     print("state reward",state_reward)
        # # print("final traj reward",final_traj_reward)
        return final_traj_reward