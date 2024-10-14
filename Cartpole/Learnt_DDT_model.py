__author__ = "akansha_kalra"
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

def get_node_dist_new(node_names, node_prob):
    torch.set_printoptions(precision=10)
    count_prob_dist = {}
    substring="Node"
    for k in node_names.values():
        pointer=list(node_names.keys())[list(node_names.values()).index(k)]
        if substring in str(pointer):
                for ((x,theta),_) in node_prob.keys():
                    count_prob_dist[k]=node_prob[((x,theta),pointer)]
    return count_prob_dist

def get_node_dist(node_names, node_prob):
    torch.set_printoptions(precision=10)
    count_prob_dist = {}
    substring="Node"
    for k in node_names.values():
        pointer=list(node_names.keys())[list(node_names.values()).index(k)]
        if substring in str(pointer):
                for ((x,theta),_) in node_prob.keys():
                    count_prob_dist[((x,theta),k)]=node_prob[((x,theta),pointer)]
    return count_prob_dist

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
        return(self.softmax(self.distribution.view(1,-1)))



class Node():
    def __init__(self, depth, nb_classes, input_size,lmbda):
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
        return [Node(depth - 1, self.nb_classes, self.input_size,self.root_lmbda),
                Node(depth - 1, self.nb_classes, self.input_size,self.root_lmbda)]
    def forward(self, x):
        return (torch.sigmoid(self.beta*self.fc(x)))


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, nb_classes, input_size,class_reward_vector,seed):

        super(SoftDecisionTree, self).__init__()

        torch.manual_seed(seed)
        self.nb_classes = nb_classes  # output_dim
        self.input_size = input_size  # input_dim
        self.depth=depth
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
                '''if node==self.root:
                    self.tree_indexing(node,0)'''
                node_counter+=1
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
            loss = torch.sum((loss_l * leaf.path_prob),dim=1)
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

    def forward_hm(self, current_node, inputs, path_prob, x, theta):

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
        elif current_node == self.root.children[0].children[1].children[1]:
            node_name[current_node] = 10

        if isinstance(current_node, Leaf):
            # print(current_node)
            current_node.path_prob = path_prob
            #             print(f"Current node: {current_node}  has path probability: {path_prob}")
            return  # end of recursion at a leaf

        # set params for penalty
        prob = current_node.forward(inputs)
        current_node.alpha = torch.sum(prob * path_prob) / torch.sum(path_prob)

        pr = prob.data.cpu()[0].detach().numpy()
        if current_node not in node_p:
            node_p[((x, theta), current_node)] = (pr.item())
        else:
            node_p[((x, theta), current_node)].extend(pr)

        # print(f"prob of the current node is {prob} {prob.dtype} and if needed current node is {current_node}")

        # Left Children -> prob = activation
        self.forward_hm(current_node.children[0], inputs, prob * path_prob, x, theta)
        # Right children -> prob = 1 - activation
        self.forward_hm(current_node.children[1], inputs, (1 - prob) * path_prob, x, theta)


    def get_loss_hm(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).double().to(self.device)
        class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        global get_Q
        get_Q = defaultdict(list)
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            get_Q[leaf] = Q
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob),dim=1)
            loss_tree += loss
        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree

    def fwd_input_hm(self,input_traj):
        global node_p
        node_p = defaultdict(list)
        #
        global node_name
        node_name = {}

        traj_reward = 0
        for input_state in input_traj:
            input_state = torch.reshape(input_state, (1, -1))
            x = input_state[0][0]
            theta = input_state[0][1]
            input_state = input_state.to(self.device)
            ones = torch.ones(len(input_state), 1).to(self.device)
            self.forward_hm(self.root, input_state, ones,x,theta)
            state_reward = self.get_loss_hm()
            # print("state reward",state_reward)
            traj_reward+=torch.sum(state_reward)
        # C = self.get_penalty()
        # print(f"value of penalty is {C}")
        final_traj_reward=traj_reward

        #     print("state reward",state_reward)
        # print("final traj reward",final_traj_reward)
        return final_traj_reward ,node_name, node_p,get_Q

    def max_prob_leaf_class(self,input):
        input = torch.reshape(input, (1, -1)).to(self.device)
        class_reward_vec = torch.tensor(self.class_reward).float().detach().cpu()
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward(self.root, input, ones)
        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        # print(chosen_predictors)
        max_Q = list(predictor.forward().detach().cpu() for predictor in chosen_predictors)
        # print(max_Q)
        # max_Q_tensor=torch.from_numpy(np.array(max_Q))
        max_reward= torch.argmax(max_Q[0])
        return max_reward


    def soft_forward_class(self, input):
        input = torch.reshape(input, (1, -1)).to(self.device)
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree