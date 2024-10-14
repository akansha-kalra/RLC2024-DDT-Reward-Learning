import numpy as np
import torch
import torch.nn as nn

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
    def __init__(self, depth, nb_classes, input_size):
        self.input_size = input_size
        self.nb_classes = nb_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc = nn.Linear(self.input_size, 1).to(device)
        self.beta = nn.Parameter(torch.rand(1).to(device))  # inverse temperature
        # # to compute penalty
        # self.root_lmbda = lmbda
        # self.lmbda = lmbda * 2 ** (-depth)
        # self.alpha = 0  # will be set according to inputs


        if depth > 0:
            self.children = self.build_children(depth)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth):
        return [Node(depth - 1, self.nb_classes, self.input_size),
                Node(depth - 1, self.nb_classes, self.input_size)]
    def forward(self, x):
        return (torch.sigmoid(self.beta*self.fc(x)))

class SoftDecisionTree(nn.Module):
    def __init__(self, depth, input_size,class_reward_vector):

        super(SoftDecisionTree, self).__init__()
        self.class_reward = class_reward_vector
        self.nb_classes = len(self.class_reward)  # output_dim
        self.input_size = input_size  # input_dim
        self.depth=depth


        # build tree
        self.root = Node(self.depth - 1, self.nb_classes, self.input_size)
        self.nodes = []
        self.leaves = []

        # global node_p
        # node_p = defaultdict(list)
        #
        global node_name
        node_name={}
      # set Torch optimizer's parameters
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
        print(
            f"total no of leaf nodes are {leaf_counter}  and non-leaf nodes are {node_counter} for depth {self.depth}")

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

    def leaf_prob(self):
        prob_list = {}
        for leaf in self.leaves:
            prob_list[leaf] = leaf.path_prob.cpu()
        # print(prob_list)
        return prob_list, node_name

    '''this forward gives the argmax reward'''

    def forward(self, input):
        input = torch.reshape(input, (1, -1)).to(self.device)
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

    def get_loss(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).double().to(self.device)
        # class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            loss_l = torch.inner(class_reward, Q.reshape(2,))
            loss = torch.sum((loss_l * leaf.path_prob),dim=1)
            loss_tree += loss
        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree

    def soft_forward(self, input):
        input = torch.reshape(input, (1, -1)).to(self.device)
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree
