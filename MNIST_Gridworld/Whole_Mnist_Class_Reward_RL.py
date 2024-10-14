import torch
import random
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset,DataLoader
import matplotlib.pylab as plt
from torchvision import transforms , datasets
from collections import defaultdict
from RL_helper_functions import value_iteration, get_optimal_policy, policy_evaluation_GT_reward, evaluate_random_policy , MDP , print_array_as_grid , visualize_policy
# import matplotlib as mpl
# mpl.use('Qt5Agg')



seed=0
torch.manual_seed(seed)

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
        global node_name
        node_name = {}
        global R_leaf
        R_leaf = {}
        global node_p
        node_p = defaultdict()

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
            if isinstance(node,Leaf):
                leaf_counter += 1
                self.param_list.append(node.distribution)
                self.leaves.append(node)
            #                 print(self.param_list)
            else:
                node_counter+=1
                nodes.append(node.children[0])
                nodes.append(node.children[1])
                self.module_list.append(node.fc)
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

        # Left Children -> prob = activation
        self.forward_set_prob(current_node.children[0], inputs, prob * path_prob)
        # # Right children -> prob = 1 - activation
        self.forward_set_prob(current_node.children[1], inputs, (1 - prob) * path_prob)

    def get_loss(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).double().to(self.device)
        class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob), dim=1)
            loss_tree += loss
        return loss_tree

    def soft_forward_class(self, input):
        input = torch.reshape(input, (1, -1)).to(self.device)
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        reward_tree = self.get_loss()
        return reward_tree

    def max_prob_leaf_class(self,input):
        input = torch.reshape(input, (1, -1)).to(self.device)
        class_reward_vec = torch.tensor(self.class_reward).double().detach().cpu()
        ones = torch.ones((len(input), 1)).to(self.device)
        self.forward_set_prob(self.root, input, ones)
        chosen_predictors = [max(self.leaves, key=lambda leaf: leaf.path_prob[i]) for i in range(len(input))]
        max_Q = list(predictor.forward().detach().cpu() for predictor in chosen_predictors)
        max_reward= torch.argmax(max_Q[0])
        return max_reward

def custom_index_dataset(index_to_use):
    dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    for i in range(len(index_to_use)):
        # print(i)
        if i==0:
            idx = dataset.train_labels==index_to_use[i]
        else:
            idx += dataset.train_labels==index_to_use[i]

    train_labels = dataset.train_labels[idx]
    train_data = dataset.train_data[idx]
    dset_train = torch.utils.data.dataset.Subset(dataset, np.where(idx==1)[0])
    return dset_train


if __name__ == '__main__':
    policy_file = open('Whole_Mnist_Class_Reward_Gridworld_RL_Eval_depth4.txt', 'w')
    policy_file.write(
        "GT Optimal Policy "+ " Class-Reward-Leaf DDT Soft " + " Class-Reward-Leaf DDT Argmax ")
    policy_file.write('\n')


    with torch.no_grad():
        class_reward_leaf_tree_depth = 4
        class_reward_leaf_tree_input_size = 28 * 28
        class_reward_leaf_tree_vector = [0, 1, 2, 3,4,5,6,7,8,9]
        class_reward_leaf_tree = SoftDecisionTree(class_reward_leaf_tree_depth,
                                                               class_reward_leaf_tree_input_size,
                                                               class_reward_leaf_tree_vector)




        class_reward_leaf_tree = torch.load( 'Trained_Reward_Models/mnist0-9/DDT/CRL/WOP-2_WHOLE_NoES_depth4_100')

        for sd in range(0,100,1):
            print(f"NEW SEEED STARTING : SEED{sd}")

            seed = sd
            random.seed(seed)
            torch.manual_seed(seed)

            '''Sampling'''

            training_dataset = custom_index_dataset([0, 1, 2, 3,4,5,6,7,8,9])

            idx = torch.randint(len(training_dataset), (100,))


            gridworld_input = []
            gridworld_gt_reward = []
            for i in idx:
                img, l = training_dataset[i]
                gridworld_input.append(img)
                gridworld_gt_reward.append(l)

            Class_Leaf_r_hat_gridworld_SOFT = []
            Class_Leaf_r_hat_gridworld_ArgMax = []

            for i in range(len(gridworld_input)):
                curr_grid_input = gridworld_input[i]
                Class_Leaf_soft_Curr_reward = class_reward_leaf_tree.soft_forward_class(curr_grid_input)
                Class_Leaf_Argmax_curr_grid_reward = class_reward_leaf_tree.max_prob_leaf_class(curr_grid_input)
                Class_Leaf_r_hat_gridworld_SOFT.append(Class_Leaf_soft_Curr_reward.item())
                Class_Leaf_r_hat_gridworld_ArgMax.append(Class_Leaf_Argmax_curr_grid_reward.item())

            gamma = 0.9  # discount factor

            terminals = []  # set no state to be terminal

            env_GT = MDP(5, 5, terminals, gridworld_gt_reward, gamma)
            env_Class_DDT_SOFT = MDP(5, 5, terminals, Class_Leaf_r_hat_gridworld_SOFT, gamma)
            env_Class_DDT_Argmax = MDP(5, 5, terminals, Class_Leaf_r_hat_gridworld_ArgMax, gamma)


            '''SOFT TREE'''
            print("--- Class Reward DDT SOFT---")
            value_under_r_hat_SOFT = value_iteration(env_Class_DDT_SOFT)
            learned_SOFT_DDT_policy = get_optimal_policy(env_Class_DDT_Argmax, epsilon=0.0001, V=value_under_r_hat_SOFT)
            SOFT_DDT_policy_eval = policy_evaluation_GT_reward(env_GT, learned_SOFT_DDT_policy, epsilon=0.0001)
            value_policy_eval_Soft_Class_DDT=np.mean(SOFT_DDT_policy_eval)
            print(f"Evaluating Learned SOFT- CLASS REWARD LEAF DDT policy under GT reward {value_policy_eval_Soft_Class_DDT}")


            print("--Class Reward DDT Argmax---")

            value_under_r_hat_Argmax = value_iteration(env_Class_DDT_SOFT)
            learned_Argmax_DDT_policy = get_optimal_policy(env_Class_DDT_Argmax, epsilon=0.0001, V=value_under_r_hat_Argmax)
            Argmax_DDT_policy_eval = policy_evaluation_GT_reward(env_GT, learned_Argmax_DDT_policy, epsilon=0.0001)
            value_policy_eval_Argmax_Class_DDT=np.mean(Argmax_DDT_policy_eval)
            print(f"Evaluating Learned Argmax-CLASS REWARD LEAF DDT policy under GT reward {value_policy_eval_Argmax_Class_DDT}")

            value_pi_optimal = value_iteration(env_GT)
            value_optimal = np.mean(value_pi_optimal)
            print("value function for optimal policy under GT reward", value_optimal)
            Optimal_policy_under_GT_reward = get_optimal_policy(env_GT, epsilon=0.0001, V=value_pi_optimal)
            print("-" * 80)

            policy_file.write(f'{value_optimal}, {value_policy_eval_Soft_Class_DDT},{value_policy_eval_Argmax_Class_DDT}')
            policy_file.write('\n')


policy_file.close()