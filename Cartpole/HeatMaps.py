__author__ = "akansha_kalra"

import os

import h5py
# import matplotlib
# matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore")
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from Learnt_DDT_model import Leaf, Node,SoftDecisionTree,get_node_dist_new
import seaborn as sns



np.set_printoptions(precision=4)
min_PA_radians=-31
max_PA_radians=38



pos_list= [i for i in np.arange(-4.8,4.8,0.05)]
# pos_list= [i for i in np.arange(min_x,max_x+1,0.5)]
angle_list=np.linspace(min_PA_radians, max_PA_radians, num=300)
print(angle_list)

print(f"No of  Pole angles : {len(angle_list)} and no of CP x values {len(pos_list)}")

def node_prob_matrix(network):
    np.set_printoptions(precision=4)
    prob_routing_matrix_node0=np.zeros((len(pos_list),len(angle_list)))
    prob_routing_matrix_node1 = np.zeros((len(pos_list),len(angle_list)))
    prob_routing_matrix_node2 = np.zeros((len(pos_list),len(angle_list)))
    # r = 0
    # c = 0
    for r in range(len(pos_list)):
        for c in range(len(angle_list)):
            # print(r,c)
            x=pos_list[r]
            theta=angle_list[c]
            input_state = torch.from_numpy(np.asarray((x, theta))).float().reshape(1, 2)
            _, n_name, n_pr, _ = network.fwd_input_hm(input_state)
            # print(n_name)
            # break
            input_state_routing_prob = get_node_dist_new(n_name, n_pr)
            # print(input_state_routing_prob)

            prob_routing_matrix_node0[r, c] = input_state_routing_prob[0]
            prob_routing_matrix_node1[r, c] = input_state_routing_prob[ 1]
            prob_routing_matrix_node2[r, c] = input_state_routing_prob[2]

    return prob_routing_matrix_node0, prob_routing_matrix_node1, prob_routing_matrix_node2

def plot_heatmap(prob_matrix,node_counter=None,save_fig_dir=None):
    pos_list = np.round(np.linspace(-4.8, 4.8, num=5).tolist(), 1)
    print(pos_list)
    theta_list = np.round(np.linspace(-3.2, 3.2, num=5).tolist(), 1)
    print(theta_list)
    p = sns.heatmap(prob_matrix)
    import matplotlib.ticker as ticker
    p.yaxis.set_major_locator(ticker.LinearLocator(5))
    p.xaxis.set_major_locator(ticker.LinearLocator(5))
    p.set_xticklabels(theta_list, size=12)
    p.set_yticklabels(pos_list, size=12)
    plt.ylabel("Cart position", fontsize=22)
    plt.xlabel("Pole Angle", fontsize=22)
    plt.tight_layout()
    if save_fig_dir is not None:
        ps = prob_matrix.shape
        plt.savefig(
            save_fig_dir + "Heatmaps for node {node_counter} of size{ps}".format(node_counter=node_counter, ps=ps))
    plt.show()



if __name__=="__main__":
    input_dim = 1 * 2
    depth = 2
    class_reward_vector = [0, 1]
    nb_classes = len(class_reward_vector)

    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector,seed=0)
        print(tree)

        tree= torch.load('Trained_Reward_Models/DDT/saved_models/CP-1_50')
        tree.eval()

        print(f"----------------------------LOADING TRAINED TREE ----------------------------------------------------------------------")

        # print("Node distribution", list(tree.module_list.named_parameters()))
        # print("Leaf distribution", [param for param in tree.param_list])

        p0, p1, p2 = node_prob_matrix(tree)
        print(f"dim of the node prob matrix is {p0.shape}")
        current_directory = os.getcwd()
        '''uncomment to save'''
        # save_fig_dir=  current_directory + '/Reward_Models/DDT/Vis/CP-1/'
        # if not os.path.exists(save_fig_dir):
        #     os.makedirs(save_fig_dir)
        save_fig_dir=None
        plot_heatmap(p0,0,save_fig_dir=save_fig_dir)
        plot_heatmap(p1, 1,save_fig_dir=save_fig_dir)
        plot_heatmap(p2, 2, save_fig_dir=save_fig_dir)
