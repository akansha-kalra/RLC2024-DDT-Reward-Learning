import os

from Consolidated_Reg_Vis_Model import Leaf,Node,SoftDecisionTree
from matplotlib import pylab as plt
import torch
import numpy as np
import seaborn as sns
from custom_subdataloader import custom_index_dataloader
from create_pref_demo import create_training_data, cre_traj


def get_node_dist(node_names, node_prob):
    torch.set_printoptions(precision=10)
    count_prob_dist = {}
    substring="Node"
    for k in node_names.values():
        pointer=list(node_names.keys())[list(node_names.values()).index(k)]
        if substring in str(pointer):
                for ((r,c),_) in node_prob.keys():
                    count_prob_dist[((r,c),k)]=node_prob[((r,c),pointer)]
    return count_prob_dist


def gen_one_hot(r,c):
    all_zeros = np.zeros((1,28, 28))
    masked=all_zeros
    masked=masked.reshape(28,28)
    masked[r,c]=1
    masked=np.expand_dims(masked,axis=0)
    masked=torch.from_numpy(masked).float()
    return masked

def nodes_prob_dist(network):

    all_zeros = np.zeros((1,28,28))
    node_name, node_prob_off = network.vis_evaluate_tree((torch.from_numpy(all_zeros).float()), 0, 0)
    count_prob_dist_off = get_node_dist(node_name, node_prob_off)
    for r in range(28):
        for c in range(28):
            if r==0 and c==0:
                pass
            else:
                one_hot = gen_one_hot(r,c)
                node_name,node_prob_on= network.vis_evaluate_tree(one_hot,r,c)
                count_prob_dist_on=get_node_dist(node_name,node_prob_on)
    return count_prob_dist_on,count_prob_dist_off

def gen_heat_map(count_prob_dist_on,count_prob_dist_off,node):
    prob_node={}
    for ((r,c),node_count) in count_prob_dist_on:
        if node_count==node:
            prob_node[(r,c)]=count_prob_dist_on[((r,c),node_count)]

    for ((r,c),node_count) in count_prob_dist_off:
        if node_count == node:
            prob_off=count_prob_dist_off[((r,c),node_count)]
    heat_map = np.zeros((28, 28))
    for (r,c) in prob_node:
        prob_on = prob_node[(r, c)]
        heat_map[r, c] = prob_on - prob_off
    return heat_map


def sep_nodes_leaves(node_names,node_prob,get_Q):
    non_leaf_node_prob_dist={}
    leaf_node_Q={}
    node_pr=dict(node_prob)
    node_pr_keys=[key for key in node_pr.keys()]

    for node in node_names.keys():
        index=node_names[node]
        if node in node_pr_keys:
            pr_index = node_prob[node]
            non_leaf_node_prob_dist[index] = pr_index
        elif node in get_Q.keys():
            leaf_node_Q[index]=get_Q[node]
    return non_leaf_node_prob_dist, leaf_node_Q

def plot_histogram(leaf_nodes_Q,class_reward_vector,save_leaf_fig_dir):
    class_reward=torch.tensor(class_reward_vector).reshape(1,2)
    for l in leaf_nodes_Q:
        y=(leaf_nodes_Q[l].squeeze(dim=1)).cpu().detach().numpy()
        x=np.array([str(i) for i in class_reward_vector])
        plt.bar(x, y)
        plt.ylabel("Probability",fontsize=22)
        plt.xlabel("Reward Outputs",fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        leaf_class=torch.inner(class_reward.double(),leaf_nodes_Q[l].cpu().detach().reshape(1,2))
        print(f"For leaf {l} Q dot class reward vector: {leaf_class.item():.5f}")
        plt.title(f"Q dot class reward vector: {leaf_class.item():.5f}")
        plt.tight_layout()
        plt.savefig(save_leaf_fig_dir + "Leaf vis of DDT  for leaf node {l}.png".format(l=l),bbox_inches='tight')
        plt.show()

if __name__=="__main__":
    input_dim=28*28
    depth = 2

    if depth == 4:
        num_nodes = 15
        class_reward_vector = [0, 9]
    elif depth == 2:
        num_nodes = 3
        class_reward_vector = [0, 3]



    with torch.no_grad():

        tree = SoftDecisionTree(depth, input_dim,class_reward_vector)
        tree = torch.load('Trained_Reward_Models/mnist0-3/DDT/IL/R1-WAY-BETTER-seed0_withES_100')
        tree.eval()
        prob_dist_on,prob_dist_off=nodes_prob_dist(tree)
        base_vis_dir = 'Vis/'
        if not os.path.exists(base_vis_dir):
            os.makedirs(base_vis_dir)
        node_vis_dir=base_vis_dir +'/Node_Activation_HeatMaps/'
        if not os.path.exists(node_vis_dir):
            os.makedirs(node_vis_dir)

        leaf_vis_dir=base_vis_dir +'/Leaf_vis/'
        if not os.path.exists(leaf_vis_dir):
            os.makedirs(leaf_vis_dir)

        for node in range(num_nodes):
            h_node = gen_heat_map(prob_dist_on, prob_dist_off, node)
            ax=sns.heatmap(h_node,cbar_kws={'shrink': 0.7})
            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

            plt.savefig(node_vis_dir+"Node Activation Heatmap of DDT of node {node}.png".format(node=node))
            plt.show()

        input_data = custom_index_dataloader(batch_size=1, index_to_use=[0,1,2,3], shuffle=False)
        x, y,_ = cre_traj(1, input_data)




        node_name, node_pr, Q = tree.leaf_vis_tree(x[0])

        nl, l = sep_nodes_leaves(node_name, node_pr, Q)
        plot_histogram(l,class_reward_vector,leaf_vis_dir)
