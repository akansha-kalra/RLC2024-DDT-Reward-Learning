__author__ = "akansha_kalra"
import matplotlib.pyplot as plt
import torch
import numpy as  np
# from DDT import Leaf, Node,SoftDecisionTree
from Learnt_DDT_model import Leaf, Node,SoftDecisionTree,get_node_dist_new
import seaborn as sns
import os


def sep_nodes_leaves(node_names,node_prob,get_Q):
    # print(node_names)
    non_leaf_node_prob_dist={}
    leaf_node_Q={}
    node_pr=dict(node_prob)
    # print("new node prob dict", node_pr)
    node_pr_keys=[key for key in node_pr.keys()]
    # print(node_pr_keys)
    for node in node_names.keys():
        # print(node)

        index=node_names[node]
        if node in node_pr_keys:
            # print("here")
            pr_index = node_prob[node]
            non_leaf_node_prob_dist[index] = pr_index
        elif node in get_Q.keys():
            leaf_node_Q[index]=get_Q[node]

    print("non-leaf node have routing probability :", non_leaf_node_prob_dist)
    print("leaf node class distribiutions are:", leaf_node_Q)
    return non_leaf_node_prob_dist, leaf_node_Q

def plot_histogram(leaf_nodes_Q,save_leaf_dir):
    # prob = node_prob[0]
    plt.rcParams["figure.figsize"] = (4, 4)
    for k in leaf_nodes_Q.keys():
        b = leaf_nodes_Q[k].cpu().detach().numpy()
        b = b[0]
        x = np.array(['0', '1'])
        plt.bar(x, b)
        # l_prime = k - 2
        c=np.argmax(b)
        plt.title('Max class of leaf {k} is {c}'.format(k=k, c=c), fontsize=10)
        plt.ylabel("Probability ",fontsize=24)
        plt.xlabel("Reward Outputs",fontsize=24)
        plt.yticks(fontsize=20)
        plt.xticks(fontsize=20)
        # plt.xlabel("classes where true label of input img is {true_label}".format(true_label=true_label))
        # plt.text(0.5, 0.5, r'root node routing prob: ' + str(prob))
        plt.tight_layout()
        if save_leaf_dir is not None:
            plt.savefig(save_leaf_dir+"leaf{k}.png".format(k=k))

        plt.show()






if __name__=="__main__":
    input_dim = 1 * 2
    depth = 2
    class_reward_vector = [0, 1]
    nb_classes = len(class_reward_vector)
    with torch.no_grad():
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector,seed=0)
        tree = torch.load('Trained_Reward_Models/DDT/saved_models/CP-1_50')
        print(tree)
        # tree = tree.to(device)
        tree.eval()

        print(
            f"----------------------------LOADING TRAINED TREE ----------------------------------------------------------------------")

        print("Node distribution", list(tree.module_list.named_parameters()))
        print("Leaf distribution", [param for param in tree.param_list])

        center_allzero = torch.from_numpy(np.asarray((0, 0))).float().reshape(1, 2)

        _,node_name, node_p,get_Q = tree.fwd_input_hm(center_allzero)
        save_leaf_dir=None

        '''uncomment to save'''
        # current_directory = os.getcwd()
        # save_leaf_dir=  current_directory + '/Reward_Models/DDT/Vis/CP-1/'

        n,l=sep_nodes_leaves(node_name,node_p,get_Q)
        print(l)
        plot_histogram(l,save_leaf_dir)
