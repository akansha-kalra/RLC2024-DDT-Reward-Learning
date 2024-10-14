__author__ = "akansha_kalra"
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt
# import seaborn as sns
# import seaborn_image as isns
from DDT_Model import Leaf, Node, SoftDecisionTree
from collections import  defaultdict
import argparse
import gc

gc.collect()
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--DDT_with_penalty', default=False, type=bool, help="traces for DDT trained-with or without penalty")
parser.add_argument('--save_fig_dir',default="Vis/", help="where to save traces, if None then not saved")

args = parser.parse_args()
env_name=args.env_name
DDT_with_penalty=args.DDT_with_penalty
leaf_save_fig_dir=args.save_fig_dir
if not os.path.exists(leaf_save_fig_dir):
    os.makedirs(leaf_save_fig_dir)

'''Choose whether BeamRider or Breakout'''
assert env_name is not None
if env_name == 'breakout':
    demons = torch.load('datasets/Breakout_demonstrations_new')
elif env_name == 'beamrider':
    demons = torch.load('datasets/BeamRider_demonstrations_new')

min_snippet_length = 50  # min length of trajectory for training comparison
maximum_snippet_length = 51
num_snippets = 6000

demo_lengths = [len(d) for d in demons]
print("demo lengths", demo_lengths)

'''Make sure seed matches the seed params you're loading'''

if env_name == 'breakout':
    if DDT_with_penalty:
        seed = 2
        Leaf_path='Trained_Reward_DDT/Breakout_PenaltySeed2_LEAF.npy'
        Node_path='Trained_Reward_DDT/Breakout_PenaltySeed2_NODE.npy'
    else:
        seed = 0
        Leaf_path = 'Trained_Reward_DDT/Breakout_Without_PenaltySeed0_LEAF.npy'
        Node_path='Trained_Reward_DDT/Breakout_Without_PenaltySeed0_NODE.npy'
elif env_name == 'beamrider':
    if DDT_with_penalty:
        seed = 1
        Leaf_path='Trained_Reward_DDT/BeamRider_PenaltySeed1_LEAF.npy'
        Node_path='Trained_Reward_DDT/BeamRider_PenaltySeed1_NODE.npy'
    else:
        seed = 0
        Leaf_path='Trained_Reward_DDT/BeamRider_Without_PenaltySeed0_LEAF.npy'
        Node_path='Trained_Reward_DDT/BeamRider_Without_PenaltySeed0_NODE.npy'


trained_leaf_params = np.load(Leaf_path, allow_pickle=True)
trained_leaf_params = trained_leaf_params.tolist()
# print(f"Trained leaf params {trained_leaf_params}")
trained_node_params = np.load(Node_path, allow_pickle=True)
trained_node_params = trained_node_params.tolist()


tree_depth = 2
class_reward_vector = [0, 1]

tree = SoftDecisionTree(int(float(tree_depth)), class_reward_vector)
# print([param for param in tree.param_list])

# print([leaf.distribution for leaf in tree.leaves])
'''loading weights'''
for leaf, leaf_distribution in trained_leaf_params:
    leaf_index = int(leaf)
    tree.leaves[leaf_index].distribution = leaf_distribution

for node, node_param in trained_node_params:
    node_num, node_param_type = node.split(".")
    node_num = int(node_num)
    if node_param_type == "weight":
        tree.module_list[node_num].weight = node_param
    else:
        tree.module_list[node_num].bias = node_param

print(tree)
# print([leaf.distribution for leaf in tree.leaves])
# print([param for param in tree.param_list])
# print("checking softmax")
softmax = nn.Softmax(dim=1)
# print([softmax(leaf.distribution.view(1,-1)) for leaf in tree.leaves])


s=demons[0][2:-1][0]
min_frame_torch=torch.from_numpy(s).double()
min_frame_torch_better=min_frame_torch.unsqueeze(0)
# max_frame_torch_better=max_frame_torch.unsqueeze(0)

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

def plot_histogram(leaf_nodes_Q,seed,leaf_save_fig_dir,env_name, DDT_with_penalty):
    class_reward=torch.tensor([0,1])
    class_reward=class_reward.reshape(1,2)
    for l in leaf_nodes_Q:
        y=(leaf_nodes_Q[l].squeeze(dim=1)).cpu().detach().numpy()
        x = np.array(['0','1'])
        plt.bar(x, y)
        plt.ylabel("Probability",fontsize=22)
        plt.xlabel("Reward Outputs",fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        leaf_class=torch.inner(class_reward.double(),leaf_nodes_Q[l].cpu().detach().reshape(1,2))

        plt.title(f"Q dot class reward vector: {leaf_class.item():.5f}")
        # plt.text(0.5, 0.5, r'root node routing prob: ' + str(prob))
        plt.tight_layout()
        if leaf_save_fig_dir is not None:
            plt.savefig(leaf_save_fig_dir + " DDT {env_name} penalty {DDT_with_penalty} seed {seed} and leaf{l}.png".format(env_name=env_name,DDT_with_penalty=DDT_with_penalty,seed=seed,l=l),bbox_inches='tight')
        plt.show()



node_name, node_pr, Q = tree.leaf_vis_tree(min_frame_torch_better)
# print(node_pr)
# print(Q)
nl, l = sep_nodes_leaves(node_name, node_pr, Q)
# print(l)
plot_histogram(l,seed=seed,leaf_save_fig_dir=leaf_save_fig_dir, env_name=env_name,DDT_with_penalty=DDT_with_penalty)