__author__ = "akansha_kalra"
import numpy as np
import torch
import matplotlib.pylab as plt
from DDT_Model import Leaf, Node,SoftDecisionTree
from collections import defaultdict,Counter
import matplotlib.gridspec as gridspec
import argparse
import os
# import matplotlib as mpl
# mpl.use('Qt5Agg')

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--env_name', default="breakout", help="Name of the Atari Env")
parser.add_argument('--DDT_with_penalty', default=False, type=bool, help="traces for DDT trained-with or without penalty")
parser.add_argument('--save_fig_dir',default="Vis/", help="where to save traces, if None then not saved")

args = parser.parse_args()
env_name=args.env_name
DDT_with_penalty=args.DDT_with_penalty
save_fig_dir=args.save_fig_dir
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

def get_node_dist(node_names, node_prob):
    torch.set_printoptions(precision=10)
    count_prob_dist = {}
    node_pr_keys = [key for key in node_prob.keys()]
    # print(node_pr_keys)
    for node in node_names.keys():
        index = node_names[node]
        if node in node_pr_keys:
            # print("here")
            pr_index = node_prob[node]
            count_prob_dist[index]=pr_index

    return count_prob_dist

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

if env_name == 'breakout':
    if DDT_with_penalty:
        Leaf_path='Trained_Reward_DDT/Breakout_PenaltySeed2_LEAF.npy'
        Node_path='Trained_Reward_DDT/Breakout_PenaltySeed2_NODE.npy'
    else:
        Leaf_path = 'Trained_Reward_DDT/Breakout_Without_PenaltySeed0_LEAF.npy'
        Node_path='Trained_Reward_DDT/Breakout_Without_PenaltySeed0_NODE.npy'
elif env_name == 'beamrider':
    if DDT_with_penalty:
        Leaf_path='Trained_Reward_DDT/BeamRider_PenaltySeed1_LEAF.npy'
        Node_path='Trained_Reward_DDT/BeamRider_PenaltySeed1_NODE.npy'
    else:
        Leaf_path='Trained_Reward_DDT/BeamRider_Without_PenaltySeed0_LEAF.npy'
        Node_path='Trained_Reward_DDT/BeamRider_Without_PenaltySeed0_NODE.npy'


trained_leaf_params = np.load(Leaf_path, allow_pickle=True)
trained_leaf_params = trained_leaf_params.tolist()
print(f"Trained leaf params {trained_leaf_params}")
trained_node_params = np.load(Node_path, allow_pickle=True)
trained_node_params = trained_node_params.tolist()



tree_depth = 2
class_reward_vector = [0, 1]
tree = SoftDecisionTree(int(float(tree_depth)), class_reward_vector)
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




min_reward = 100000
max_reward = -100000

state_counter=0
traj_counter=0

prob_Node0_dict=defaultdict(list)
prob_Node1_dict=defaultdict(list)
prob_Node2_dict=defaultdict(list)
'''SPECIFY CHOICE OF TRACE'''
choice=0
# save_fig_dir='/home/ak/Documents/AtariRL_SD/BeamRider_Penalty_Seed1_Vis/full_len/'

# save_fig_dir='/home/ak/Documents/AtariRL_SD/Breakout_Penalty_Seed2_Vis/'
'''Dicts for explaining the single input in a trace'''
explain_prob_Node0_dict=defaultdict()
explain_prob_Node1_dict=defaultdict()
explain_prob_Node2_dict=defaultdict()
explain_state_tracker=defaultdict()

with torch.no_grad():
    for d in demons:
        # print(cnt)
        s_cnt_in_single_traj =0
        for i,s in enumerate(d[2:-1]):
            # print(s)
            # print(s.shape)
            explain_counter=(traj_counter,s_cnt_in_single_traj)
            node_name,node_prob=tree.vis_evaluate_tree(torch.from_numpy(s).unsqueeze(0))
            probdict = get_node_dist(node_name, node_prob)
            # print(node_name)
            # print(node_prob)
            # print("PROB DICT", probdict)
            #
            prob_Node0_dict[(state_counter,probdict[0].item())]=s
            explain_prob_Node0_dict[traj_counter,s_cnt_in_single_traj]=probdict[0].item()
            explain_state_tracker[traj_counter,s_cnt_in_single_traj]=s

            if probdict[0].item() >0.5:
                prob_Node1_dict[(state_counter, probdict[1].item())] = s
                explain_prob_Node1_dict[traj_counter,s_cnt_in_single_traj]=probdict[1].item()
            elif probdict[0].item() <0.5:
                prob_Node2_dict[(state_counter,probdict[2].item())]=s
                explain_prob_Node2_dict[traj_counter, s_cnt_in_single_traj] = probdict[2].item()


            s_cnt_in_single_traj+=1
            state_counter+=1
        traj_counter+=1
    # print(prob_Node0_dict.keys())

    print("State counter", state_counter)
    ls_keys=list(prob_Node0_dict.keys())
    sorted_by_routing_prob = sorted(ls_keys, key=lambda tup: tup[1], reverse=True)

    # print(sorted_by_routing_prob)


    count_dict=Counter(elem[1] for elem in sorted_by_routing_prob)

    ls_keys1 = list(prob_Node1_dict.keys())
    sorted_by_routing_prob_Node1 = sorted(ls_keys1, key=lambda tup: tup[1], reverse=True)
    count_dict1 = Counter(elem[1] for elem in sorted_by_routing_prob_Node1)
    # print(count_dict1)

    ls_keys2 = list(prob_Node2_dict.keys())
    sorted_by_routing_prob_Node2 = sorted(ls_keys2, key=lambda tup: tup[1], reverse=True)
    count_dict2= Counter(elem[1] for elem in sorted_by_routing_prob_Node2)

    print(f"Len of node 0: {len(ls_keys)}, node 1: {len(ls_keys1)}, node 2: {len(ls_keys2)}")
    Node_1_routing_probs_ls=list(explain_prob_Node1_dict.values())
    print(f"Routing prob in node 1 has max value {max(Node_1_routing_probs_ls)}) and min {min(Node_1_routing_probs_ls)}")

    Node_2_routing_probs_ls = list(explain_prob_Node2_dict.values())
    print(f"Routing prob in node 2 has max value {max(Node_2_routing_probs_ls)}) and min {min(Node_2_routing_probs_ls)}")


    prob_1_dict = {}
    prob_1 = []
    prob_75 = []
    prob_75_dict = {}
    prob_50 = []
    prob_50_dict = {}
    prob_25 = []
    prob_25_dict = {}
    prob_0 = []
    prob_0_dict = {}
    for tk in sorted_by_routing_prob:
        _, k = tk
        if k > 0.9 and k <= 1:
            prob_1_dict[tk] = prob_Node0_dict[tk]
            prob_1.append(prob_Node0_dict[tk])
        if k <= 0.7599 and k >= 0.7488:
            prob_75_dict[tk] = prob_Node0_dict[tk]
            prob_75.append(prob_Node0_dict[tk])
        if k <= 0.549 and k >= 0.459:
            prob_50_dict[tk] = prob_Node0_dict[tk]
            prob_50.append(prob_Node0_dict[tk])
        if k <= 0.255 and k >= 0.2459:
            prob_25_dict[tk] = prob_Node0_dict[tk]
            prob_25.append(prob_Node0_dict[tk])
        if k <= 0.05 and k >= 0:
            prob_0_dict[tk] = prob_Node0_dict[tk]
            prob_0.append(prob_Node0_dict[tk])

    print(len(prob_0_dict))
    print("-------------------------------------------")
    print(len(prob_75_dict))

    # np.random.seed(0)

    # img_1=np.random.choice(np.array(prob_1))
    # img_75=np.random.choice(np.array(prob_75))
    # img_50=np.random.choice(np.array(prob_50))
    # img_25=np.random.choice(np.array(prob_25))
    # img_0=np.random.choice(np.array(prob_0))

    img_1 = prob_1[choice]
    img_75 = prob_75[choice]
    img_50 = prob_50[choice]
    img_25 = prob_25[choice]
    img_0 = prob_0[choice]

    img_to_plt=[img_1,img_75,img_50,img_25,img_0]
    fig = plt.figure(figsize=(20, 3))
    fig.patch.set_facecolor('xkcd:black')
    outer = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.1)
    for i in range(5):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
        row = 0
        col = 0
        maxCol = 4

        for ch in range(4):
            ax = plt.Subplot(fig, inner[row, col])
            ax.imshow(img_to_plt[i][:, :, ch], cmap='viridis',aspect='auto')
            ax.axis('off')
            # t = ax.text(0.5, 0.5, 'outer=%d\nrow=%d\ncol=%d' % (i, row, col))
            # ax.set_xticks([])
            # ax.set_yticks([])
            # t.set_ha('center')
            fig.add_subplot(ax)
            col += 1
            if col == maxCol:
                col = 0
                row += 1
# plt.tight_layout()
if save_fig_dir is not None:
    plt.savefig(save_fig_dir +f"choice{choice} Node0_InverseFERL of DDT {env_name} with penalty{DDT_with_penalty} ")
plt.show()


'''FOR NODE 1 '''


Node_1prob_1_dict={}
Node_1prob_1=[]
Node_1prob_75=[]
Node_1prob_75_dict={}
Node_1prob_50=[]
Node_1prob_50_dict = {}
Node_1prob_25=[]
Node_1prob_25_dict = {}
Node_1prob_0=[]
Node_1prob_0_dict = {}
for tk in sorted_by_routing_prob_Node1:
    _,k=tk
    if k >0.9 and k<=1:
        Node_1prob_1_dict[tk] =prob_Node1_dict[tk]
        Node_1prob_1.append(prob_Node1_dict[tk])
    if k <= 0.7599 and k>=0.7488:
        Node_1prob_75_dict[tk] = prob_Node1_dict[tk]
        Node_1prob_75.append(prob_Node1_dict[tk])
    if k<=0.549 and k>=0.459:
        Node_1prob_50_dict[tk] = prob_Node1_dict[tk]
        Node_1prob_50.append(prob_Node1_dict[tk])
    if k<=0.255 and k>=0.2459:
        Node_1prob_25_dict[tk] = prob_Node1_dict[tk]
        Node_1prob_25.append(prob_Node1_dict[tk])
    if k<=0.05 and k>=0:
        Node_1prob_0_dict[tk] = prob_Node1_dict[tk]
        Node_1prob_0.append(prob_Node1_dict[tk])

print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")

print("NODE 1")

print(len(Node_1prob_0_dict))
print("-------------------------------------------")
print(len(Node_1prob_75_dict))


# np.random.seed(0)

Node1_img_1 = Node_1prob_1[choice]
Node1_img_75= Node_1prob_75[choice]
Node1_img_50= Node_1prob_50[choice]
Node1_img_25= Node_1prob_25[choice]
Node1_img_0= Node_1prob_0[choice]

'''Uncomment for last trace and comment out the initial trace'''

# Node1_img_1 = Node_1prob_1[len(Node_1prob_1)-1]
# Node1_img_75= Node_1prob_75[len(Node_1prob_75)-1]
# Node1_img_50= Node_1prob_50[len(Node_1prob_50)-1]
# Node1_img_25= Node_1prob_25[len(Node_1prob_25)-1]
# Node1_img_0= Node_1prob_0[len(Node_1prob_0)-1]

Node1_img_to_plt=[Node1_img_1,Node1_img_75,Node1_img_50,Node1_img_25,Node1_img_0]
fig = plt.figure(figsize=(20, 3))
fig.patch.set_facecolor('xkcd:black')
outer = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.1)
for i in range(5):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    row = 0
    col = 0
    maxCol = 4

    for ch in range(4):
        ax = plt.Subplot(fig, inner[row, col])
        ax.imshow(Node1_img_to_plt[i][:, :, ch], cmap='viridis',aspect='auto')
        ax.axis('off')
        # t = ax.text(0.5, 0.5, 'outer=%d\nrow=%d\ncol=%d' % (i, row, col))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # t.set_ha('center')
        fig.add_subplot(ax)
        col += 1
        if col == maxCol:
            col = 0
            row += 1

plt.tight_layout()
if save_fig_dir is not None:
    plt.savefig(save_fig_dir +f"Choice {choice}Node1_InverseFERL of DDT {env_name} with penalty{DDT_with_penalty}")

plt.show()



'''FOR NODE 2 '''


Node_2prob_1_dict={}
Node_2prob_1=[]
Node_2prob_75=[]
Node_2prob_75_dict={}
Node_2prob_50=[]
Node_2prob_50_dict = {}
Node_2prob_25=[]
Node_2prob_25_dict = {}
Node_2prob_0=[]
Node_2prob_0_dict = {}
for tk in sorted_by_routing_prob_Node2:
    _,k=tk
    if k >0.9 and k<=1:
        Node_2prob_1_dict[tk] =prob_Node2_dict[tk]
        Node_2prob_1.append(prob_Node2_dict[tk])
    if k <= 0.7599 and k>=0.7488:
        Node_2prob_75_dict[tk] = prob_Node2_dict[tk]
        Node_2prob_75.append(prob_Node2_dict[tk])
    if k<=0.549 and k>=0.459:
        Node_2prob_50_dict[tk] = prob_Node2_dict[tk]
        Node_2prob_50.append(prob_Node2_dict[tk])
    if k<=0.255 and k>=0.2459:
        Node_2prob_25_dict[tk] = prob_Node2_dict[tk]
        Node_2prob_25.append(prob_Node2_dict[tk])
    if k<=0.05 and k>=0:
        Node_2prob_0_dict[tk] = prob_Node2_dict[tk]
        Node_2prob_0.append(prob_Node2_dict[tk])

print("-------------------------------------------")
print("-------------------------------------------")
print("-------------------------------------------")

print("NODE 2")
print(len(Node_2prob_0_dict))
print("-------------------------------------------")
print(len(Node_2prob_75_dict))




Node2_img_1 = Node_2prob_1[choice]
Node2_img_75= Node_2prob_75[choice]
Node2_img_50= Node_2prob_50[choice]
Node2_img_25= Node_2prob_25[choice]
Node2_img_0= Node_2prob_0[choice]

'''Uncomment for last trace and comment out the initial trace'''

# Node2_img_1 = Node_2prob_1[len(Node_2prob_1)-1]
# Node2_img_75= Node_2prob_75[len(Node_2prob_75)-1]
# Node2_img_50= Node_2prob_50[len(Node_2prob_50)-1]
# Node2_img_25= Node_2prob_25[len(Node_2prob_25)-1]
# Node2_img_0= Node_2prob_0[len(Node_2prob_0)-1]

Node2_img_to_plt=[Node2_img_1,Node2_img_75,Node2_img_50,Node2_img_25,Node2_img_0]
fig = plt.figure(figsize=(20, 3))
fig.patch.set_facecolor('xkcd:black')
outer = gridspec.GridSpec(1, 5, wspace=0.1, hspace=0.1)
for i in range(5):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[i], wspace=0.1, hspace=0.1)
    row = 0
    col = 0
    maxCol = 4

    for ch in range(4):
        ax = plt.Subplot(fig, inner[row, col])
        ax.imshow(Node2_img_to_plt[i][:, :, ch], cmap='viridis',aspect='auto')
        ax.axis('off')
        # t = ax.text(0.5, 0.5, 'outer=%d\nrow=%d\ncol=%d' % (i, row, col))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # t.set_ha('center')
        fig.add_subplot(ax)
        col += 1
        if col == maxCol:
            col = 0
            row += 1
plt.tight_layout()
plt.savefig(save_fig_dir +f"Choice{choice} Node2_InverseFERL of DDT {env_name} with penalty{DDT_with_penalty}")
plt.show()


