__author__ = "akansha_kalra"
import os
from collections import defaultdict
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms , datasets
from trace_utils import Leaf,Node,SoftDecisionTree , get_node_dist
import random
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--save_fig_dir',default="Vis/", help="where to save traces, if None then not saved")

args = parser.parse_args()
base_fig_dir=args.save_fig_dir
if not os.path.exists(base_fig_dir):
    os.makedirs(base_fig_dir)
save_fig_dir=base_fig_dir+'Traces/'
if not os.path.exists(save_fig_dir):
    os.makedirs(save_fig_dir)

seed=1001
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

'''Main arguments'''
num_samples=5

def plot_trace(states_to_plot,node,routing_prob_list=None,save_fig_dir=None):
    f, axarr = plt.subplots(1, len(states_to_plot))
    for i  in range(len(states_to_plot)) :
        state=states_to_plot[i]
        axarr[i].imshow(state.numpy().squeeze(), cmap='gray_r')
        axarr[i].set_axis_off()
    plt.tight_layout()
    if save_fig_dir is not None:
        plt.savefig( save_fig_dir+f"Trace for Node{node} with {len(states_to_plot)} digits using seed{seed}")
    plt.show()


def custom_index_dataset(dataset,index_to_use):

    for i in range(len(index_to_use)):
        if i==0:
            idx = dataset.train_labels==index_to_use[i]
        else:
            idx += dataset.train_labels==index_to_use[i]

    train_labels = dataset.train_labels[idx]
    train_data = dataset.train_data[idx]
    dset_train = torch.utils.data.dataset.Subset(dataset, np.where(idx==1)[0])

    print(len(train_labels))
    return dset_train


train_dataset = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
training_dataset=custom_index_dataset(train_dataset,[0,1,2,3])

input_dim=28*28
class_reward_vector=[0,3]
print(class_reward_vector)
tree_depth=2

tree = SoftDecisionTree(int(float(tree_depth)),input_dim, class_reward_vector)
tree=torch.load('Trained_Reward_Models/mnist0-3/DDT/IL/R1-WAY-BETTER-seed0_withES_100')

prob_Node0_dict=defaultdict()
prob_Node1_dict=defaultdict()
prob_Node2_dict=defaultdict()



with torch.no_grad():
    for i in range(len(training_dataset)):
            state,label=training_dataset[i]
            flatten_state=state.reshape(1,-1)
            node_name,node_prob=tree.vis_evaluate_tree(flatten_state)
            probdict = get_node_dist(node_name, node_prob)
            prob_Node0_dict[probdict[0][0]] = state

            if probdict[0][0] > 0.5:
                prob_Node1_dict[(probdict[1])[0]] = state

            elif probdict[0][0]< 0.5:
                prob_Node2_dict[probdict[2][0]] = state



'''Node 1 sorted according to routing probabilities'''
ls_keys1=list(prob_Node1_dict.keys())
sorted_by_routing_prob_Node1 = sorted(ls_keys1, reverse=True)

ls_keys2=list(prob_Node2_dict.keys())
sorted_by_routing_prob_Node2 = sorted(ls_keys2,reverse=True)


ls_keys=list(prob_Node0_dict.keys())
sorted_by_routing_prob = sorted(ls_keys,reverse=True)


def select_every_T(prob_list,start_index,num_samples):
    sampling_timestep=len(prob_list)//num_samples
    sampled_prob_list=[]
    for i in range(start_index,len(prob_list),sampling_timestep):
        curr_prob=prob_list[i]
        sampled_prob_list.append(curr_prob)

    assert len(sampled_prob_list)>=num_samples
    return sampled_prob_list

def get_states(prob_state_dict,prob_list):
    return [prob_state_dict[sampledProb] for sampledProb in prob_list]

def autolabel(rects, ax):
    for rect in rects:
        x = rect.get_x() + rect.get_width()/2.
        y = rect.get_height()
        ax.annotate("{}".format(y), (x,y), xytext=(0,5), textcoords="offset points",
                    ha='center', va='bottom')




'''Node 0 '''
node0=0
Node0_prob_list=select_every_T(ls_keys,start_index=100,num_samples=5)
Node0_prob_list=sorted(Node0_prob_list,reverse=True)
print(f"No of digits in trace of Node 0 is {len(Node0_prob_list)}")
Node0_sampled_states=get_states(prob_Node0_dict,Node0_prob_list)
print(Node0_prob_list)
plot_trace(Node0_sampled_states,node0,save_fig_dir=save_fig_dir)
x_values0=[i for i in range(len(Node0_prob_list))]
plt.figure(figsize=(4,2))
a0=plt.bar(x_values0,Node0_prob_list,width=0.2)

plt.ylabel('Routing Prob',fontsize=16)
plt.ylim(0,1)
plt.yticks(fontsize=12)
plt.xticks([])
plt.tight_layout()
plt.savefig(save_fig_dir+ f"Final Routing Prob Barplot for Node{node0}")
plt.show()

print("___"*50)






node1=1
Node1_prob_list=select_every_T(ls_keys1,start_index=700,num_samples=5)
Node1_prob_list=sorted(Node1_prob_list,reverse=True)
print(f"No of digits in trace of Node 1 is {len(Node1_prob_list)}")
Node1_sampled_states=get_states(prob_Node1_dict,Node1_prob_list)
print(Node1_prob_list)
plot_trace(Node1_sampled_states,node=node1,save_fig_dir=save_fig_dir)
x_values1=[i for i in range(len(Node1_prob_list))]
plt.figure(figsize=(4,2))
a=plt.bar(x_values1,Node1_prob_list,width=0.2)
plt.ylabel('Routing Prob',fontsize=16)
plt.ylim(0,1)
plt.yticks(fontsize=12)
plt.xticks([])
plt.tight_layout()
plt.savefig( save_fig_dir+ f"Final Routing Prob Barplot for Node{node1}")
plt.show()


'''Node 2'''
node2=2
Node2_prob_list=select_every_T(ls_keys2,start_index=0,num_samples=4)
Node2_prob_list=sorted(Node2_prob_list,reverse=True)
print(f"No of digits in trace of Node 2 is {len(Node2_prob_list)}")
Node2_sampled_states=get_states(prob_Node2_dict,Node2_prob_list)
print(Node2_prob_list)
plot_trace(Node2_sampled_states,node2,save_fig_dir=save_fig_dir)
x_values2=[i for i in range(len(Node1_prob_list))]
plt.figure(figsize=(4,2))
a2=plt.bar(x_values2,Node2_prob_list,width=0.2)
plt.ylabel('Routing Prob',fontsize=16)
plt.yticks(fontsize=12)
plt.xticks([])
plt.tight_layout()
plt.savefig(save_fig_dir+ f"Final Routing Prob Barplot for Node{node2}")
plt.show()




