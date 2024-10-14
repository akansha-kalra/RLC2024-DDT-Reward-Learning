import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import create_training_data ,pairwise_demons_dataloader,EarlyStopping, AtariDataset
from collections import defaultdict
import argparse
import random
import gc

parser=argparse.ArgumentParser(description=None)
parser.add_argument('--seed',default=0,help="random seed for experiments")
parser.add_argument('--demos_path', default="datasets/Breakout_demonstrations_new", help="path where demonstrations are stored")
parser.add_argument('--pth', default=".", help="path where tensorboard events are stored")
parser.add_argument('--save_model_dir',default=".",help="where to save trained model")
parser.add_argument('--exp_no',default="XX",help="which experiment number are you on")
parser.add_argument('--lr',default=0.0001,help="lr for experiments")
parser.add_argument('--num_epochs',default=10,help="no of epochs to train tree")
parser.add_argument('--tr_index',default=3000,help=" training_index")
parser.add_argument('--batch_size',default=10,help="batch size Bx2x25x84x84x4")

args=parser.parse_args()
seed=int(args.seed)
exp_no=args.exp_no
pth=args.pth + exp_no
demos_path=args.demos_path
save_model_dir=args.save_model_dir
lr=args.lr
num_epochs=args.num_epochs
tr_index=args.tr_index
batch_size=int(args.batch_size)


seed=seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")



writer = SummaryWriter(pth)
if not os.path.exists(save_model_dir):
    print(' Creating Project : ' + save_model_dir)
    os.makedirs(save_model_dir)

class Leaf():
    def __init__(self, nb_classes):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


        if depth > 0:
            self.children = self.build_children(depth,module_list)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth,module_list):
        return [Node(depth - 1, self.nb_classes,module_list),
                Node(depth - 1, self.nb_classes,module_list)]
    def forward(self, x):
        x=x.permute(0,3,1,2)
        out = F.leaky_relu(self.conv1(x))
        if out.dim()==4:
            input_linear=out.reshape((out.size(dim=0),-1))
        out = self.fc1(input_linear)
        out = torch.sigmoid(out)
        return out

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


    def forward(self, current_node, inputs,path_prob):
        if isinstance(current_node, Leaf):
            current_node.path_prob = path_prob
            #             print(f"Current node: {current_node}  has path probability: {path_prob}")
            return  # end of recursion at a leaf


        prob = current_node.forward(inputs)
        # print(f"prob of the current node is {prob} and if needed current node is {current_node}")

        # Left Children -> prob = activation
        self.forward(current_node.children[0], inputs, prob * path_prob)
        # # Right children -> prob = 1 - activation
        self.forward(current_node.children[1], inputs, (1 - prob) * path_prob)

    def get_loss(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).double().to(self.device)
        loss_tree = 0
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob), dim=1)
            loss_tree += loss
        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree


def train(ddt,tr_pref_trajs,tr_pref_labels, no_epochs, optimizer,val_pref_trajs,val_pref_labels,batch_size):
    early_stopping=EarlyStopping(patience=100,min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    ddt=ddt.to(device)
    training_data = list(zip(tr_pref_trajs,tr_pref_labels))
    val_data=list(zip(val_pref_trajs, val_pref_labels))
    for epoch in range(no_epochs):
        np.random.shuffle(training_data)
        np.random.shuffle(val_data)
        print(epoch)
        acc_counter = 0
        losses = []
        iterable_pref_trajs, iterable_pref_labels = zip(*training_data)
        val_iterable_pref_trajs, val_iterable_pref_labels = zip(*val_data)
        for i in range(0,len(tr_pref_labels),batch_size):
        # for pairwise_demo_batch,label in tr_pref_trajs:
            optimizer.zero_grad()
            label=iterable_pref_labels[i:i+batch_size]
            pref_trajs=iterable_pref_trajs[i:i+batch_size]

            pref_label = torch.from_numpy(np.array(label)).reshape(batch_size,)
            pref_label = pref_label.to(device)

            torch_pref_trajs=torch.from_numpy(np.array(pref_trajs)).reshape(batch_size,2,25,84,84,4).float().to(device)
            torch_pref_train=torch_pref_trajs.view(batch_size*2*25,84,84,4)
            ones = torch.ones(len(torch_pref_train),1).to(device)
            ddt.forward(ddt.root,torch_pref_train, ones)
            loss_tree = ddt.get_loss()
            loss_tree = loss_tree.reshape(batch_size,2, 25)
            loss_tree_traj = torch.sum(loss_tree, dim=2)
            pred_label = torch.argmax(loss_tree_traj, dim=1)

            acc_counter += torch.sum((pred_label == pref_label).float())
            final_loss = loss_criterion(loss_tree_traj, pref_label)
            # print(final_loss.item())
            losses.append(final_loss.detach().cpu().numpy())

            final_loss.backward()
            optimizer.step()

            train_size = len(tr_pref_labels)
        if epoch==0:
            print("Training dataset size is", train_size)

        training_loss_per_epoch = np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (train_size) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch', training_loss_per_epoch, epoch)
        writer.add_scalar('Training Accuracy per epoch', training_acc_per_epoch, epoch)

        with torch.no_grad():
            val_acc_counter = 0
            val_losses = []
            for v in range(0,len(val_pref_labels),batch_size):
                val_label = val_iterable_pref_labels[v:v + batch_size]
                val_pref_trajs = val_iterable_pref_trajs[v:v+ batch_size]

                val_pref_label = torch.from_numpy(np.array(val_label)).reshape(batch_size,)
                val_pref_label = val_pref_label.to(device)

                val_torch_pref_trajs = torch.from_numpy(np.array(val_pref_trajs)).reshape(batch_size, 2, 25, 84, 84, 4).float().to(device)
                val_torch_pref_train = val_torch_pref_trajs.view(batch_size * 2 * 25, 84, 84, 4)
                val_ones = torch.ones((len(val_torch_pref_train), 1)).to(device)
                ddt.forward(ddt.root, val_torch_pref_train, val_ones)
                val_loss_tree = ddt.get_loss()
                val_loss_tree = val_loss_tree.reshape(batch_size,2,25)
                val_loss_tree_traj = torch.sum(val_loss_tree, dim=2)
                val_pred_label = torch.argmax(val_loss_tree_traj, dim=1)

                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)
                # print(final_loss.item())
                val_losses.append(val_final_loss.detach().cpu().numpy())
                val_size = len(val_pref_labels)

            if epoch == 0:
                print("Val dataset size is", val_size)

            val_loss_per_epoch = np.mean(val_losses)
            print("Validation Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (val_size) * 100
            print(" Validation Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Validation Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Validation Accuracy per epoch', val_acc_per_epoch, epoch)
            early_stopping(val_loss_per_epoch)

        if early_stopping.early_stop:
            print("We are at epoch:", epoch)
            # print(
            #     f"total no of iterations are {no_epochs * len(training_labels)} and len of training data is {len(training_labels)}  and no of epochs are {epoch} ")

            torch.save(ddt, save_model_dir + exp_no + "_" + str(epoch))
            break
    if early_stopping.early_stop:
        pass
    else:
        torch.save(ddt, save_model_dir + exp_no + "_" + str(no_epochs))
        print(f"no of epochs are {no_epochs}")
    #



if __name__=="__main__":
    depth = 2
    class_reward_vector=[0,1]
    demons = torch.load(demos_path)

    min_snippet_length = 50  # min length of trajectory for training comparison
    maximum_snippet_length = 51
    num_snippets = 6000

    demo_lengths = [len(d) for d in demons]
    print("demo lengths", demo_lengths)
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    print("max snippet length", max_snippet_length)

    training_obs, training_labels = create_training_data(demons, num_snippets, min_snippet_length, max_snippet_length,seed=0)
    print(f"total number of pairwise trajectories {len(training_labels)}")

    del demons
    demons=None
    gc.collect()
    torch.cuda.empty_cache()
    tr_index=int(tr_index)

    tr_pref_trajs=training_obs[:tr_index]
    tr_pref_labels=training_labels[:tr_index]
    val_pref_trajs=training_obs[tr_index:]
    val_pref_labels=training_labels[tr_index:]


    tree = SoftDecisionTree(int(depth), class_reward_vector)
    print(tree)

    optimizer = optim.Adam(tree.parameters(), lr=float(lr), weight_decay=0.00)
    num_epochs=int(num_epochs)
    train(tree,tr_pref_trajs,tr_pref_labels,int(num_epochs),optimizer,val_pref_trajs,val_pref_labels,batch_size)