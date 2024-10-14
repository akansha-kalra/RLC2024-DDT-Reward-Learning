import random
import os
from torch.utils.data import TensorDataset,DataLoader
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torchvision import transforms , datasets
from custom_subdataloader import custom_index_dataloader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pylab as plt
import argparse



parser=argparse.ArgumentParser(description=None)
parser.add_argument('--seed',default=0,help="random seed for experiments")
parser.add_argument('--pth', default=".", help="path where tensorboard events are stored")
parser.add_argument('--save_model_dir',default=".",help="where to save trained model")
parser.add_argument('--exp_no',default="XX",help="which experiment number are you on")



args=parser.parse_args()
seed=args.seed
exp_no=args.exp_no
pth=args.pth + exp_no
save_model_dir=args.save_model_dir




torch.manual_seed(seed)
writer = SummaryWriter(pth)
if not os.path.exists(save_model_dir):
    print(' Creating Project : ' + save_model_dir)
    os.makedirs(save_model_dir)



def custom_index_dataset(dataset,index_to_use):

    for i in range(len(index_to_use)):
        # print(i)
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
training_dataset=custom_index_dataset(train_dataset,[0,1,2,3,4,5,6,7,8,9])


vald_dataset=datasets.MNIST(root = './', train=False, download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
validation_dataset= custom_index_dataset(vald_dataset,[0,1,2,3,4,5,6,7,8,9])

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
        print(f"total no of leaf nodes are {leaf_counter} for depth {self.depth}")
        print(f"total no of non-leaf nodes are {node_counter} for depth {self.depth}")

    def forward(self, current_node, inputs, path_prob):
        if isinstance(current_node, Leaf):
            current_node.path_prob = path_prob
            #             print(f"Current node: {current_node}  has path probability: {path_prob}")
            return  # end of recursion at a leaf

        # set params for penalty
        prob = current_node.forward(inputs)
        current_node.alpha = torch.sum((prob * path_prob),dim=1) / torch.sum(path_prob,dim=1)
        # print(f"prob of the current node is {prob} and if needed current node is {current_node}")

        # Left Children -> prob = activation
        self.forward(current_node.children[0], inputs, prob * path_prob)
        # Right children -> prob = 1 - activation
        self.forward(current_node.children[1], inputs, (1 - prob) * path_prob)


    def get_loss(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).double().to(self.device)
        class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).double().to(self.device)
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob),dim=1)
            loss_tree += loss
        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree



def train(traj_length,pairwise_batch_size,no_train_batches,training_dataset,num_epochs,ddt,optimizer, validation_dataset, no_val_batches):
    # early_stopping = EarlyStopping(patience=15, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    # reduction='none' gives batch of CE loss bt then we need to do manual reduction :https://discuss.pytorch.org/t/reduction-none-leads-to-different-computed-loss/129515
    ddt = ddt.to(device)

    for epoch in range(num_epochs):
        print(f"------------------Epoch:{epoch}---------")
        acc_counter = 0
        accuracies = []
        losses = []
        cum_loss = 0
        count = 0


        '''Generate trainining data for each epoch'''

        pairwise_trajs = []
        pairwise_labels = []
        for nb in range(no_train_batches):
            # print(f"current batch no {nb}")
            curr_batch_size = 0
            while curr_batch_size < pairwise_batch_size:
                idx = torch.randint(len(training_dataset), (2*traj_length,))
                # print(idx,idx.shape)
                traj1_gt_return = 0
                traj2_gt_return = 0
                traj1 = []
                traj2 = []
                for count,i in enumerate(idx):
                    img, l = training_dataset[i]
                    # print(l)
                    if count< traj_length:
                        traj1_gt_return+=l
                        traj1.append(img)
                    else:
                        traj2_gt_return+=l
                        traj2.append(img)
                # print(f"gt reward of traj1 and traj2 are {traj1_gt_return} and {traj2_gt_return} respectively")
                if traj1_gt_return != traj2_gt_return:
                    if traj1_gt_return > traj2_gt_return:
                        pref_label = 0
                    elif traj1_gt_return < traj2_gt_return:
                        pref_label = 1

                    t1 = torch.stack(traj1)
                    t2 = torch.stack(traj2)
                    traj_pair = torch.stack((t1, t2))

                    pairwise_trajs.append(traj_pair)
                    pairwise_labels.append(pref_label)
                    curr_batch_size+=1
            # print(pairwise_labels)

        training_pairwise_demos = torch.stack((pairwise_trajs)).reshape(-1,pairwise_batch_size,2,traj_length,1,28,28)

        training_labels = torch.tensor(pairwise_labels).reshape(no_train_batches,pairwise_batch_size,1)

        train_dataset = TensorDataset(training_pairwise_demos, training_labels)
        training_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        pair_traj, l = next(iter(training_dataloader))

        val_pairwise_trajs = []
        val_pairwise_labels = []
        for nb in range(no_val_batches):
            # print(f"current batch no {nb}")
            val_curr_batch_size = 0
            while val_curr_batch_size < pairwise_batch_size:
                val_idx = torch.randint(len(validation_dataset), (2 * traj_length,))
                # print(idx,idx.shape)
                val_traj1_gt_return = 0
                val_traj2_gt_return = 0
                val_traj1 = []
                val_traj2 = []
                for val_count, val_i in enumerate(val_idx):
                    val_img, val_l = validation_dataset[val_i]
                    # print(l)
                    if val_count < traj_length:
                        val_traj1_gt_return += val_l
                        val_traj1.append(val_img)
                    else:
                        val_traj2_gt_return += val_l
                        val_traj2.append(val_img)
                # print(f"gt reward of traj1 and traj2 are {traj1_gt_return} and {traj2_gt_return} respectively")
                if val_traj1_gt_return != val_traj2_gt_return:
                    if val_traj1_gt_return > val_traj2_gt_return:
                        val_pref_label = 0
                    elif val_traj1_gt_return < val_traj2_gt_return:
                        val_pref_label = 1

                    val_t1 = torch.stack(val_traj1)
                    val_t2 = torch.stack(val_traj2)
                    val_traj_pair = torch.stack((val_t1, val_t2))

                    val_pairwise_trajs.append(val_traj_pair)
                    val_pairwise_labels.append(val_pref_label)
                    val_curr_batch_size += 1
            # print(pairwise_labels)

        val_pairwise_demos = torch.stack((val_pairwise_trajs)).reshape(-1, pairwise_batch_size, 2, traj_length, 1, 28, 28)
        val_labels = torch.tensor(val_pairwise_labels).reshape(no_val_batches, pairwise_batch_size, 1)
        val_dataset = TensorDataset(val_pairwise_demos, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        val_pair_traj, val_l = next(iter(val_dataloader))

        if epoch == 0:
            print("dimension of a input from the dataloader", pair_traj.size())
            print("dimension of val input from the val dataloader", val_pair_traj.size())
            print(f"Length of training dataloader{len(training_dataloader)} and of val dataloader {len(val_dataloader)}")

        for pairwise_demo_batch, label_batch in training_dataloader:
            optimizer.zero_grad()
            count += 1

            pairwise_demo_batch_train = pairwise_demo_batch.view(pairwise_batch_size * 2 * traj_length, -1).to(device)

            pref_label = label_batch.reshape((pairwise_batch_size))
            # pref_label = pref_label.unsqueeze(0)
            pref_label = pref_label.to(device)

            ones = torch.ones((len(pairwise_demo_batch_train), 1)).to(device)
            ddt.forward(ddt.root, pairwise_demo_batch_train, ones)
            # C=ddt.get_penalty()
            loss_tree = ddt.get_loss()
            loss_tree = loss_tree.reshape(pairwise_batch_size,2,traj_length)
            loss_tree_traj = torch.sum(loss_tree, dim=2)

            pred_label = torch.argmax(loss_tree_traj, dim=1)
            # print(f"pred label is {pred_label} and pref label is {pref_label}")
            acc_counter += torch.sum((pred_label == pref_label).float())
            final_loss = loss_criterion(loss_tree_traj, pref_label)

            losses.append(final_loss.detach().cpu().numpy())
            cum_loss += final_loss.item()

            final_loss.backward()
            optimizer.step()
            train_size = no_train_batches * pairwise_batch_size


        training_loss_per_epoch = np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (train_size) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch', training_loss_per_epoch, epoch)
        writer.add_scalar(' Training Accuracy per epoch', training_acc_per_epoch, epoch)

        with torch.no_grad():
            val_acc_counter = 0
            val_cum_loss = 0
            val_accuracies = []
            val_losses = []
            for val_pairwise_demo_batch, val_label_batch in val_dataloader:
                count += 1

                val_pref_label = val_label_batch.reshape(pairwise_batch_size)
                val_pref_label = val_pref_label.to(device)

                val_pairwise_demo_batch_train = val_pairwise_demo_batch.view(pairwise_batch_size * 2 * traj_length, -1).to(device)

                ones = torch.ones((len(val_pairwise_demo_batch_train), 1)).to(device)
                ddt.forward(ddt.root, val_pairwise_demo_batch_train, ones)

                val_loss_tree = ddt.get_loss()
                val_loss_tree = val_loss_tree.reshape(pairwise_batch_size,2,traj_length)
                val_loss_tree_traj = torch.sum(val_loss_tree, dim=2)

                val_pred_label = torch.argmax(val_loss_tree_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())

                val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)
                val_losses.append(val_final_loss.detach().cpu().numpy())

                val_size = no_val_batches * pairwise_batch_size


            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (val_size) * 100
            print(" Val Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Validation Accuracy per epoch', val_acc_per_epoch, epoch)
    #
    #     #     early_stopping(val_loss_per_epoch)
    #     #     if early_stopping.early_stop:
    #     #         print("We are at epoch:", epoch)
    #     #         # print(
    #     #         #     f"total no of iterations are {no_epochs * len(training_labels)} and len of training data is {len(training_labels)}  and no of epochs are {epoch} ")
    #     #         # print(save_model_dir)
    #     #         torch.save(ddt, save_model_dir + exp_no + "_" + str(epoch))
    #     #         break
    #     # if early_stopping.early_stop:
    #     #     pass
    #     # else:
    torch.save(ddt, save_model_dir + exp_no + "_" + str(num_epochs))
    print(f"no of epochs are {num_epochs}")
    #
    # print("done")


input_dim=28*28
class_reward_vector=[0,9]
print(class_reward_vector)
tree_depth=4

tree = SoftDecisionTree(int(float(tree_depth)),input_dim, class_reward_vector)
print(tree)
weight_decay=0.000001
lr=0.00001
optimizer = optim.Adam(tree.parameters(), lr=float(lr), weight_decay=float(weight_decay))


train(traj_length=10,pairwise_batch_size=50,no_train_batches=3000,training_dataset=training_dataset,num_epochs=100,ddt=tree,optimizer=optimizer ,validation_dataset=validation_dataset,no_val_batches=500)


