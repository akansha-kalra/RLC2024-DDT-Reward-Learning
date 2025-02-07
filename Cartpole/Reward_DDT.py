__author__ = "akansha_kalra"
import torch
import random
import torch.nn as nn
import numpy as np
import os
import matplotlib.pylab as plt
from collections import defaultdict
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from Utils import EarlyStopping

seed=0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
print(f"seed is {seed}")

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
    def __init__(self, depth, nb_classes, input_size,lmbda):
        self.input_size = input_size
        self.nb_classes = nb_classes
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.fc = nn.Linear(self.input_size, 1).to(device)
        self.beta = nn.Parameter(torch.rand(1).to(device))  # inverse temperature
        # to compute penalty
        self.root_lmbda = lmbda
        self.lmbda = lmbda * 2 ** (-depth)
        self.alpha = 0  # will be set according to inputs


        if depth > 0:
            self.children = self.build_children(depth)
        else:
            self.children = [Leaf(nb_classes), Leaf(nb_classes)]

    def build_children(self, depth):
        return [Node(depth - 1, self.nb_classes, self.input_size,self.root_lmbda),
                Node(depth - 1, self.nb_classes, self.input_size,self.root_lmbda)]
    def forward(self, x):
        return (torch.sigmoid(self.beta*self.fc(x)))


class SoftDecisionTree(nn.Module):
    def __init__(self, depth, nb_classes, input_size,class_reward_vector,seed):

        super(SoftDecisionTree, self).__init__()

        torch.manual_seed(seed)
        self.nb_classes = nb_classes  # output_dim
        self.input_size = input_size  # input_dim
        self.depth=depth
        self.class_reward = class_reward_vector


        # build tree
        self.root = Node(self.depth - 1, self.nb_classes, self.input_size, lmbda=0.1)
        self.nodes = []
        self.leaves = []

        # global node_p
        # node_p = defaultdict(list)
        #
        # global node_name
        # node_name={}
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

        # Left Children -> prob = activation
        self.forward(current_node.children[0], inputs, prob * path_prob)
        # Right children -> prob = 1 - activation
        self.forward(current_node.children[1], inputs, (1 - prob) * path_prob)


    def get_loss(self):
        loss = 0
        class_reward = torch.tensor(self.class_reward).float().to(self.device)
        class_reward = torch.unsqueeze(class_reward, dim=0)
        loss_tree = 0
        #         print("no nof leaves", len(self.leaves))
        for leaf in self.leaves:
            Q = (leaf.forward()).float().to(self.device)
            loss_l = torch.inner(class_reward, Q)
            loss = torch.sum((loss_l * leaf.path_prob),dim=1)
            loss_tree += loss
        #     print("loss of a tree", loss)
        # print(" final loss of a tree", loss_tree)
        return loss_tree

    def fwd_input(self, input_traj):
        traj_reward = 0
        for input_state in input_traj:
            # print(input_state)
            input_state = input_state.to(self.device)
            input_state = torch.reshape(input_state, (1, -1))
            ones = torch.ones(len(input_state), 1).to(self.device)
            self.forward(self.root, input_state, ones)
            state_reward = self.get_loss()
            # print("state reward",state_reward)
            traj_reward += torch.sum(state_reward)

        final_traj_reward = traj_reward

        #     print("state reward",state_reward)
        # print("final traj reward",final_traj_reward)
        return final_traj_reward

def train(ddt,train_dl, optimizer,val_dl, num_epochs,save_model_dir='.',exp_no=0,ES_patience=15,lr_scheduler=None):

    early_stopping = EarlyStopping(patience=ES_patience, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    ddt = ddt.to(device)

    for epoch in range(num_epochs):
        acc_counter = 0
        losses = []

        if lr_scheduler!=None:
            print(f"-----------Epoch{epoch} and lr is {lr_scheduler.get_last_lr()}  ---------------")
        else:
            print(f"-----------Epoch{epoch}---------------")
        for pref_demo, pref_label in train_dl:
            optimizer.zero_grad()
            pref_label = pref_label.to(device)
            pref_demo_train = pref_demo.view(len(pref_demo)*len(pref_demo[0])*len(pref_demo[0][0]),2).float().to(device)
            ones = torch.ones((len(pref_demo_train), 1)).float().to(device)
            ddt.forward(ddt.root, pref_demo_train, ones)
            loss_tree = ddt.get_loss()
            loss_tree = loss_tree.reshape(len(pref_demo),len(pref_demo[0]), len(pref_demo[0][0]))
            loss_tree_traj = torch.sum(loss_tree, dim=2)

            pred_label = torch.argmax(loss_tree_traj, dim=1)
            # print(f"pred label is {pred_label} and pref label is {pref_label}")
            acc_counter += torch.sum((pred_label == pref_label).float())
            final_loss = loss_criterion(loss_tree_traj, pref_label)
            losses.append(final_loss.detach().cpu().numpy())

            final_loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        training_loss_per_epoch = np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (len(train_dl)*len(pref_demo)) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch', training_loss_per_epoch, epoch)
        writer.add_scalar(' Training Accuracy per epoch', training_acc_per_epoch, epoch)

        with torch.no_grad():
            val_acc_counter = 0
            val_losses = []
            for val_pref_demo, val_pref_label in val_dl:

                val_pref_label = val_pref_label.to(device)
                val_pref_demo_train = val_pref_demo.view(len(val_pref_demo)*len(val_pref_demo[0]) * len(val_pref_demo[0][0]), 2).float().to(device)
                val_ones = torch.ones((len(val_pref_demo_train), 1)).float().to(device)
                ddt.forward(ddt.root, val_pref_demo_train, val_ones)
                val_loss_tree = ddt.get_loss()
                val_loss_tree = val_loss_tree.reshape(len(val_pref_demo), len(val_pref_demo[0]), len(val_pref_demo[0][0]))
                val_loss_tree_traj = torch.sum(val_loss_tree, dim=2)

                val_pred_label = torch.argmax(val_loss_tree_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)
                val_losses.append(val_final_loss.detach().cpu().numpy())


            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (len(val_dl)*len(val_pref_demo)) * 100
            print("VAL Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Val Accuracy per epoch', val_acc_per_epoch, epoch)
            '''use this for ReduceLRonPlateau- NOT USING IT RIGHT NOW'''
            # if lr_scheduler is not None:
            #     scheduler.step(val_loss_per_epoch)
            early_stopping(val_loss_per_epoch)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                torch.save(ddt, save_model_dir + exp_no + "_" + str(epoch))
                break
    if early_stopping.early_stop:
        pass
    elif not early_stopping.early_stop:
        torch.save(ddt, save_model_dir + exp_no + "_" + str(num_epochs))
        print(f"no of epochs are {num_epochs}")

if __name__== '__main__':

    '''prep data'''
    num_prefs=2200
    traj_snippet_len=20
    pref_dataset_path='Pref_Dataset_num_prefs_'+str(num_prefs)+'_traj_snippet_len_'+str(traj_snippet_len)
    pref_dataset=torch.load(pref_dataset_path)
    pref_demos=pref_dataset['pref_demos']
    pref_labels=pref_dataset['pref_labels']
    assert len(pref_demos) == len(pref_labels) == num_prefs
    num_train_prefs=2000

    train_pref_demos=pref_demos[:num_train_prefs]
    train_pref_labels=pref_labels[:num_train_prefs]

    val_pref_demos=pref_demos[num_train_prefs:]
    val_pref_labels=pref_labels[num_train_prefs:]

    train_dataset = TensorDataset(torch.stack(train_pref_demos),torch.tensor(train_pref_labels))
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_datset = TensorDataset(torch.stack(val_pref_demos),torch.tensor(val_pref_labels))
    val_dl = DataLoader(val_datset, batch_size=1, shuffle=False)

    val_dl_len=len(val_dl)
    train_dl_len=len(train_dl)

    save_config=True
    input_dim = 1 * 2
    depth = 2
    class_reward_vector = [0, 1]
    nb_classes = len(class_reward_vector)
    tree = SoftDecisionTree(depth, nb_classes, input_dim, class_reward_vector, seed=seed)
    lr=0.001
    weight_decay=0.000

    optimizer = optim.Adam(tree.parameters(), lr=lr, weight_decay=weight_decay)
    Exp_name = 'CP-DDT-1'
    current_directory = os.getcwd()
    save_model_dir = current_directory +'/Reward_Models/DDT/saved_models/'
    tensorboard_path = current_directory +'/Reward_Models/DDT/TB/' + Exp_name

    writer = SummaryWriter(tensorboard_path)
    if not os.path.exists(save_model_dir):
        print(' Creating Project : ' + save_model_dir)
        os.makedirs(save_model_dir)

    if save_config:
        config=dict()
        config['seed'] = seed
        config['input_dim'] = input_dim
        config['depth'] = depth
        config['class_reward_vector'] = class_reward_vector
        config['lr'] = lr
        config['weight_decay'] = weight_decay
        config[' num_train_prefs'] = num_train_prefs
        config['train_dl_len']=train_dl_len
        config['val_dl_len']=val_dl_len

        save_config_dir = current_directory +'/Reward_Models/DDT/configs/'
        if not os.path.exists(save_config_dir):
            print('Creating Project : ' + save_config_dir)
            os.makedirs(save_config_dir)
        path = save_config_dir + Exp_name + "_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

    train(tree, train_dl, optimizer, val_dl, num_epochs=50, save_model_dir=save_model_dir, exp_no=Exp_name,
          ES_patience=10, lr_scheduler=None)
