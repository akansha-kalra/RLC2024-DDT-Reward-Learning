import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pylab as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from custom_subdataloader import custom_index_dataloader
from create_pref_demo import cre_traj,create_training_data
from torch.utils.data import TensorDataset,DataLoader
import argparse
import json
# import matplotlib as mpl
# mpl.use('Qt5Agg')
np.seterr(divide='ignore', invalid='ignore')

parser=argparse.ArgumentParser(description=None)
parser.add_argument('--seed',default=0,help="random seed for experiments")
parser.add_argument('--pth', default=".", help="path where tensorboard events are stored")
parser.add_argument('--save_model_dir',default=".",help="where to save trained model")
parser.add_argument('--tree_depth',default=1,help="depth of DDT")
parser.add_argument('--exp_no',default="XX",help="which experiment number are you on")
parser.add_argument('--states_in_a_traj',default=1,help="no of states in a traj")
parser.add_argument('--traj_indices',default=[0,1],help="which labels to use for creating trajs",type=json.loads)
parser.add_argument('--total_traj',default=10000,help=" total no of trajs")
parser.add_argument('--total_pairwise_demos',default=110,help=" total no of pairwise demons")
parser.add_argument('--training_pairwise_demos',default=100,help=" total no of training pairwise demons")
parser.add_argument('--pairwise_demons_batch_size',default=50,help="no of pairwise demonstrations fed in as a abatch")
parser.add_argument('--lr',default=0.0001,help="lr for experiments")
parser.add_argument('--weight_decay',default=0.0001,help="weight decay for experiments")
parser.add_argument('--num_epochs',default=1,help="no of epochs to train tree")
parser.add_argument('--penalty', default=True, action=argparse.BooleanOptionalAction)



args=parser.parse_args()
seed=args.seed
pth=args.pth
save_model_dir=args.save_model_dir
tree_depth=args.tree_depth
exp_no=args.exp_no
states_in_a_traj=args.states_in_a_traj
traj_indices= args.traj_indices
print(traj_indices)
total_traj=args.total_traj
total_pairwise_demos=args.total_pairwise_demos
training_pairwise_demos=args.training_pairwise_demos
pairwise_demons_batch_size=args.pairwise_demons_batch_size
weight_decay=args.weight_decay
lr=args.lr
num_epochs=args.num_epochs
penalty=args.penalty



writer = SummaryWriter(pth)

if not os.path.exists(save_model_dir):
    print(' Creating Project : ' + save_model_dir)
    os.makedirs(save_model_dir)
# if not os.path.isdir(dir)
# seed=0
torch.manual_seed(seed)




class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True




def pairwise_demons_dataloader(pairwise_demos_list,labels_list,batch_size):
    # print(pairwise_demos_list.size())
    # s=torch.stack(pairwise_demos_list)
    l=torch.tensor(labels_list)
    dataset=TensorDataset(pairwise_demos_list,l)
    pairwise_demos_dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=False)
    return pairwise_demos_dataloader

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

        self.fc = nn.Linear(self.input_size, 1,dtype=torch.float32).to(device)
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
    def __init__(self, depth, nb_classes, input_size,class_reward_vector,penalty):

        super(SoftDecisionTree, self).__init__()

        torch.manual_seed(seed)
        self.nb_classes = nb_classes  # output_dim
        self.input_size = input_size  # input_dim
        self.depth=depth
        self.class_reward = class_reward_vector
        self.penalty=penalty
        print(f"Tree has penalty {self.penalty}")

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
        if self.penalty == True:
            if torch.any(path_prob == 0) or torch.any(torch.isnan(prob)) == True:
                # pass
                raise ("Tree is dying, one cause: heavy penalty")
        ''' Causing Memory leak intensively
        pr=prob.data.cpu()[0]
        if current_node  in node_p:
            node_p[current_node].append(pr)
        else:
            node_p[current_node].extend(pr)'''
        # print(f"prob of the current node is {prob} and if needed current node is {current_node}")

        # Left Children -> prob = activation
        self.forward(current_node.children[0], inputs, prob * path_prob)
        # Right children -> prob = 1 - activation
        self.forward(current_node.children[1], inputs, (1 - prob) * path_prob)

    def get_penalty(self):
        C = 0
        for node in self.nodes:
            # print(node.alpha)
            C += -node.lmbda * 0.5 * (torch.log(node.alpha+1e-8) + torch.log((1 - node.alpha)+1e-8))

        return C

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



def train(ddt,training_dataloader, no_epochs, optimizer,val_dataloader,penalty=False):
    '''Comment in ES if needed- commenting it out for Longer Traj'''
    # early_stopping = EarlyStopping(patience=15, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    loss_criterion = nn.CrossEntropyLoss()
    # reduction='none' gives batch of CE loss bt then we need to do manual reduction :https://discuss.pytorch.org/t/reduction-none-leads-to-different-computed-loss/129515
    ddt=ddt.to(device)
    if penalty==True:
        print("WARNING : Tree will be penalized")

    for epoch in range(no_epochs):
        print(epoch)
        acc_counter=0
        # running_training_loss=0

        accuracies = []
        losses = []
        cum_loss=0
        count=0
        for pairwise_demo_batch,label_batch in training_dataloader:
            count+=1

            pref_label = label_batch.reshape((len(pairwise_demo_batch)))
            # pref_label = pref_label.unsqueeze(0)
            pref_label = pref_label.to(device)

            pairwise_demo_batch_train=pairwise_demo_batch.view(((len(pairwise_demo_batch))*(len(pairwise_demo_batch[0])*len(pairwise_demo_batch[0][0])),-1)).to(device)

            ones = torch.ones((len(pairwise_demo_batch_train), 1)).to(device)
            ddt.forward(ddt.root, pairwise_demo_batch_train, ones)
            # C=ddt.get_penalty()
            loss_tree = ddt.get_loss()
            loss_tree=loss_tree.reshape(len(pairwise_demo_batch),len(pairwise_demo_batch[0]),len(pairwise_demo_batch[0][0]))
            loss_tree_traj=torch.sum(loss_tree,dim=2)

            pred_label = torch.argmax(loss_tree_traj, dim=1)
            # print(f"pred label is {pred_label} and pref label is {pref_label}")
            acc_counter += torch.sum((pred_label == pref_label).float())

            if penalty==False:
                final_loss = loss_criterion(loss_tree_traj, pref_label)
            elif penalty==True:

                C = ddt.get_penalty()
                final_loss = loss_criterion(loss_tree_traj, pref_label) + torch.mean(C)

            losses.append(final_loss.detach().cpu().numpy())
            # print(final_loss.item())
            cum_loss+=final_loss.item()

            final_loss.backward()
            optimizer.step()
            train_size=len(training_dataloader)*len(pairwise_demo_batch)
        # print(count)
        # print(f"total no of training pairwise demons {train_size}")
        # print(f"training loss{cum_loss}")

        training_loss_per_epoch=np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (train_size) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch',training_loss_per_epoch, epoch)
        writer.add_scalar(' Training Accuracy per epoch', training_acc_per_epoch, epoch)


        with torch.no_grad():
            val_acc_counter=0
            val_cum_loss=0
            val_accuracies = []
            val_losses = []
            for val_pairwise_demo_batch, val_label_batch in val_dataloader:
                count += 1

                val_pref_label = val_label_batch.reshape((len(val_pairwise_demo_batch)))
                # pref_label = pref_label.unsqueeze(0)
                val_pref_label = val_pref_label.to(device)

                val_pairwise_demo_batch_train = val_pairwise_demo_batch.view(((len(val_pairwise_demo_batch)) * (len(val_pairwise_demo_batch[0]) * len(val_pairwise_demo_batch[0][0])), -1)).to(device)


                ones = torch.ones((len(val_pairwise_demo_batch_train), 1)).to(device)
                ddt.forward(ddt.root, val_pairwise_demo_batch_train, ones)

                val_loss_tree = ddt.get_loss()
                val_loss_tree = val_loss_tree.reshape(len(val_pairwise_demo_batch), len(val_pairwise_demo_batch[0]),len(val_pairwise_demo_batch[0][0]))
                val_loss_tree_traj = torch.sum(val_loss_tree, dim=2)

                val_pred_label = torch.argmax(val_loss_tree_traj, dim=1)
                # print(f"pred label is {pred_label} and pref label is {pref_label}")
                # print(f"val loss {val_loss_tree_traj, val_loss_tree_traj.size()} and pref label is {val_pref_label, val_pref_label.size()}")
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                if penalty == False:
                    val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)
                elif penalty==True:
                    val_C = ddt.get_penalty()
                    val_final_loss = loss_criterion(val_loss_tree_traj, val_pref_label)+torch.mean(val_C)

                val_losses.append(val_final_loss.detach().cpu().numpy())
                # print(final_loss.item())
                # val_cum_loss += val_final_loss.item()

                val_size = len(val_dataloader) * len(val_pairwise_demo_batch)

            # val_epoch_cum_loss=val_cum_loss / val_size
            # print(f"total no of val pairwise demons {val_size}")
            # print(f"Val loss{val_cum_loss}")

            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch",val_loss_per_epoch )
            val_acc_per_epoch = val_acc_counter / (val_size) * 100
            print(" Val Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Validation Accuracy per epoch', val_acc_per_epoch, epoch)
            scheduler.step(val_acc_per_epoch)

        '''Comment in ES if needed-commenting it out for Longer Traj'''
    #     early_stopping(val_loss_per_epoch)
    #     if early_stopping.early_stop:
    #         print("We are at epoch:", epoch)
    #         # print(
    #         #     f"total no of iterations are {no_epochs * len(training_labels)} and len of training data is {len(training_labels)}  and no of epochs are {epoch} ")
    #
    #         torch.save(ddt, save_model_dir + exp_no + "_" + str(epoch))
    #         break
    # if early_stopping.early_stop:
    #     pass
    # else:
    torch.save(ddt, save_model_dir + exp_no + "_" + str(num_epochs))
    print(f"no of epochs are {no_epochs}")






if __name__=="__main__":
    training_dataloader=custom_index_dataloader(int(states_in_a_traj),traj_indices,shuffle=False)

    x, y,_ = cre_traj(int(total_traj),training_dataloader)
    # print("total no of trajectories created", len(x))
    # for image in x[0:10]:
    #     plt.imshow(image.numpy().squeeze(), cmap='gray_r')
    #     plt.show()
    training_pairwise_demos=int(training_pairwise_demos)

    d, dr,td,_ = create_training_data(x, y, int(total_pairwise_demos))
    t, tr = td[:training_pairwise_demos], dr[:training_pairwise_demos]
    tr_dl=pairwise_demons_dataloader(t,tr,int(pairwise_demons_batch_size))
    print("no of objects in  training dataloader", len(tr_dl))

    v,vr=td[training_pairwise_demos:],dr[training_pairwise_demos:]
    v_dl=pairwise_demons_dataloader(v,vr,int(pairwise_demons_batch_size))
    print("no of objects in  val dataloader", len(v_dl))


    pr_demos,pr_label=next(iter(tr_dl))
    # print(pr_label)
    print("dimension of a input from the dataloader",pr_demos.size())
    # trajB1,trajB2=pr_demos[1]

    input_dim=28*28
    class_reward_vector=traj_indices
    print(class_reward_vector)
    nb_classes=len(class_reward_vector)

    tree = SoftDecisionTree(int(float(tree_depth)), nb_classes, input_dim,class_reward_vector,penalty)
    optimizer = optim.Adam(tree.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    train(tree,tr_dl, int(num_epochs), optimizer,v_dl,penalty)
    writer.flush()
