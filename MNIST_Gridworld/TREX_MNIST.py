import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torchvision import transforms , datasets
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from MNIST_Gridworld.Pref_Validation_Accuracy_CRL import mnist03

parser=argparse.ArgumentParser(description=None)
parser.add_argument('--seed',default=0,help="random seed for experiments")
parser.add_argument('--pth', default=".", help="path where tensorboard events are stored")
parser.add_argument('--save_model_dir',default=".",help="where to save trained model")
parser.add_argument('--exp_no',default="XX",help="which experiment number are you on")
parser.add_argument('--mnist03', default=True, type=bool, help="training neural net reward on mnist0-3 or mnist0-9")

args=parser.parse_args()
seed=args.seed
exp_no=args.exp_no
pth=args.pth+exp_no
save_model_dir=args.save_model_dir


torch.manual_seed(seed)



writer = SummaryWriter(pth)
if not os.path.exists(save_model_dir):
    print(' Creating Project : ' + save_model_dir)
    os.makedirs(save_model_dir)


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





class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.conv1 = nn.Conv2d(1, 5, kernel_size=7).to(self.device)
        self.conv2 = nn.Conv2d(5, 1, kernel_size=5).to(self.device)
        # self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(324, 10).to(self.device)
        self.fc2 = nn.Linear(10, 1).to(self.device)

    def cum_return(self, traj):
        conv_out1 = F.leaky_relu(self.conv1(traj))
        conv_out = F.leaky_relu(self.conv2(conv_out1))
        x = conv_out.reshape((conv_out.size(dim=0),-1))
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        return r


    def forward(self, traj_i):
        '''compute cumulative return for each trajectory and return logits'''
        cum_r = self.cum_return(traj_i)
        return cum_r

def train(traj_length, pairwise_batch_size, no_train_batches, training_dataset,reward_network,optimizer,num_epochs,validation_dataset, no_val_batches):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print(f"------------------Epoch:{epoch}---------")
        acc_counter = 0
        accuracies = []
        losses = []
        '''Generate training data for each epoch'''

        pairwise_trajs = []
        pairwise_labels = []
        for nb in range(no_train_batches):
            curr_batch_size = 0
            while curr_batch_size < pairwise_batch_size:
                idx = torch.randint(len(training_dataset), (2 * traj_length,))
                # print(idx,idx.shape)
                traj1_gt_return = 0
                traj2_gt_return = 0
                traj1 = []
                traj2 = []
                for count, i in enumerate(idx):
                    img, l = training_dataset[i]
                    # print(l)
                    if count < traj_length:
                        traj1_gt_return += l
                        traj1.append(img)
                    else:
                        traj2_gt_return += l
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
                    curr_batch_size += 1
            # print(pairwise_labels)

        training_pairwise_demos = torch.stack((pairwise_trajs)).reshape(-1, pairwise_batch_size, 2, traj_length, 1, 28, 28)

        training_labels = torch.tensor(pairwise_labels).reshape(no_train_batches, pairwise_batch_size, 1)
        # print(training_labels)
        # print(training_pairwise_demos.shape)
        # print(len(training_labels))

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

        val_pairwise_demos = torch.stack((val_pairwise_trajs)).reshape(-1, pairwise_batch_size, 2, traj_length, 1, 28,28)
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

            pairwise_demo_batch_train = pairwise_demo_batch.view(pairwise_batch_size * 2 * traj_length, 1,28,28).to(device)

            pref_label = label_batch.reshape((pairwise_batch_size))
            # pref_label = pref_label.unsqueeze(0)
            pref_label = pref_label.to(device)

            outputs = reward_network.forward(pairwise_demo_batch_train)
            reward_outputs = outputs.reshape(pairwise_batch_size, 2, traj_length)
            reward_outputs_traj=torch.sum(reward_outputs, dim=2)

            pred_label = torch.argmax(reward_outputs_traj, dim=1)
            acc_counter += torch.sum((pred_label == pref_label).float())

            final_loss = loss_criterion(reward_outputs_traj, pref_label)
            losses.append(final_loss.detach().cpu().numpy())

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


                val_pref_label = val_label_batch.reshape(pairwise_batch_size)
                # pref_label = pref_label.unsqueeze(0)
                val_pref_label = val_pref_label.to(device)

                val_pairwise_demo_batch_train = val_pairwise_demo_batch.view(pairwise_batch_size * 2 * traj_length,1,28,28).to(device)

                val_outputs = reward_network.forward(val_pairwise_demo_batch_train)
                val_reward_outputs = val_outputs.reshape(pairwise_batch_size, 2, traj_length)
                val_reward_outputs_traj = torch.sum(val_reward_outputs, dim=2)


                val_pred_label = torch.argmax(val_reward_outputs_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())

                val_final_loss = loss_criterion(val_reward_outputs_traj, val_pref_label)
                val_losses.append(val_final_loss.detach().cpu().numpy())
                val_size = no_val_batches * pairwise_batch_size


            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (val_size) * 100
            print(" Val Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', val_loss_per_epoch, epoch)
            writer.add_scalar('Validation Accuracy per epoch', val_acc_per_epoch, epoch)

    torch.save(reward_network.state_dict(), save_model_dir + exp_no + "_" + str(num_epochs))
    print(f"no of epochs are {num_epochs}")

if __name__ == '__main__':

    train_dataset = datasets.MNIST('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    vald_dataset = datasets.MNIST(root='./', train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))

    if mnist03:
        print("Training neural network reward on Mnist0-3")
        training_dataset = custom_index_dataset(train_dataset, [0, 1, 2, 3])
        validation_dataset = custom_index_dataset(vald_dataset, [0, 1, 2, 3])
    else:
        assert mnist03==False
        print("Training neural network reward on Mnist0-9")
        training_dataset=custom_index_dataset(train_dataset,[0,1,2,3,4,5,6,7,8,9])
        validation_dataset= custom_index_dataset(vald_dataset,[0,1,2,3,4,5,6,7,8,9])
    reward_nw=Net()
    weight_decay=0.001
    lr=0.0001
    optimizer = optim.Adam(reward_nw.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    if mnist03:
        train(traj_length=5,pairwise_batch_size=50,no_train_batches=3000,training_dataset=training_dataset,reward_network=reward_nw,optimizer=optimizer,num_epochs=100,validation_dataset=validation_dataset,no_val_batches=500)
    else:
        train(traj_length=10, pairwise_batch_size=50, no_train_batches=3000, training_dataset=training_dataset,
              reward_network=reward_nw, optimizer=optimizer, num_epochs=100, validation_dataset=validation_dataset,
              no_val_batches=500)

