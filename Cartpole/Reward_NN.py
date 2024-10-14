__author__ = "akansha_kalra"

import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pylab as plt
from collections import defaultdict
from torch.utils.data import TensorDataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import yaml
from Utils import EarlyStopping

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(2, 16,dtype=torch.float32)
        self.fc2 = nn.Linear(16, 16,dtype=torch.float32)
        self.fc3 = nn.Linear(16, 1,dtype=torch.float32)
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, traj):
        input_traj = traj.to(self.device)
        # sum_rewards=0
        x = F.leaky_relu(self.fc1(input_traj))
        x = F.leaky_relu(self.fc2(x))
        r = self.fc3(x)
        return r





def train(model, train_dl, optimizer, val_dl, num_epochs, save_model_dir='.', exp_no=0, ES_patience=15, lr_scheduler=None):
    early_stopping = EarlyStopping(patience=ES_patience, min_delta=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()
    model = model.to(device)

    for epoch in range(num_epochs):
        acc_counter = 0
        losses = []

        if lr_scheduler != None:
            print(f"-----------Epoch{epoch} and lr is {lr_scheduler.get_last_lr()}  ---------------")
        else:
            print(f"-----------Epoch{epoch}---------------")
        for pref_demo, pref_label in train_dl:
            optimizer.zero_grad()
            pref_label = pref_label.to(device)
            pref_demo_train = pref_demo.view(len(pref_demo) * len(pref_demo[0]) * len(pref_demo[0][0]), 2).float().to(device)

            loss_nn = model.forward(pref_demo_train)
            loss_net = loss_nn.reshape(len(pref_demo), len(pref_demo[0]), len(pref_demo[0][0]))
            loss_net_traj = torch.sum(loss_net, dim=2)

            pred_label = torch.argmax(loss_net_traj, dim=1)
            # print(f"pred label is {pred_label} and pref label is {pref_label}")
            acc_counter += torch.sum((pred_label == pref_label).float())
            final_loss = loss_criterion(loss_net_traj, pref_label)
            # print("loss",final_loss)
            losses.append(final_loss.detach().cpu().numpy())

            final_loss.backward()
            optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        training_loss_per_epoch = np.mean(losses)
        print("Training Loss per epoch", training_loss_per_epoch)
        training_acc_per_epoch = acc_counter / (len(train_dl) * len(pref_demo)) * 100
        print(" Training Accuracy per epoch", training_acc_per_epoch)
        writer.add_scalar('Training Loss per epoch', training_loss_per_epoch, epoch)
        writer.add_scalar(' Training Accuracy per epoch', training_acc_per_epoch, epoch)

        with torch.no_grad():
            val_acc_counter = 0
            val_losses = []
            for val_pref_demo, val_pref_label in val_dl:
                val_pref_label = val_pref_label.to(device)
                val_pref_demo_train = val_pref_demo.view(
                    len(val_pref_demo) * len(val_pref_demo[0]) * len(val_pref_demo[0][0]), 2).float().to(device)


                val_loss_nn = model.forward(val_pref_demo_train)
                val_loss_net = val_loss_nn.reshape(len(val_pref_demo), len(val_pref_demo[0]),
                                                      len(val_pref_demo[0][0]))
                val_loss_net_traj = torch.sum(val_loss_net, dim=2)

                val_pred_label = torch.argmax(val_loss_net_traj, dim=1)
                val_acc_counter += torch.sum((val_pred_label == val_pref_label).float())
                val_final_loss = loss_criterion(val_loss_net_traj, val_pref_label)
                val_losses.append(val_final_loss.detach().cpu().numpy())

            val_loss_per_epoch = np.mean(val_losses)
            print("Val Loss per epoch", val_loss_per_epoch)
            val_acc_per_epoch = val_acc_counter / (len(val_dl) * len(val_pref_demo)) * 100
            print("VAL Accuracy per epoch", val_acc_per_epoch)
            writer.add_scalar('Val Loss per epoch', training_loss_per_epoch, epoch)
            writer.add_scalar('Val Accuracy per epoch', training_acc_per_epoch, epoch)
            '''use this for ReduceLRonPlateau- NOT USING IT RIGHT NOW'''
            # if lr_scheduler is not None:
            #     scheduler.step(val_loss_per_epoch)
            early_stopping(val_loss_per_epoch)
            if early_stopping.early_stop:
                print("We are at epoch:", epoch)
                if save_model_dir is not None:
                    torch.save(model, save_model_dir + exp_no + "_" + str(epoch))
                break
    if early_stopping.early_stop:
        pass
    elif not early_stopping.early_stop:
        if save_model_dir is not None:
            torch.save(model, save_model_dir + exp_no + "_" + str(num_epochs))
        print(f"no of epochs are {num_epochs}")


if __name__ == '__main__':
    '''prep data'''
    num_prefs = 2200
    traj_snippet_len = 20
    pref_dataset_path = 'Pref_Dataset_num_prefs_' + str(
        num_prefs) + '_traj_snippet_len_' + str(traj_snippet_len)
    pref_dataset = torch.load(pref_dataset_path)
    pref_demos = pref_dataset['pref_demos']
    pref_labels = pref_dataset['pref_labels']
    assert len(pref_demos) == len(pref_labels) == num_prefs
    num_train_prefs = 2000

    train_pref_demos = pref_demos[:num_train_prefs]
    train_pref_labels = pref_labels[:num_train_prefs]

    val_pref_demos = pref_demos[num_train_prefs:]
    val_pref_labels = pref_labels[num_train_prefs:]

    train_dataset = TensorDataset(torch.stack(train_pref_demos), torch.tensor(train_pref_labels))
    train_dl = DataLoader(train_dataset, batch_size=1, shuffle=False)

    val_datset = TensorDataset(torch.stack(val_pref_demos), torch.tensor(val_pref_labels))
    val_dl = DataLoader(val_datset, batch_size=1, shuffle=False)

    val_dl_len = len(val_dl)
    train_dl_len = len(train_dl)

    save_config = True

    net = SimpleNet()
    lr = 0.001
    weight_decay = 0.000

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    Exp_name = 'CP-NN-1'
    current_directory = os.getcwd()
    save_model_dir = current_directory +'/Reward_Models/NN/saved_models/'
    tensorboard_path = current_directory +'/Reward_Models/NN/TB/' + Exp_name

    writer = SummaryWriter(tensorboard_path)
    if not os.path.exists(save_model_dir):
        print(' Creating Project : ' + save_model_dir)
        os.makedirs(save_model_dir)

    if save_config:
        config = dict()
        config['seed'] = seed
        config['lr'] = lr
        config['weight_decay'] = weight_decay
        config[' num_train_prefs'] = num_train_prefs
        config['train_dl_len'] = train_dl_len
        config['val_dl_len'] = val_dl_len

        save_config_dir = current_directory +'/Reward_Models/NN/configs/'
        if not os.path.exists(save_config_dir):
            print('Creating Project : ' + save_config_dir)
            os.makedirs(save_config_dir)
        path = save_config_dir + Exp_name + "_config.yaml"
        with open(path, "w") as f:
            yaml.dump(config, f)

    train(net, train_dl, optimizer, val_dl, num_epochs=50, save_model_dir=save_model_dir, exp_no=Exp_name,
          ES_patience=10, lr_scheduler=None)