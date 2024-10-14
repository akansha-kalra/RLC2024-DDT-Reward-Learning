import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import gc


class AtariDataset(Dataset):
    def __init__(self,pref_traj_list,pref_label_list):
        self.pref_traj_list=pref_traj_list
        self.pref_traj_indexes=dict(enumerate(pref_traj_list))
        self.pref_label_list=pref_label_list
    def __len__(self):
        return len(self.pref_label_list)
    def __getitem__(self,index):
        pairwise_demos=self.pref_traj_indexes[index]
        torch_pairwise_demos=torch.from_numpy(np.array(pairwise_demos))
        pref_label=torch.tensor(self.pref_label_list[index])
        return torch_pairwise_demos, pref_label








def create_training_data(demonstrations, num_snippets, min_snippet_length, max_snippet_length,seed):
    np.random.seed(seed)
    # collect training data
    max_traj_length = 0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    # fixed size snippets with progress prior
    for n in range(num_snippets):
        ti = 0
        tj = 0
        # only add trajectories that are different returns
        while (ti == tj):
            # pick two random demonstrations
            ti = np.random.randint(num_demos)
            tj = np.random.randint(num_demos)
        # create random snippets
        # find min length of both demos to ensure we can pick a demo no earlier than that chosen in worse preferred demo
        min_length = min(len(demonstrations[ti]), len(demonstrations[tj]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)
        if ti < tj:  # pick tj snippet to be later than ti
            ti_start = np.random.randint(min_length - rand_length + 1)
            # print(ti_start, len(demonstrations[tj]))
            tj_start = np.random.randint(ti_start, len(demonstrations[tj]) - rand_length + 1)
        else:  # ti is better so pick later snippet in ti
            tj_start = np.random.randint(min_length - rand_length + 1)
            # print(tj_start, len(demonstrations[ti]))
            ti_start = np.random.randint(tj_start, len(demonstrations[ti]) - rand_length + 1)
        traj_i = demonstrations[ti][ti_start:ti_start + rand_length:2]  # skip everyother framestack to reduce size
        traj_j = demonstrations[tj][tj_start:tj_start + rand_length:2]

        max_traj_length = max(max_traj_length, len(traj_i), len(traj_j))
        if ti > tj:
            label = 0
        else:
            label = 1
        training_obs.append((traj_i, traj_j))
        training_labels.append(label)

    print("maximum traj length", max_traj_length)
    return training_obs, training_labels



def pairwise_demons_dataloader(pairwise_demos_list,labels_list,batch_size,shuffle):
    pairwise_demos=torch.from_numpy(np.array(pairwise_demos_list))
    l=torch.tensor(labels_list)
    dataset=TensorDataset(pairwise_demos,l)
    pairwise_demos_list=None
    labels_list=None
    gc.collect()
    print("Dataset made")
    pairwise_demos_dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
    print("Dataloader ready")
    return pairwise_demos_dataloader


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