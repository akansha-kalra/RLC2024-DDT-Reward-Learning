import torch
from matplotlib import pylab as plt
import numpy as np
from torchvision import transforms , datasets
# from create_pref_demo import cre_traj,create_training_data


def custom_index_dataset(batch_size,index_to_use):
    dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    for i in range(len(index_to_use)):
        # print(i)
        if i==0:
            idx = dataset.train_labels==index_to_use[i]
        else:
            idx += dataset.train_labels==index_to_use[i]
    # idx += dataset.train_labels == 2
    # idx += dataset.train_labels == 3
    # idx += dataset.train_labels == 4

    train_labels = dataset.train_labels[idx]
    train_data = dataset.train_data[idx]
    custom_dataset= torch.utils.data.dataset.Subset(dataset, np.where(idx==1)[0])
    # print(dset_train)
    return custom_dataset

'''traj_indices=[0,1,2,3]

custom_dataset=custom_index_dataset(1,traj_indices)

train_size=int(0.8*len(custom_dataset))
val_size=len(custom_dataset)-train_size
print(train_size)
print(val_size)
print(len(custom_dataset))

train_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size])

train_dl=torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
dl=iter(train_dl) 
images, labels = next(dl)
for i in images:
    plt.imshow(i.numpy().squeeze(), cmap='gray_r')
    plt.show()'''

def custom_index_dataloader(batch_size,index_to_use,shuffle):
    dataset = datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))]))
    for i in range(len(index_to_use)):
        # print(i)
        if i==0:
            idx = dataset.train_labels==index_to_use[i]
        else:
            idx += dataset.train_labels==index_to_use[i]
    # idx += dataset.train_labels == 2
    # idx += dataset.train_labels == 3
    # idx += dataset.train_labels == 4

    train_labels = dataset.train_labels[idx]
    train_data = dataset.train_data[idx]
    dset_train = torch.utils.data.dataset.Subset(dataset, np.where(idx==1)[0])
    # print(dset_train)
    dl_train = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=shuffle)
    # print(dl_train)
    #
    # print(len(train_labels))
    return dl_train

def single_index_datatloader(index,batch_size=1):
    index=int(index)
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose(
                                 [transforms.ToTensor(), transforms.Normalize((0.1307), (0.3081))]))
    idx = dataset.train_labels == index
    dset_train = torch.utils.data.dataset.Subset(dataset, np.where(idx == 1)[0])
    # print(len(dset_train))
    single_index_dataset = torch.utils.data.DataLoader(dset_train, batch_size=batch_size, shuffle=True)
    return single_index_dataset





'''check0=single_index_datatloader(index=0)
print("no of 0s", len(check0))

check1=single_index_datatloader(index=1)
print("no of 1s", len(check1))

check2=single_index_datatloader(index=2)
print("no of 2s", len(check2))

check3=single_index_datatloader(index=3)
print("no of 3s", len(check3))'''



'''dl_train=custom_index_dataloader(25,index_to_use=[0,1,2,3,4])
dl=iter(dl_train)

images, labels = next(dl)
for i in images:
    plt.imshow(i.numpy().squeeze(), cmap='gray_r')
    plt.show()'''

# images1, labels1 = dl.next().next()
# print(labels)
# print(torch.equal(images[0],images[1]))

# x,y=cre_traj(10,dl_train)
# print(f" no of trajs created {len(x)} and each traj has {len(x[6])} states ")
# print(y)
#
# d,dr=create_training_data(x,y,4)
# print(len(d), len(d[1]))
# print(dr)'''
# # print(train_labels)
# # print(train_data[0])
# # plt.imshow(train_data[10].numpy().squeeze(), cmap='gray_r')
# # plt.show()

#
''' code for creating trajectories without a dataloader
def cre_traj(total_traj):
    traj=[]
    demo = []
    demo_gt = []
    traj_counter = 0
    demo_counter=0
    traj_gt = 0
    for s, l in zip(train_data,train_labels):

        if demo_counter<total_traj:
            if traj_counter <3:
                traj.append(s)
                traj_gt += torch.sum(l)
                # print(traj_gt)
                # print(len(traj))
                traj_counter += 1
            else:
                print(traj_counter)
                print(f"len of traj {len(traj)}")
                demo.append(traj)
                demo_gt.append(traj_gt)
                traj_gt=0
                traj_counter=0
                demo_counter+=1
                traj=[]
    return demo, demo_gt'''
