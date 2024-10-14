import torch
from collections import defaultdict
from custom_subdataloader import custom_index_dataloader
from torchvision import transforms , datasets
import matplotlib.pyplot as plt

# state_batch_size=2
# train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
#                transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307), (0.3081))])),
# batch_size=state_batch_size, shuffle=True)

def cre_traj(total_traj,data_loader):
    # traj=[]
    demo = []
    demo_gt = []
    traj_counter = 0
    d_traj={}
    for s, l in data_loader:

        traj_gt = 0
        if traj_counter < total_traj:
            # if torch.equal(s[0], s[1]):
            #     print("danger")
            # else:
                # traj.append(s)
            traj_gt = torch.sum(l)
            # print(traj_gt)

            if traj_counter != 0:
                if not torch.equal(demo[traj_counter - 1], s):
                    pass
                demo.append(s)
                demo_gt.append(traj_gt)
            elif traj_counter == 0:
                demo.append(s)
                demo_gt.append(traj_gt)
            key=traj_gt.item()
            if key not in d_traj.keys():
                d_traj[key]=1
            elif key in d_traj.keys():
                d_traj[key]+=1

            traj_counter += 1
        else:
            break
    return demo, demo_gt,d_traj




def create_training_data(demonstrations, demos_true_reward,pref_demos):
    '''pref counter - if want to specify no of pref trajectories to be created-skip until loss thing resolved'''
    pref_counter=0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    done=False
    tensor_pairwise_demos = []
    d={}


    for i in range(0,num_demos-1):
        for j in range(i+1,num_demos):

            if pref_counter< pref_demos:
                if demos_true_reward[i] != demos_true_reward[j]:
                    # print("reward of 1st traj", demos_true_reward[i])
                    # print("%reward of 2nd traj", demos_true_reward[j])

                    if demos_true_reward[i]>  demos_true_reward[j] :
                        label = 0
                    else:
                        label = 1
                    # print("preference label",label)
                    key=(demos_true_reward[i].item(),demos_true_reward[j].item())
                    if key not in d.keys():
                        d[key]=1
                    elif key in d.keys():
                        d[key]+=1
                    traj_i= demonstrations[i]
                    traj_j= demonstrations[j]
                    training_labels.append(label)
                    training_obs.append((traj_i, traj_j))
                    t = torch.stack(list((traj_i, traj_j)))
                    tensor_pairwise_demos.append(t)
                    pref_counter+=1
            # elif demos_true_reward[i] == demos_true_reward[j]:
            #     pass
            #     i+=1
            #     j+=1
            else:
                done = True
                break
        if done:
            break
    tensor_pairwise_demos=torch.stack(tensor_pairwise_demos)
    return training_obs, training_labels,tensor_pairwise_demos,d

def create_balanced_training_data(demonstrations, demos_true_reward,individual_num_comparison):
    '''pref counter - if want to specify no of pref trajectories to be created-skip until loss thing resolved'''
    pref_counter=0
    training_obs = []
    training_labels = []
    num_demos = len(demonstrations)
    done=False
    tensor_pairwise_demos = []
    d={}
    # k=[(0,1,_),(1,0),(0,2),(2,0),(0,3),(3,0),(1,2),(2,1),(1,3),(3,1),(2,3),(3,2)]
    # d_track=dict.fromkeys(k,0)
    val_tensor_pairwise_demos=[]
    val_pref_labels=[]
    # val_d_track={}
    val_d=defaultdict(int)

    d_track={}

    for i in range(0,num_demos-1):
        for j in range(i+1,num_demos):

            # if pref_counter< pref_demos:
            if demos_true_reward[i] != demos_true_reward[j]:
                # print("reward of 1st traj", demos_true_reward[i])
                # print("%reward of 2nd traj", demos_true_reward[j])

                if demos_true_reward[i]>  demos_true_reward[j] :
                    label = 0
                    # key=(demos_true_reward[i])

                else:
                    label = 1
                # print("preference label",label)
                track_key=(demos_true_reward[i].item(),demos_true_reward[j].item(),label)
                # if track_key not in d_track.keys() or d_track[track_key]
                if track_key not in d_track.keys():
                    d_track[track_key]=1
                elif track_key in d_track.keys():
                    d_track[track_key]+=1
                key = (demos_true_reward[i].item(), demos_true_reward[j].item())
                if d_track[track_key]<= int(individual_num_comparison/2):
                    if key not in d.keys():
                        d[key]=1
                    elif key in d.keys():
                        d[key]+=1
                    traj_i= demonstrations[i]
                    traj_j= demonstrations[j]
                    training_labels.append(label)
                    # training_obs.append((traj_i, traj_j))
                    t = torch.stack(list((traj_i, traj_j)))
                    tensor_pairwise_demos.append(t)
                    # pref_counter+=1
                elif d_track[track_key]> int(individual_num_comparison/2) and val_d[key]<int(individual_num_comparison / 20):
                    # print(val_d)
                    # print(key)

                    if key not in val_d.keys() or  val_d[key] < int(individual_num_comparison / 20):

                        if key not in val_d.keys():
                            val_d[key] = 1
                        elif key in val_d.keys():
                            val_d[key] += 1
                        val_traj_i = demonstrations[i]
                        val_traj_j=demonstrations[j]
                        val_pref_labels.append(label)
                        val_t=torch.stack(list((val_traj_i, val_traj_j)))
                        val_tensor_pairwise_demos.append(val_t)

                elif key in val_d.keys() and (val_d[key] >= int(individual_num_comparison / 20)):
                    break

                # elif ((key not in val_d.keys()  and d_track[track_key]> int(individual_num_comparison/2))) or( val_d_track[key] <= int(individual_num_comparison / 20) and d_track[track_key]> int(individual_num_comparison/2)):
                    # elif key in val_d.keys():
                    #     if  val_d_track[key] <= int(individual_num_comparison / 20) :
                    #         val_d[key]+=1
                    #         val_traj_i=demonstrations[i]
                    #         val_traj_j=demonstrations[j]
                    #         val_pref_labels.append(label)
                    #         val_t=torch.stack(list((val_traj_i, val_traj_j)))
                    #         val_tensor_pairwise_demos.append(val_t)
                    #     if val_d_track[key]> int(individual_num_comparison / 20):
                    #         break
    tensor_pairwise_demos=torch.stack(tensor_pairwise_demos)
    val_tensor_pairwise_demos=torch.stack(val_tensor_pairwise_demos)
    return tensor_pairwise_demos,training_labels,d,d_track,val_tensor_pairwise_demos,val_pref_labels,val_d






def specific_pref_trajs(seq1,seq2,data_loader):
    traj1=None
    traj2=None
    pairwise_demons=[]
    traj_gt1=0
    traj_gt2 =0

    label=None
    for s , l in data_loader:
        # print(l)
        if torch.all(l.eq(seq1))==True:
            # print(l)
            traj1=s
            traj_gt1 = torch.sum(l)
        if torch.all(l.eq(seq2))==True:
            # print(l)
            traj2=s
            traj_gt2=torch.sum(l)


        if traj_gt1 !=0 and traj_gt2 !=0:
            if traj_gt1>traj_gt2:
                label=0
            else:
                label=1
            pairwise_demons.append((traj1,traj2))
            break
        elif (traj_gt1==0 or traj_gt2 ==0) and traj1 != None and traj2 != None:
            if traj_gt1 > traj_gt2:
                label = 0
            else:
                label = 1
            pairwise_demons.append((traj1, traj2))


            break
        else:
            pass

    return pairwise_demons,label


def specific_pref_traj_grid(seq,data_loader):
    grid_traj=None
    for s , l in data_loader:
        # print(l)
        if torch.all(l.eq(seq))==True:
            # print(l)
            grid_traj=s
            break
        else:
            pass

    return grid_traj



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

#
# if __name__=="__main__":
#     dt=custom_index_dataloader(3)
#     seq1=torch.tensor([3,3,0])
#     seq2=torch.tensor([4,2,1])
#     # seq1=[torch.tensor(3),torch.tensor(3),torch.tensor(0)]
#     # seq2= [torch.tensor(4), torch.tensor(2), torch.tensor(1)]
#     d,l=specific_pref_trajs(seq1,seq2,dt)
#     print(l)
#     print(len(d))
#     a,b=d[0]
#     print(len(a))
#     for s in a:
#         # print(s.size())
#         plt.imshow(s.numpy().squeeze(), cmap='gray_r')
#         plt.show()
#     for s in b:
#         plt.imshow(s.numpy().squeeze(), cmap='gray_r')
#         plt.show()
    # x, y = cre_traj(100)
    # d, dr = create_training_data(x, y, 1)