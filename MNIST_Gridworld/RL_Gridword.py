'''
Pipeline Setup
- 5x5 gridworld
sample 25 images randomly from mnist to create this gridworld , also store these images for visualizing gridworld and for creating ground truth reward matrix
- once you've the input images - pass them through the learnt DDT model to get the DDT reward matrix  r_hat
- given r_hat , run value iteration on this r_hat to get V_hat under r_hat
- From V_hat get Q_hat , get optimal policy under r_hat
- run value iteration on ground truth reward r to get optimal policy pi_star
- run policy evaluation on pi_star and pi_hat

'''
import random
import os
import torch
import numpy as np
from RL_helper_functions import value_iteration, get_optimal_policy, policy_evaluation_GT_reward, evaluate_random_policy , MDP , print_array_as_grid , visualize_policy , Net
import matplotlib.pylab as plt
from TestTime_Brand_New_Tree import Leaf, Node, SoftDecisionTree , custom_index_dataset
import matplotlib as mpl
mpl.use('Qt5Agg')


policy_file=open('R1-IL-0-3Grid_RL_Eval_depth2.txt','w')

policy_file.write("GT Optimal Policy "+ " TREX " +" Random " + " Interpolated DDT Soft " + " Interpolated DDT Argmax ")
policy_file.write('\n')

TREX_MNIST_net= Net()
net_path='Trained_Reward_Models/mnist0-3/TREX/Long1_100'
TREX_MNIST_net.load_state_dict(torch.load(net_path))


for sd in range(100):
    print(f"NEW SEEED STARTING : SEED{sd}")
    seed=sd
    random.seed(seed)
    torch.manual_seed(seed)


    '''Sampling'''

    training_dataset=custom_index_dataset([0,1,2,3])

    idx = torch.randint(len(training_dataset), (25,))

    gridworld_input=[]
    gridworld_gt_reward=[]
    for i in idx:
        # e=idx[i]
        img,l=training_dataset[i]
        gridworld_input.append(img)
        gridworld_gt_reward.append(l)


    input_dim=28*28
    class_reward_vector=[0,3]
    r_hat_gridworld_SOFT=[]
    r_hat_gridworld_ArgMax=[]

    TREX_r_hat_gridworld=[]



    with torch.no_grad():
        depth=2
        tree = SoftDecisionTree(depth, input_dim,class_reward_vector)
        tree = torch.load('Trained_Reward_Models/mnist0-3/DDT/IL/R1-WAY-BETTER-seed0_withES_100')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tree.to(device)
        tree.eval()



        for i in range(len(gridworld_input)):
            curr_grid_input=gridworld_input[i]
            soft_Curr_reward = tree.soft_forward(curr_grid_input)
            Argmax_curr_grid_reward=tree.forward(curr_grid_input)
            curr_grid_input=curr_grid_input.to(device)
            TREX_curr_reward=TREX_MNIST_net.forward(curr_grid_input)
            r_hat_gridworld_SOFT.append(soft_Curr_reward.item())
            r_hat_gridworld_ArgMax.append(Argmax_curr_grid_reward.item())
            TREX_r_hat_gridworld.append(TREX_curr_reward.item())


    gamma = 0.9  # discount factor

    terminals = []  # set no state to be terminal


    env_GT = MDP(5, 5, terminals, gridworld_gt_reward, gamma)
    env_DDT_SOFT = MDP(5, 5, terminals, r_hat_gridworld_SOFT, gamma)
    env_DDT_Argmax = MDP(5, 5, terminals, r_hat_gridworld_ArgMax, gamma)
    env_TREX_MNIST= MDP(5,5 ,terminals, TREX_r_hat_gridworld, gamma)


    '''SOFT TREE'''
    print("---DDT SOFT---")
    value_under_r_hat_SOFT= value_iteration(env_DDT_SOFT)
    learned_SOFT_DDT_policy= get_optimal_policy(env_DDT_SOFT,epsilon=0.0001, V=value_under_r_hat_SOFT)
    SOFT_DDT_policy_eval = policy_evaluation_GT_reward(env_GT, learned_SOFT_DDT_policy, epsilon=0.0001)
    print(f"SOFT DDT POLICY EVAL")
    value_policy_eval_SOFT_DDT=np.mean(SOFT_DDT_policy_eval)
    print(f"Evaluating Learned SOFT-DDT policy under GT reward {value_policy_eval_SOFT_DDT}")

    print("--DDT Argmax---")
    value_under_r_hat_Argmax = value_iteration(env_DDT_Argmax)
    learned_Argmax_DDT_policy = get_optimal_policy(env_DDT_Argmax, epsilon=0.0001, V=value_under_r_hat_Argmax)
    Argmax_DDT_policy_eval = policy_evaluation_GT_reward(env_GT, learned_Argmax_DDT_policy, epsilon=0.0001)
    value_policy_eval_Argmax_DDT =np.mean(Argmax_DDT_policy_eval)
    print(f"Evaluating Learned Argmax-DDT policy under GT reward {value_policy_eval_Argmax_DDT}")


    '''TREX MNIST POLICY'''
    value_under_TREX_reward= value_iteration(env_TREX_MNIST)
    learned_TREX_policy= get_optimal_policy(env_TREX_MNIST,epsilon=0.0001,V=value_under_TREX_reward)
    TREX_policy_eval= policy_evaluation_GT_reward(env_GT,learned_TREX_policy,epsilon=0.0001)
    value_policy_eval_TREX_MNIST=np.mean(TREX_policy_eval)
    print(f"Evaluating TREX MNIST policy {value_policy_eval_TREX_MNIST}")


    Random_policy_eval = evaluate_random_policy(env_GT, epsilon=0.0001)
    value_policy_eval_Random=np.mean(Random_policy_eval)
    print(f"Evaluating Random policy under GT reward {value_policy_eval_Random}")

    value_pi_optimal = value_iteration(env_GT)
    value_optimal=np.mean(value_pi_optimal)
    print("value function for optimal policy under GT reward", value_optimal)



    policy_file.write(f'{value_optimal},{value_policy_eval_TREX_MNIST}, {value_policy_eval_Random}, {value_policy_eval_SOFT_DDT},{value_policy_eval_Argmax_DDT}')
    policy_file.write('\n')

    Optimal_policy_under_GT_reward = get_optimal_policy(env_GT, epsilon=0.0001, V=value_pi_optimal)
    print("-" * 80)


policy_file.close()

