# DDT Cartpole Experiments
Begin by installing our conda environment on a Linux machine with Nvidia GPU. On Ubuntu 22.04 ,
```console
$ conda env create -f cartpole_env.yml
```
And activate it by 
```console
$ conda activate cartpole
```

## To reproduce our results and check for Misalignment wrt : ##
 - **Interpretability** 
 To create heatmaps for internal nodes, run 
```console
$ python3 HeatMaps.py 
```
and for visualizing leaf nodes , run 

```console
$ python3 Leaf_Vis.py 
```
- **Evaluating RL** 
 To evaluate mean, standard deviation and IQM (Inter-Quartile Mean) of our RL checkpoints for learned DDT reward model using soft as well as argmax (max probability path) rewards, and similarly those of learned neural network reward and of ground truth run

```console
$ python3 Eval_RewardNetwork_RL_10_seeds_Mean_IQM_Std.py
```
Similarly for evaluating cartpole starting at cartposition not at center of track use,

```console
$ python3 Eval_GT_RewardNetwork_SIDE_RL_10seeds_Mean_IQM_Std.py 
```

## To train reward DDT and run RL on reward DDT ##

We make the Cartpole environment have a fixed horizon and collect demonstrations that ignore the done flag to avoid leaking any information. (See paper for more details.)

We release our demonstrations dataset and our preference dataset used to learn the reward functions, titled as `Dataset_epsiodelen200_num_episodes_100.zip` and `Pref_Dataset_num_prefs_2200_traj_snippet_len_20` respectively. Our demonstration dataset is collected over 200 timesteps per episode for 100 episodes each as use Cartpole-v0.

Optional- incase you want to collect your own demonstrations and construct your own preference dataset, use
```console
$ python3 create_dataset.py 
```


To learn a reward function DDT from preferences , run 
```console
$ python3 Reward_DDT.py 
```
Make sure your path to preference dataset is specified correclty. This script saves your final DDT reward model,config.yaml (includes lr and other hyperparameters) as well as tensorboard logs. This learns a multi-class reward leaf (CRL) DDT with simple internal nodes.

Similarly to train a reward neural network to compare against learned DDT reward model
```console
$ python3 Reward_NN.py 
```

To run RL on our learned reward DDT, execute following command
```console
python3 RL_DDT.py  --RL_seed=0 --exp_no=RL_DDT_Argmax_Seed0 --soft_routing_argmax=1
```
You must set `soft_routing_argmax` to 0 for soft reward during inference and 1 for argmax reward aka the reward using max probability path. 

Similarly to train a RL policy on our learned neural network , use
```console
python3  RL_NN.py --RL_seed=0 --exp_no=RL_NN_Seed0
```
You can change the learned reward model paths to use your own learned reward DDTs and neural networks.

To run Out-Of-Distribution RL experiments on our learned DDT reward model where the starting cart position is in the range [2.35, 2.45], execute 
```console
python3 Side_RL_DDT.py --RL_seed=0 --exp_no=Soft_DDT_RL_Side_seed0 --soft_routing_argmax=0 --start_side=2.4
```
You must set `soft_routing_argmax` to 0 for soft reward during inference and 1 for argmax reward aka the reward using max probability path. 

Similarly to train an Out-Of-Distribution RL policy on our learned neural network , run the following command 
```console
python3 Side_RL_NN.py --RL_seed=0 --exp_no=NN_RL_Side_seed0 --start_side=2.4
```

# If you find this repository is useful in your research, please cite the paper:
```
@article{kalra2024differentiable,
    title={Can Differentiable Decision Trees Enable Interpretable Reward Learning from Human Feedback?},
    author={Kalra, Akansha and Brown, Daniel S.},
    journal={Reinforcement Learning Journal},
    volume={4},
    pages={1887--1910},
    year={2024}
}

```