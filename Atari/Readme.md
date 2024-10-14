# DDT Atari Experiments

First download the datasets for Breakout and BeamRider from 
https://uofu.box.com/v/DDT-Atari-Datset  into a new folder called datasets

We use Stable-Baselines3 for training RL policies on the learned reward model, so create a conda environment 
```console
$ conda env create -f atari_rl_env.yml
```
And activate it by 
```console
$ conda activate AtariDDT_SB3
```
## To reproduce our results and check for Misalignment wrt : ##
 - **Interpretability** (Fig 6 and 7 in paper)
 To create Hybrid Explanation,as in the synthetic traces for sophisticated internal nodes using our learned reward DDTs with Interpolated leafs run 
```console
$ python3 FERL_Node_trace.py  --env_name=breakout --DDT_with_penalty=False
```
Set the `DDT_with_penalty` flag to True if you want to create traces for DDT trained with regularization/penalty and change the `env_name` to beamrider if creating traces for beamrider. This script automatically saves traces to folder Vis.
Note: If running without penalty throws an error, it's normal, refer to paper for more details on why internal nodes without penalty can not be visualized. 

and for visualizing leaf nodes , run 

```console
$ python3 Leaf_Vis.py --env_name=breakout --DDT_with_penalty=False
```
Similar to internal node traces,set the `DDT_with_penalty` flag to True if you want to create traces for DDT trained with regularization/penalty and change the `env_name` to beamrider if creating traces for beamrider. This script automatically saves leaf nodes to folder Vis.



- **Evaluating RL** (table 3 in paper)
 To evaluate mean, standard deviation of our RL checkpoints for learned DDT reward model using soft as well as argmax (max probability path) rewards, run the command

```console
$ python3 Eval_Fixed_Leaf_A2C_DDT.py --env_name=breakout --RL_Penalty_Arg_Type=NPNA
```
Set `RL_Penalty_Arg_Type` to be either NPNA, NPA,PNA OR PA depending upon whether the RL policy was trained using learned reward DDT that was trained from preferences with or without penalty and whether we used soft rewards or argmax rewards.( NPNA- Not Penalty Not Argmax, NPA-Not Penalty Argmax, PNA-Penalty Not Argmax, PA-Penalty Argmax). Change the `env_name` to to beamrider if evaluating RL for beamrider.

Similarly to evaluate mean and std of learned neural network reward 

```console
$ python3 Eval_TREX_A2C.py --env_name=breakout 
```

For evaluting IQM (Inter-Quartile Mean) of our RL checkpoints for learned DDT reward model using soft as well as argmax (max probability path) rewards and TREX 
```console
$ python3 IQM.py --env_name=breakout --RL_Penalty_Arg_Type=NPNA --TREX=False
```
Set `RL_Penalty_Arg_Type` to be either NPNA, NPA,PNA OR PA depending upon whether the RL policy was trained using learned reward DDT or None if using TREX and in that case set `TREX` to True.


## To run RL on learned reward DDT : ##
```console
python3 Fixed_Leaf_A2C_DDT.py --env_name=breakout --RL_seed=0 --pth=A2C_DDT_RL/Breakout/NPNA/runs/seed0/ --save_model_dir=A2C_DDT_RL/Breakout/NPNA/ --exp_no=Breakout_NPNA_seed0 --checkpointing_dir=A2C_DDT_RL/Breakout/NPNA/checkpoints/seed0/ --Leaf_path=Trained_Reward_DDT/Breakout_Without_PenaltySeed0_LEAF.npy --Node_path=Trained_Reward_DDT/Breakout_Without_PenaltySeed0_NODE.npy --soft_routing_argmax=0
```
This script saves the final model, checkpoints and tensorboard to `save_model_dir` , `checkpointing_dir` and `pth` respectively with the exp_no for current experimental run,modify them as desired. Change the `env_name` to to beamrider if running RL for beamrider and set the `Leaf_path` and `Node_path` correctly to ensure if you are using DDT trained with or without penalty for both environments. The flag `soft_routing_argmax`  should be 0 for learning RL policy with soft rewards and 1 to learn RL policy using reward from maximum probability path in the learned DDT. 

Similarly to train a RL policy on learned TREX reward network, use 
```console
 python3 TREX_A2C.py --env_name=beamrider --RL_seed=0 --pth=A2C_DDT_RL/TREX/BeamRider/runs/seed0/ --save_model_dir=A2C_DDT_RL/TREX/BeamRider/ --exp_no=BeamRider_TREX_seed0 --checkpointing_dir=A2C_DDT_RL/TREX/BeamRider/checkpoints/seed0/
```
You can modify the  `env_name`  and  `seed` as desired and same as before , this script saves the final model, checkpoints and tensorboard to `save_model_dir` , `checkpointing_dir` and `pth` respectively with the exp_no for current experimental run,modify them as desired.

You can change path to reward models in both the scripts to learn a RL policy under your own reward function.

## To train reward DDT ##

To train reward models in Atari begin by creating another conda environment 
```console
$ conda env create -f atari_train_ddt_env.yml
```
And activate it by 
```console
$ conda activate Atari_TrainDDT
```

To learn a reward function DDT from preferences on Breakout, run 
```console
$ python3 AtariDDT_without_Penalty_NoDL.py --seed=0 --exp_no=Breakout-NoPenalty-DDT1 --pth=Breakout/runs/DDT/ --save_model_dir=Breakout/saved_models/DDT/ --num_epochs=100 
--tr_index=3000 --demos_path=datasets/Breakout_demonstrations_new
```
To learn a reward function DDT with penalty regularization from preferences on Breakout,

```console
$ python3 AtariDDT_without_Penalty_NoDL.py --seed=0 --exp_no=Breakout-NoPenalty-DDT1 --pth=Breakout/runs/DDT/ --save_model_dir=Breakout/saved_models/DDT/ --num_epochs=100 
--tr_index=3000 --demos_path=datasets/Breakout_demonstrations_new
```
You can modify the demos_path to `datasets/BeamRider_demonstrations_new` in both scripts above to learn a reward DDT for BeamRider. 

There's some incompability between the DDT models trained using older version of Pytorch and loading it with Stable Baselines3, so we extract the learned DDT parameters which are then used to train RL policy under our learned DDT reward. To extract 
```console
$ python3 Extract_DDTparams_for_RL.py --env_name=breakout --DDT_with_penalty=False
```
Similar as before,set the `DDT_with_penalty` flag to True if you want to create traces for DDT trained with regularization/penalty and change the `env_name` to beamrider if extracting parameters for beamrider. This script automatically saves leaf and internal node params to `Trained_Extracted_Params` folder. If you learned your own reward DDT, modify the  `reward_model_pth` path to your own model to extract the learned DDT parameters. 


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