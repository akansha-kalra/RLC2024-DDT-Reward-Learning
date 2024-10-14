# DDT MNIST Gridworlds
First installthe  conda environment on a Linux machine with Nvidia GPU. On Ubuntu 22.04 ,
```console
$ conda env create -f mnist_DDT_env.yml
```
And activate it by 
```console
$ conda activate mnistDDT
```

##  MNIST (0-3) Gridworld ## 

### To reproduce our results: ###

- **Interpretability** (Fig 5 in paper)
  - To create ***pixel-level activation heatmaps*** (Fig 5a) of learned reward DDT with Interpolated Leaf nodes(IL-DDT) 
    ```console
    $ python3 Consolidated_Reg_Vis.py
     ```
    This script saves the node activation heatmap and leaf nodes with learned reward values under Node_Action_HeatMaps and Leaf_vis folders respectively inside the Vis folder.
  
  - To create ***Hybrid Explanations*** (Fig 5b) as in synthetic traces to interpret learned reward IL-DDT
      ```console
      $ python3 MNIST_trace_for_paper.py
      ```
       This script saves the traces with mnist digits and routing probability for each node under Traces folder

- **Training and Evaluating RL on learned DDT** (table 2 row 1 in paper)

    We begin by running RL on learned reward DDT with Interpolated Leaf nodes(IL-DDT) 
    ```console
    $ python3 RL_griworld.py
    ```
    This script also learns an RL policy using neural network reward model as baseline.

    Next to measure RL Performance as the percentage of expected return obtained relative to the performance of an optimal policy on the ground-truth reward, run

    ```console
    $ python3 Eval_RL_griworld.py --pth=R1-IL-0-3Grid_RL_Eval_depth2.txt
    ```
    
    Similalry to run RL on learned reward DDT with Class Reward Leaf nodes (CRL-DDT)
    ```console
    $ python3 Gridworld_ClassReward_Tree.py
    ```
    and to evaluate RL performance, run 
    ```console
    $ python3 Eval_ClassRewardRL_Gridworld.py --pth=Class_Reward_0-3Grid_RL_Eval_depth2.txt
    ```

### To train reward DDTs and neural network reward function ###

- To train your own ***IL-DDT*** from preferences,
```console
python3 Brand_New_Longer.py  --seed=0 --exp_no=IL-DDT1 --pth=MNIST03/runs/DDT/IL/ --save_model_dir=MNIST03/saved_models/DDT/IL/
```
- To learn your own ***CRL-DDT*** from preferences,
```console
python3 Reg_consolidated.py --seed=0 --pth=MNIST03/runs/DDT/CRL/ --save_model_dir=MNIST03/saved_models/DDT/CRL/ --tree_depth=2 --exp_no=CRL-DDT1 --states_in_a_traj=5 --traj_indices=[0,1,2,3] --total_traj=10000 --total_pairwise_demos=1100 --training_pairwise_demos=1000 --pairwise_demons_batch_size=50 --lr=0.5 --weight_decay=0.09 --num_epochs=100 --no-penalty
```
Similarly to learn your own ***Neural network reward*** from preferences,

```console
python3 TREX_MNIST.py  --seed=0 --exp_no=MNIST03-NN1 --pth=MNIST03/runs/NN/ --save_model_dir=MNIST03/saved_models/NN/ --mnist03=True
```




##  MNIST (0-9) Gridworld ## 

 **Training and Evaluating RL on learned DDT** (table 2 row 2 in paper)

- We begin by running RL on learned reward DDT with Interpolated Leaf nodes(IL-DDT) 
    ```console
    $ python3 RL_whole_MNIST_Griworld.py
    ```
    This script also learns an RL policy using neural network reward model as baseline.

    Next to measure RL Performance as the percentage of expected return obtained relative to the performance of an optimal policy on the ground-truth reward, run

    ```console
    $ python3 Eval_RL_griworld.py --pth=Whole_RL_Eval_depth4.txt
    ```
    
    Similalry to run RL on learned reward DDT with Class Reward Leaf nodes (CRL-DDT)
    ```console
    $ python3 Whole_Mnist_Class_Reward_RL.py
    ```
    and to evaluate RL performance, run 
    ```console
    $ python3 Eval_ClassRewardRL_Gridworld.py --pth=Whole_Mnist_Class_Reward_Gridworld_RL_Eval_depth4.txt
    ```
### To train reward DDTs and neural network reward function ###

- To train your own ***IL-DDT*** from preferences,
```console
python3 Whole_MNIST_gridworld_IL.py .py  --seed=0 --exp_no=IL-whole-DDT1 --pth=MNIST09/runs/DDT/IL/ --save_model_dir=MNIST09/saved_models/DDT/IL/
```


- To learn your own ***CRL-DDT*** from preferences,
```console
python3 Reg_consolidated.py --seed=0 --pth=MNIST09/runs/DDT/CRL/ --save_model_dir=MNIST09/saved_models/DDT/CRL/ --tree_depth=4 --exp_no=CRL-whole-DDT1 --states_in_a_traj=10 --traj_indices=[0,1,2,3,4,5,6,7,8,9] --total_traj=10000 --total_pairwise_demos=1100 --training_pairwise_demos=1000 --pairwise_demons_batch_size=50 --lr=0.001 --weight_decay=0.001 --num_epochs=100 --no-penalty
```

Similarly to learn your own ***Neural network reward*** from preferences,

```console
python3 TREX_MNIST.py  --seed=0 --exp_no=MNIST09-whole-NN1 --pth=MNIST09/runs/NN/ --save_model_dir=MNIST09/saved_models/NN/ --mnist03=False
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