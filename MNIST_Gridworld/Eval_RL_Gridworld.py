'''
Read in the files in new format GT Optimal Policy "+ " TREX " +" Random " + " Interpolated DDT Soft " + " Interpolated DDT Argmax"
Split on each line and inside each line split on comma
Make separate array for each policy to be evaluated
After sep arrays - do either the diff (GT policy value - Given policy value) or ratio (GT policy value/Given policy value) and append into semi-result arrays
Call np.mean over semi result arrays, assert that len(Semi-result array)==100 ( 100 seeds)
'''
'''Note : Use RL_Gridworld to first create the txt file that evaluates all policies'''
import random
import os
import torch
import numpy as np
import argparse

parser=argparse.ArgumentParser(description=None)
parser.add_argument('--pth', default=".", help="path of txt file containing RL policies")
args=parser.parse_args()
pth=args.pth

with open (pth,'r') as f:
    lines=f.readlines()
    all_GT= []
    all_Trex=[]
    all_Random=[]
    all_SoftDDT=[]
    all_ArgmaxDDT=[]
    for i,line in enumerate(lines):
        if i !=0:
            GT, Trex, Random, Soft,Argmax= line.strip('\n').split(',')
            all_GT.append(float(GT))
            all_Trex.append(float(Trex))
            all_Random.append(float(Random))
            all_SoftDDT.append(float(Soft))
            all_ArgmaxDDT.append(float(Argmax))



    GT_array=np.array(all_GT)
    Trex_array=np.array(all_Trex)
    Random_aray=np.array(all_Random)
    SoftDDT_aray = np.array(all_SoftDDT)
    ArgmaxDDT_aray = np.array(all_ArgmaxDDT)


    '''Ratio of policies'''
    Trex_wrt_GT=np.divide(Trex_array,GT_array)
    Random_wrt_GT=np.divide(Random_aray,GT_array)
    SoftDDT_wrt_GT=np.divide(SoftDDT_aray,GT_array)
    ArgmaxDDT_wrt_GT=np.divide(ArgmaxDDT_aray,GT_array)


    '''Difference of policies'''
    Trex_subtracted_from_GT=np.subtract(GT_array, Trex_array)
    Random_subracted_from_GT=np.subtract(GT_array, Random_aray)
    SoftDDT_subtracted_from_GT=np.subtract(GT_array, SoftDDT_aray)
    ArgmaxDDT_subtracted_from_GT=np.subtract(GT_array, ArgmaxDDT_aray)

    '''Average over ratios of policies'''
    assert len(Trex_wrt_GT)==100
    avg_Trex_wrt_GT=np.round(np.mean(Trex_wrt_GT),decimals=4)
    assert len(Random_wrt_GT) == 100
    avg_Random_wrt_GT =np.round(np.mean(Random_wrt_GT),decimals=4)
    assert len(SoftDDT_wrt_GT) == 100
    avg_SoftDDT_wrt_GT = np.round(np.mean(SoftDDT_wrt_GT),decimals=6)
    assert len(ArgmaxDDT_wrt_GT) == 100
    avg_ArgmaxDDT_wrt_GT =np.round( np.mean(ArgmaxDDT_wrt_GT),decimals=4)

    '''Average over difference of policies '''
    assert len(Trex_subtracted_from_GT) == 100
    avg_Trex_subtracted_from_GT =np.round(np.mean(Trex_subtracted_from_GT),decimals=4)
    assert len(Random_subracted_from_GT) == 100
    avg_Random_subtracted_from_GT = np.round(np.mean(Random_subracted_from_GT),decimals=4)
    assert len(SoftDDT_subtracted_from_GT) == 100
    avg_SoftDDT_subtracted_from_GT = np.round(np.mean(SoftDDT_subtracted_from_GT),decimals=4)
    assert len(ArgmaxDDT_subtracted_from_GT) == 100
    avg_ArgmaxDDT_subtracted_from_GT =np.round(np.mean(ArgmaxDDT_subtracted_from_GT),decimals=4)

    print(f"Policy Ratio wrt GT of Trex {avg_Trex_wrt_GT} , of Random {avg_Random_wrt_GT}, of SoftDDT {avg_SoftDDT_wrt_GT} and of ArgmaxDDT {avg_ArgmaxDDT_wrt_GT}")
    print(f"Policy Diff from GT of % of SoftDDT {avg_SoftDDT_subtracted_from_GT} and of ArgmaxDDT {avg_ArgmaxDDT_subtracted_from_GT}")


