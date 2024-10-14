__author__ = "akansha_kalra"
import numpy as np
import argparse

parser=argparse.ArgumentParser(description=None)
parser.add_argument('--pth', default=".", help="path of txt file containing RL policies")
args=parser.parse_args()
pth=args.pth

with open (pth,'r') as f:
    lines=f.readlines()
    all_GT= []
    all_ClassReward_SoftDDT=[]
    all_ClassReward_ArgmaxDDT=[]
    for i,line in enumerate(lines):
        if i !=0:
            GT, ClassReward_Soft,ClassReward_Argmax= line.strip('\n').split(',')
            all_GT.append(float(GT))
            all_ClassReward_SoftDDT.append(float(ClassReward_Soft))
            all_ClassReward_ArgmaxDDT.append(float(ClassReward_Argmax))



    GT_array=np.array(all_GT)
    ClassReward_SoftDDT_aray = np.array(all_ClassReward_SoftDDT)
    ClassReward_ArgmaxDDT_aray = np.array(all_ClassReward_ArgmaxDDT)


    '''Ratio of policies'''
    ClassReward_SoftDDT_wrt_GT=np.divide(ClassReward_SoftDDT_aray,GT_array)
    ClassReward_ArgmaxDDT_wrt_GT=np.divide(ClassReward_ArgmaxDDT_aray,GT_array)


    '''Difference of policies'''
    ClassReward_SoftDDT_subtracted_from_GT=np.subtract(GT_array, ClassReward_SoftDDT_aray)
    ClassReward_ArgmaxDDT_subtracted_from_GT=np.subtract(GT_array, ClassReward_ArgmaxDDT_aray)

    '''Average over ratios of policies'''
    assert len(ClassReward_SoftDDT_wrt_GT) == 100
    avg_ClassReward_SoftDDT_wrt_GT = np.round(np.mean(ClassReward_SoftDDT_wrt_GT),decimals=4)
    assert len(ClassReward_ArgmaxDDT_wrt_GT) == 100
    avg_ClassReward_ArgmaxDDT_wrt_GT =np.round( np.mean(ClassReward_ArgmaxDDT_wrt_GT),decimals=4)

    '''Average over difference of policies '''
    assert len(ClassReward_SoftDDT_subtracted_from_GT) == 100
    avg_ClassReward_SoftDDT_subtracted_from_GT = np.round(np.mean(ClassReward_SoftDDT_subtracted_from_GT),decimals=4)
    assert len(ClassReward_ArgmaxDDT_subtracted_from_GT) == 100
    avg_ClassReward_ArgmaxDDT_subtracted_from_GT =np.round(np.mean(ClassReward_ArgmaxDDT_subtracted_from_GT),decimals=4)

    print(f"Policy Ratio wrt GT of SoftDDT {avg_ClassReward_SoftDDT_wrt_GT} and of ArgmaxDDT {avg_ClassReward_ArgmaxDDT_wrt_GT}")
    print(f"Policy Diff from GT of SoftDDT {avg_ClassReward_SoftDDT_subtracted_from_GT} and of ArgmaxDDT {avg_ClassReward_ArgmaxDDT_subtracted_from_GT}")