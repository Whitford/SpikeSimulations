########################################################
# trajectories_with_force_analysis.py is a script that 
# will analyze simulations of the Spike protein when
# simulated under the influence of an external force.
# Written by Esteban Dodero-Rojas and edited by Paul
# Whitford. This analysis is described in Grunst et al,
# Science, 2024.
#
# Prerequisites: python3 and the following python
# libraries: mdtraj, numpy, matplotlib and joblib
#
# To run this analysis, you should download the set of
# simulated trajectories. The data may be downloaded 
# from DRYAD:
# DOI: 10.5061/dryad.ncjsxkt3h
# 
# After downloading the full set of data (directory 
# SpikeSimData), run this script from within SpikeSimData,
# using the command:
# >python analysis_for_trajectories_with_force.py
#
# this will generate the following files:
#   TM-HR1_distance_vs_time.png
#   TM-HR1_distance_vs_time.svg
#   fraction_of_successful_transitions.png
#   fraction_of_successful_transitions.svg
#
########################################################

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

PATH_TO_TRAJECTORIES_WITH_FORCE='./trajectories_with_force_Grunst_Science_2024'
THREADS=10
# DOWNSAMPLE allows one to skip some frames during analysis. 1=all frames, N=analyze every Nth frame. 
DOWNSAMPLE=1

def analyzerep(repid,w):
    if w:
        w="/with_antibody"
        gro="/sim_with_CV3.gro"
        ww="with Ab"
    else:
        w="/without_antibody"
        gro="/sim_without_CV3.gro"
        ww="without Ab"
  
    print("\t{} run {}".format(ww,repid), end='\r')
    trj=md.load(PATH_TO_TRAJECTORIES_WITH_FORCE+w+'/force_'+stspN[s]+'pN/run_'+str(repid)+'/trajectory.xtc',
                  top=PATH_TO_TRAJECTORIES_WITH_FORCE+w+gro,stride=DOWNSAMPLE)
    CM_TM=np.mean(trj.xyz[:,TM_index,:],axis=1)
    CM_HR1=np.mean(trj.xyz[:,HR1_index,:],axis=1)
    dist_TM_HR1=np.linalg.norm(CM_TM-CM_HR1,axis=1)
    time=trj.time/1000
    
    return time, dist_TM_HR1


# TM index 
TM1_index = list(range(3928,4092))
TM2_index = list(range(8020,8184))
TM3_index = list(range(12112,12276))
TM_index=np.concatenate([TM1_index,TM2_index,TM3_index])
# HR1 index
HR11_index = list(range(9832-2*4092,9909-2*4092))
HR12_index = list(range(9832,9909))
HR13_index = list(range(9832-4092,9909-4092))
HR1_index=np.concatenate([HR11_index,HR12_index,HR13_index])

# this is the force, in reduced units. They are strings, since we only use these numbers to reference directory names
stspN=['0.0','20.6','26.7','30.8','32.9','34.9','37.0','39.0','41.1','43.2','45.2','47.3','49.3','51.4','61.7']
number_replicas=40

fig=plt.figure(figsize=(5*len(stspN),5))
ods=[]
pds=[]

for s in range(len(stspN)):
    ax2=fig.add_subplot(1,len(stspN),s+1)
    print("Analyzing simulations with force of {} pN".format(stspN[s]))
    od=[]
    pd=[]
    output = Parallel(n_jobs=THREADS)(delayed(analyzerep)(repid,False) for repid in range(number_replicas))
    print("")
    output2 = Parallel(n_jobs=THREADS)(delayed(analyzerep)(repid,True) for repid in range(number_replicas))
    print("")
    for repid in range(number_replicas):
        # plot (-) trace
        time=output[repid][0]
        dist=output[repid][1]
        completed=0
        cli=np.where(np.array(dist)<6)[0]
        if len(cli) > 0:
            # at least one frame made it to the post conformation before 600 us
            if time[cli[0]] < 600:
                completed=1
        od.append(completed)
        ax2.plot(time,dist,color='k')

        # plot (+) trace
        time2=output2[repid][0]
        dist2=output2[repid][1]
        completed2=0
        cli2=np.where(np.array(dist2)<6)[0]
        if len(cli2) > 0:
            # at least one frame made it to the post conformation before 600 us
            if time2[cli2[0]] < 600:
                completed2=1
        pd.append(completed2)
        ax2.plot(time2,dist2,color='#cd3f96ff')
        ax2.legend(['(-) CV3-25','(+) CV3-25'],loc='upper right')

    ax2.set_ylabel('TM-HR1 distance (nm)')
    ax2.set_title('Applied Force: '+stspN[s]+' pN')
    ax2.set_xlabel('Simulated time ($\mu s$)')
    ax2.set_xlim([0,600])
    ax2.set_ylim([0,30])

    ods.append((np.sum(od))/len(od))
    pds.append((np.sum(pd))/len(pd))
    
plt.tight_layout()
plt.savefig('TM-HR1_distance_vs_time.png')
plt.savefig('TM-HR1_distance_vs_time.svg')

plt.figure()
sts=np.array(stspN).astype(float)
plt.plot(sts,ods,'ko-')
plt.plot(sts,pds,'o-',color='#cd3f96ff')
plt.ylim([0,1.0])
plt.xticks(sts,rotation=90)
plt.xlabel('Applied force (pN)')
plt.legend(['(-) CV3-25','(+) CV3-25'],loc='upper right')
plt.ylabel('Fraction of \nsuccessful transitions')
plt.savefig('fraction_of_successful_transitions.png')
plt.savefig('fraction_of_successful_transitions.svg')

sts=sts[3:-1]
ods=ods[3:-1]
pds=pds[3:-1]
plt.figure()
plt.plot(sts,ods,'ko-')
plt.plot(sts,pds,'o-',color='#cd3f96ff')
plt.ylim([0,1.0])
plt.xticks(sts,rotation=90)
plt.legend(['(-) CV3-25','(+) CV3-25'],loc='upper right')
plt.xlabel('Applied force (pN)')
plt.ylabel('Fraction of \nsuccessful transitions')
plt.savefig('fraction_of_successful_transitions_sub.png')
plt.savefig('fraction_of_successful_transitions_sub.svg')

