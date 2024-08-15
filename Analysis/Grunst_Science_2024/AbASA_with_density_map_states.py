########################################################
# AbASA_with_density_map_states.py is a script that will 
# calculate AbASA values from a simulation of the Spike 
# protein.
#
# Written by Esteban Dodero-Rojas and edited by Paul
# Whitford. This analysis is described in Grunst et al,
# Science, 2024.

# Prerequisites: python3, and the following python
# libraries: mdtraj, numpy and matplotlib
#
# To run this analysis, you should download the set of
# simulated trajectories that are described in 
# Dodero-Rojas, Onuchic and Whitford, eLife, 2021. The
# data directory may be downloaded from DRYAD:
# DOI: 10.5061/dryad.ncjsxkt3h
# 
# After downloading the full set of data (directory 
# SpikeSimData), run this script from within 
# SpikeSimData, using the command:
# >python AbASA_with_density_map_states.py
#
# The following files will be generated:
#   AbASA_epitope_trajectory_run_<runIndex>_pr_<radius>_with_states.png
#   AbASA_epitope_trajectory_run_<runIndex>_pr_<radius>_with_states.svg
#   AbASA_epitope_trajectory_run_<runIndex>_pr_<radius>+N709_with_states.png
#   AbASA_epitope_trajectory_run_<runIndex>_pr_<radius>+N709_with_states.svg
#   <runIndex> (default: 286) is the index of the 
#   simulation used to generate the plot and <radius> 
#   is the radius of the probe used to calculate the 
#   AbASA values (default: 7.2 Angstrom) 
#   +N709 means the AbASA of N709 glycan is shown
#   for reference 
#
########################################################

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt

PATH_TO_TRAJECTORIES='./trajectories_without_force_Dodero-Rojas_eLife_2021/'

# dens_name is a list of codes for the different EM densities considered. In relation to Grunst et al, the naming is:
dens_name=np.array(['low tilt','medium tilt','high tilt'])

# EnsembleDefs defines the upper and lower bounds on the angle, HG-HR1 distance and TM-HR1 distance used to identify simulated models that are within each ensemble
# format: EnsembleDefs[entryname]= [[min TM-HG-HR1 angle, max TM-HG-HR1 angle ][min HG_HR1 distance, max HG_HR1 distance][min TM_HR1 distance, max TM_HR1 distance]]

EnsembleDefs = {}
EnsembleDefs["low tilt"]=[[0,25] ,[15,18],[24.5,30.4]]
EnsembleDefs["medium tilt"]=[[30,60],[15,18],[23.4,27.3]]
EnsembleDefs["high tilt"]=[[60,80],[15,18],[20.8,24.0]]

# TM index 
TM1_index = list(range(3928,4092))
TM2_index = list(range(8020,8184))
TM3_index = list(range(12112,12276))
TM_index=np.concatenate([TM1_index,TM2_index,TM3_index])
# HG index 
HG1_index = list(range(2490,3255))
HG2_index = list(range(6582,7347))
HG3_index = list(range(10674,11439))
HG_index=np.concatenate([HG1_index,HG2_index,HG3_index])
# HR1 index
HR11_index = list(range(9832-2*4092,9909-2*4092))
HR12_index = list(range(9832,9909))
HR13_index = list(range(9832-4092,9909-4092))
HR1_index=np.concatenate([HR11_index,HR12_index,HR13_index])
# FP index
FP1_index = range(849,1148)
FP2_index = range(4941,5240)
FP3_index = range(9034,9332)
FP_index = np.concatenate([FP1_index,FP2_index,FP3_index])

def get_times(trj_all):
    # get_times:
    # Identify the frames that satisfy the conditions
    # for each density map 

    # Compute center of mass of TM, HG and HR1 motifs
    CM_TM=np.mean(trj_all.xyz[:,TM_index,:],axis=1)
    CM_HG=np.mean(trj_all.xyz[:,HG_index,:],axis=1)
    CM_HR1=np.mean(trj_all.xyz[:,HR1_index,:],axis=1)

    # Compute distance between HG and HR1, and TM and HR1
    dist_HG_HR1=np.linalg.norm(CM_HG-CM_HR1,axis=1)
    dist_TM_HR1=np.linalg.norm(CM_TM-CM_HR1,axis=1)

    # Compute vectors between HG and HR1, and TM and HR1
    vect_TM_HG=CM_TM-CM_HG
    vect_HG_HR1=CM_HG-CM_HR1
    vect_TM_HG=vect_TM_HG/np.linalg.norm(vect_TM_HG,axis=1)[:,np.newaxis]
    vect_HG_HR1=vect_HG_HR1/np.linalg.norm(vect_HG_HR1,axis=1)[:,np.newaxis]

    # Compute angle between TM, HG and HR1
    vdot=np.sum(vect_TM_HG*vect_HG_HR1, axis=1)
    # fix for rounding issues
    ones=np.ones(len(vdot))
    vdot=np.minimum(vdot,ones)
    ones=ones*(-1)
    vdot=np.maximum(vdot,ones)
    angle=np.arccos(vdot)*180/np.pi

    times=np.empty(3,dtype=object)
    for i in range(3):
        frames=  (EnsembleDefs[dens_name[i]][0][0]<angle)      *(EnsembleDefs[dens_name[i]][0][1]>angle) \
                *(EnsembleDefs[dens_name[i]][1][0]<dist_HG_HR1)*(EnsembleDefs[dens_name[i]][1][1]>dist_HG_HR1) \
                *(EnsembleDefs[dens_name[i]][2][0]<dist_TM_HR1)*(EnsembleDefs[dens_name[i]][2][1]>dist_TM_HR1) 
        times[i]= trj_all.time[frames]/1000
    timearray=np.array([len(angle),times[0],times[1],times[2],0],dtype=object)
    return timearray
 
#Load one transition trajectory
probe_radius=7.2 # In Angstroms
runID=286
trj_all=md.load(PATH_TO_TRAJECTORIES+'/traj_'+str(runID)+'.xtc',
                top=PATH_TO_TRAJECTORIES+'/prefusion_with_glycans_bb_0.gro',stride=4)

sasa = md.shrake_rupley(trj_all,mode='residue',probe_radius=probe_radius/10)
glycan_sasa=sasa[:,1584:]
psasa=sasa[:,:1584].reshape((len(sasa),3,-1))
psasa.shape

times = get_times(trj_all)
alltimes=trj_all.time/1000
m=1

# make the plot with N709 glycan shown, for reference
ymaxv=24
plt.figure(figsize=(10,6))
#Plot CC25.106 AbASA
indx=np.array([1148, 1151, 1152, 1153, 1155, 1156, 1158])
plt.plot(alltimes,np.sum(psasa[:,m,indx-706],axis=1),linewidth=2,color='#5164aeff')
#Plot CV3-25 AbASA
indx=np.array([1153, 1156, 1157, 1159, 1160, 1161, 1162, 1163])
plt.plot(alltimes,np.sum(psasa[:,m,indx-706],axis=1),linewidth=2,color='#cd3f96ff')
#Plot N709 glycans AbASA
plt.plot(alltimes,np.sum(sasa[:,1584:1591],axis=1),color='k')
#Plot vertical lines for each frame where the structure matches each density map
plt.vlines(times[1],ymin=ymaxv-3,ymax=ymaxv,color='skyblue',linewidth=3)
plt.vlines(times[2],ymin=ymaxv-3,ymax=ymaxv,color='yellow',linewidth=3)
plt.vlines(times[3],ymin=ymaxv-3,ymax=ymaxv,color='red',linewidth=3)

plt.legend(['CC25.106','CV3-25','709N-glycan','Low tilt','Medium tilt','High tilt'],loc='upper left')
plt.ylabel('Epitope AbASA ($nm^2$)')
plt.ylim([0,ymaxv])
plt.xlim([0,18])
plt.xlabel('Time ($\mu s$)')
plt.savefig('AbASA_epitope_trajectory_run_'+str(runID)+'_pr_'+str(probe_radius)+'+N709_with_states.svg')
plt.savefig('AbASA_epitope_trajectory_run_'+str(runID)+'_pr_'+str(probe_radius)+'+N709_with_states.png')

# make the plot without N709 glycan shown
ymaxv=12
plt.figure(figsize=(10,6))
#Plot CC25.106 AbASA
indx=np.array([1148, 1151, 1152, 1153, 1155, 1156, 1158])
plt.plot(alltimes,np.sum(psasa[:,m,indx-706],axis=1),linewidth=2,color='#5164aeff')
#Plot CV3-25 AbASA
indx=np.array([1153, 1156, 1157, 1159, 1160, 1161, 1162, 1163])
plt.plot(alltimes,np.sum(psasa[:,m,indx-706],axis=1),linewidth=2,color='#cd3f96ff')
#Plot vertical lines for each frame where the structure matches each density map
plt.vlines(times[1],ymin=ymaxv-3,ymax=ymaxv,color='skyblue',linewidth=3)
plt.vlines(times[2],ymin=ymaxv-3,ymax=ymaxv,color='yellow',linewidth=3)
plt.vlines(times[3],ymin=ymaxv-3,ymax=ymaxv,color='red',linewidth=3)

plt.legend(['CC25.106','CV3-25','Low tilt','Medium tilt','High tilt'],loc='upper right')
plt.ylabel('AbASA ($nm^2$)')
plt.ylim([0,ymaxv])
plt.xlim([0,18])
plt.xlabel('Time ($\mu s$)')
plt.savefig('AbASA_epitope_trajectory_run_'+str(runID)+'_pr_'+str(probe_radius)+'_with_states.svg')
plt.savefig('AbASA_epitope_trajectory_run_'+str(runID)+'_pr_'+str(probe_radius)+'_with_states.png')


