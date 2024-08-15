########################################################
# SASA_1.4A_calculations_plot_only.py is a script that 
# will plot SASA values from simulations of the Spike 
# protein.
#
# Written by Esteban Dodero-Rojas and edited by Paul
# Whitford. This analysis is described in Grunst et al,
# Science, 2024.

# Prerequisites: python3, and the following python
# libraries: mdtraj, numpy, matplotlib and joblib
#
# To run this analysis, you should download the set of
# simulated trajectories that are described in 
# Dodero-Rojas, Onuchic and Whitford, eLife, 2021. The
# data directory may be downloaded from DRYAD:
# DOI: 10.5061/dryad.ncjsxkt3h
# 
# After downloading the full set of data (directory 
# SpikeSimData), run this script from within SpikeSimData,
# using the command:
# >python SASA_1.4A_calculations_plot_only.py
#
# Note: it is assumed that you have already run
#   SASA_1.4A_calculations.py
#
# The following files will be generated:
#   SASA_CV3-CC25_per_ensemble.png
#   SASA_CV3-CC25_per_ensemble.svg
#   SASA_woglycans_CV3-CC25_per_ensemble.png
#   SASA_woglycans_CV3-CC25_per_ensemble.svg
#   SASA_time_sampled_by_each_ensemble.png
#   SASA_time_sampled_by_each_ensemble.svg
#   SASA_times-summary.txt
#
########################################################

import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
import pickle

# If you would like to run this script from somewhere other than the directory that contains the trajectories, you can change the value of PATH_TO_TRAJECTORIES
PATH_TO_TRAJECTORIES='./trajectories_without_force_Dodero-Rojas_eLife_2021/'
# DOWNSAMPLE allows one to skip some frames during analysis. 1=all frames, N=analyze every Nth frame
DOWNSAMPLE=1
# THREADS is the number of parallel threads to use, when possible
THREADS=64

def plotSingleRun():
    singlerun=1
    trj_all=md.load(PATH_TO_TRAJECTORIES+'/traj_'+str(singlerun)+'.xtc',
                    top=PATH_TO_TRAJECTORIES+'/prefusion_with_glycans_bb_0.gro',stride=10*DOWNSAMPLE)
    
    # sasa has SASA values for all residues, including glycans
    sasa = md.shrake_rupley(trj_all,mode='residue')
    
    # psasa has the SASA values for only the protein residues
    psasa=sasa[:,:1584].reshape((len(sasa),3,-1))
    psasa.shape
    m=1
    
    # create image of time traces and save
    plt.plot(trj_all.time/1000,np.sum(psasa[:,m,indxCC25_106-S1offset],axis=1),color='#5164aeff')
    plt.plot(trj_all.time/1000,np.sum(psasa[:,m,indxCV3_25-S1offset],axis=1),color='#cd3f96ff')
    plt.plot(trj_all.time/1000,np.sum(sasa[:,1584:1591],axis=1),color='k')
    plt.xlim([0,24.0])
    plt.legend(['CC25.106','CV3-25','709N-glycan'])
    plt.ylabel('SASA (nm^2)')
    plt.xlabel('Time (us)')
    plt.savefig('SASA_epitopes.svg')
    plt.savefig('SASA_epitopes.png')
    # Plot the epitopes' amino acids SASA along one trajectory
    fig=plt.figure(figsize=(10,10))
    for m in range(3):
        ax = fig.add_subplot(3,1,m+1)
        indx=np.array(list(range(1148,1166)))
        max=ax.matshow(psasa[:,m,indx-S1offset].T,cmap='Reds', aspect='auto')
        ax.set_yticks(list(range(len(indx))),indx)
        ax.set_xticks(list(range(len(trj_all.time)))[::int(40/DOWNSAMPLE)],(trj_all.time[::int(40/DOWNSAMPLE)]/1000).astype(int))
    
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(max, cax=cbar_ax, label='SASA (nm^2)')
    plt.savefig('SASA_single_trajectory.png')
    plt.savefig('SASA_single_trajectory.svg')

def getFramesInEnsembles(runID,ensemble=None):
    """ Find the simulated frames that are consistent with each ensemble
    :param runID (int): numerical id of xtc file being analyzed (traj_<runID>.xtc)
    :param ensemble (str): which ensemble should be analyzed (default: None)
    :return 
         if ensemble==None: numpy array: [number of analyzed frames in xtc file (after downsampling), num of frames in ensemble 1, # in ensemble 2, # in ensemble 3]
         if ensemble defined: a trajectory object that only contains the frames of this xtc that were found to correspond to the specified ensemble
    """

    XTC=PATH_TO_TRAJECTORIES+'/traj_'+str(runID)+'.xtc'
    if ensemble ==  None:
        print("Analyzing {}".format(XTC), end='\r')
    else:
        print("Analyzing ensemble \"{}\" for {}".format(ensemble,XTC), end='\r')

    try:

        trj_all=md.load(XTC,top=PATH_TO_TRAJECTORIES+'/prefusion_with_glycans_bb_0.gro',stride=DOWNSAMPLE)

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

        if ensemble == None:
            # Count the number of frames that pass filter based on TM-HG-HR1 angle, HG-HR1 distance and HR1-TM distance
            # based on thresholds obtained from experimental Cryo-ET densities

            counts=np.empty(3,dtype=int)
            times=np.empty(3,dtype=object)
            for i in range(3):
                frames=(EnsembleDefs[dens_name[i]][0][0]<angle)      *(EnsembleDefs[dens_name[i]][0][1]>angle) \
                      *(EnsembleDefs[dens_name[i]][1][0]<dist_HG_HR1)*(EnsembleDefs[dens_name[i]][1][1]>dist_HG_HR1) \
                      *(EnsembleDefs[dens_name[i]][2][0]<dist_TM_HR1)*(EnsembleDefs[dens_name[i]][2][1]>dist_TM_HR1) 
                counts[i]= np.sum( frames)
                times[i]= trj_all.time[frames]/1000
            countarray=np.array([len(angle),counts[0],counts[1],counts[2],0],dtype=object)
            timearray=np.array([len(angle),times[0],times[1],times[2],0],dtype=object)
            return countarray, timearray
        else:

            # Filter frames based on TM-HG-HR1 angle, HG-HR1 distance and HR1-TM distance
            # based on thresholds obtained from experimental Cryo-ET densities
            ensembleframes= \
             (EnsembleDefs[ensemble][0][0]<angle)      *(EnsembleDefs[ensemble][0][1]>angle) \
            *(EnsembleDefs[ensemble][1][0]<dist_HG_HR1)*(EnsembleDefs[ensemble][1][1]>dist_HG_HR1) \
            *(EnsembleDefs[ensemble][2][0]<dist_TM_HR1)*(EnsembleDefs[ensemble][2][1]>dist_TM_HR1)
        
            return trj_all[ensembleframes]
 
    except: 
        print("ERROR: UNABLE TO ANALYZE {}\n".format(XTC))
        sys.exit(1)

def plotTimes():
    # Compute the fraction of transitions where each density map was sampled 
    nframes=np.sum(results[results[:,-1]==0,0])
    ntraj=np.sum(results[:,-1]==0)
    outFile = open('SASA_time-summary.txt', 'w')
    
    print('density map,\t# of matched structures,\tpercent of sampled conformations,\tpercent of simulated trajectories',file=outFile)
    for k in range(1,4):
        print("\"{}\"".format(dens_name[k-1]),np.round(np.sum(results[results[:,-1]==0,k]),2), \
                             np.round(np.sum(results[results[:,-1]==0,k])/nframes*100,1), \
                             np.round(np.sum(results[results[:,-1]==0,k]>0)/ntraj,2), \
                             file=outFile)
    
    outFile.close()
    
    times=[]
    for d in range(1,4):
        t=[]
        for i in range(len(extracted_times)):
            if len(extracted_times[i][d])!=0:
                t.append(extracted_times[i][d])
        times.append(np.concatenate(t))
    # this is just for cosmetic purposes when making violin plots
    hti=np.where(dens_name=="high tilt")[0][0]
    times[hti]=times[hti][times[hti]<60]
    
    pos=[1,2,3]
    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(2,1,1)
    ax.violinplot(times,pos,showmeans=True,showmedians=False,showextrema=False)
    ax.set_xticks(pos,dens_name,rotation=45)
    ax.set_ylabel('Time (us)')
    plt.tight_layout()
    plt.savefig('SASA_time_sampled_by_each_ensemble.png')
    plt.savefig('SASA_time_sampled_by_each_ensemble.svg')

def getSingleSASA(trj):
    sasa = md.shrake_rupley(trj,mode='residue')
    psasa=sasa[:,:1584].reshape((len(sasa),3,-1))
    vvv1=np.concatenate([np.sum(psasa[:,m,indxCC25_106-S1offset],axis=1) for m in range(3)])
    vvv2=np.concatenate([np.sum(psasa[:,m,indxCV3_25-S1offset],axis=1) for m in range(3)])
    return vvv1,vvv2

def doSASA():
    dens_entries=[]
    dens_entries_p=[]
#    for k in [1,2,3]:
#        ddict   = []
#        ddict_p = []
#        outputs = Parallel(n_jobs=THREADS)(delayed(distributedSASA)(xtcs[m],k) for m in np.where(results[:,k]>0)[0])
#        # store CC25 - including glycans
#        dens_entries.append(np.concatenate([outputs[i][0] for i in range(len(outputs))]))
#        # store CV3 -  including glycans
#        dens_entries.append(np.concatenate([outputs[i][1] for i in range(len(outputs))]))  
#        # store CC25 - glycans not used in SASA evaluations
#        dens_entries_p.append(np.concatenate([outputs[i][2] for i in range(len(outputs))]))  
#        # store CV3 - glycans not used in SASA evaluations
#        dens_entries_p.append(np.concatenate([outputs[i][3] for i in range(len(outputs))]))  
#
#    # save binary files that can be used to skip the above lines. 
#    with open("SASA_step2data.pkl", "wb") as fp: 
#        pickle.dump(dens_entries, fp)
#    with open("SASA_step2pdata.pkl", "wb") as fp: 
#        pickle.dump(dens_entries_p, fp)
  # read previous analysis
    with open("SASA_step2data.pkl", "rb") as fp:
        dens_entries=pickle.load(fp)
    with open("SASA_step2pdata.pkl", "rb") as fp:
        dens_entries_p=pickle.load(fp)

    pos=[1,2,4,5,7,8]
    pos2=[1,1.5,2,4,4.5,5,7,7.5,8]
    xtics=np.array(['CC25.106',dens_name[0],'CV3-25','CC25.106',dens_name[1],'CV3-25','CC25.106',dens_name[2],'CV3-25'],dtype=str)
    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(2,1,1)
    ax.violinplot(dens_entries[::2],pos[::2],showmeans=True,showmedians=False,showextrema=False)
    ax.violinplot(dens_entries[1::2],pos[1::2],showmeans=True,showmedians=False,showextrema=False)
    ax.set_xticks(pos2,xtics,rotation=45)
    ax.set_ylim([0,16])
    ax.set_ylabel('SASA (nm^2)')
    plt.tight_layout()
    plt.savefig('SASA_CV3-CC25_per_ensemble.png')
    plt.savefig('SASA_CV3-CC25_per_ensemble.svg')

    # make plots for -glycan SASA calculations
    fig=plt.figure(figsize=(5,5))
    ax = fig.add_subplot(2,1,1)
    ax.violinplot(dens_entries_p[::2],pos[::2],showmeans=True,showmedians=False,showextrema=False)
    ax.violinplot(dens_entries_p[1::2],pos[1::2],showmeans=True,showmedians=False,showextrema=False)
    ax.set_xticks(pos2,xtics,rotation=45)
    ax.set_ylim([0,16])
    ax.set_ylabel('SASA (nm^2)')
    plt.tight_layout()
    plt.savefig('SASA_woglycans_CV3-CC25_per_ensemble.png')
    plt.savefig('SASA_woglycans_CV3-CC25_per_ensemble.svg')

def distributedSASA(xtc,k):
    trj=getFramesInEnsembles(xtc,dens_name[k-1])
    # calculate and save SASA values, calculated based on full system (including glycans)
    vals=getSingleSASA(trj)

    # calculate and save SASA values, calculated based on only the protein (not including glycans)
    trj=trj.atom_slice(range(12276))
    valsp=getSingleSASA(trj)
    return vals[0], vals[1],valsp[0], valsp[1]
     


#################### MAIN PROGRAM #################### 


# DEFINE INFORMATION ABOUT THE EM MAPS  

# dens_name is a list of codes for the different EM densities considered. In relation to Grunst et al, the naming is:
dens_name=np.array(['low tilt','medium tilt','high tilt'])

# EnsembleDefs defines the upper and lower bounds on the angle, HG-HR1 distance and TM-HR1 distance used to identify simulated models that are within each ensemble
# format: EnsembleDefs[entryname]= [[min TM-HG-HR1 angle, max TM-HG-HR1 angle ][min HG_HR1 distance, max HG_HR1 distance][min TM_HR1 distance, max TM_HR1 distance]]

EnsembleDefs = {}
EnsembleDefs["low tilt"]=[[0,25] ,[15,18],[24.5,30.4]]
EnsembleDefs["medium tilt"]=[[30,60],[15,18],[23.4,27.3]]
EnsembleDefs["high tilt"]=[[60,80],[15,18],[20.8,24.0]]

# indxCV3_25 contains the residue indexes (full length protein numbering) of the CV3-25 epitope
indxCV3_25=np.array([1153, 1156, 1157, 1159, 1160, 1161, 1162, 1163])

# indxCC25_106 contains the residue indexes (full length protein numbering) of the CC25.106 epitope
indxCC25_106=np.array([1148, 1151, 1152, 1153, 1155, 1156, 1158])

# These simulations only include the S2 subunit. S1offset maps full-length protein numbering to the numbering used in the simulations and analysis. 
# In our simulation, the first residue (gro number 1, mdtraj internally numbered 0) corresponds to residue 706 in the full length system
S1offset=706

## DEFINE REGIONS OF THE SPIKE
# TM atom indexes 
TM1_index = list(range(3928,4092))
TM2_index = list(range(8020,8184))
TM3_index = list(range(12112,12276))
TM_index=np.concatenate([TM1_index,TM2_index,TM3_index])
# HG atom indexes 
HG1_index = list(range(2490,3255))
HG2_index = list(range(6582,7347))
HG3_index = list(range(10674,11439))
HG_index=np.concatenate([HG1_index,HG2_index,HG3_index])
# HR1 atom indexes
HR11_index = list(range(9832-2*4092,9909-2*4092))
HR12_index = list(range(9832,9909))
HR13_index = list(range(9832-4092,9909-4092))
HR1_index=np.concatenate([HR11_index,HR12_index,HR13_index])
# FP atom indexes
FP1_index = range(849,1148)
FP2_index = range(4941,5240)
FP3_index = range(9034,9332)
FP_index = np.concatenate([FP1_index,FP2_index,FP3_index])

#print ("Creating a time trace figure for a single trajectory")
#plotSingleRun()

# open file that contains indexes of trajectories to analyze 
#f=open(PATH_TO_TRAJECTORIES+'/listofxtcs',"r")
#xtcs=[]
#for line in f:
#    xtcs.append(int(line))


# DO THE MAIN CALCULATIONS
#print('Processing files, step 1\n\n')
# Step 1 identifies which trajectories sample each ensemble
#outputs = Parallel(n_jobs=THREADS)(delayed(getFramesInEnsembles)(i) for i in xtcs)
#outputs=np.array(outputs)
# Save a binary file that contains all output to this point.  This can be useful for additional post-processing purposes. Or, if you want to avoid repeating the above call to getFramesInEnsembles, then just comment out the two lines above and uncomment the line below.
#with open("SASA_step1data.pkl", "wb") as fp: 
#    pickle.dump(outputs, fp)
with open("SASA_step1data.pkl", "rb") as fp: 
    outputs=pickle.load(fp)

results = np.array([outputs[i][0] for i in range(len(outputs))])
extracted_times = np.array([outputs[i][1] for i in range(len(outputs))])

######### get time plots
plotTimes()

#print('\n\nProcessing files, step 2\n\n')
# Step 2 calculates the SASA values for each ensemble

doSASA()
