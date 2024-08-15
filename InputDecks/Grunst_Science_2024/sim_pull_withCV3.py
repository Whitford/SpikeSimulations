from openmm.app import *
from openmm import *
from openmm.unit import *
import sys
from OpenSMOG import SBM
import mdtraj as md
import numpy as np


def add_extra_forces(sbm):
    print('***************************************************************')
    print('Loading other forces (TM and HG forces)')
    TM1_index = range(3928,4092)
    TM2_index = range(8020,8184)
    TM3_index = range(12112,12276)

    viral_membranepot = CustomExternalForce("step(-wm/2 + mem -z)*k*(1 - exp((-(z + (wm/2) - mem)^2)/(2*sigma^2))) + step(z - wm/2 - mem)*k*(1 - exp((-(z - (wm/2) - mem)^2)/(2*sigma^2)))")
    viral_membranepot.addPerParticleParameter("wm")
    viral_membranepot.addPerParticleParameter("mem")
    viral_membranepot.addPerParticleParameter("sigma")
    viral_membranepot.addPerParticleParameter("k")

    for i in TM1_index:
        viral_membranepot.addParticle(int(i),[5,35,0.2,5])
    for i in TM2_index:
        viral_membranepot.addParticle(int(i),[5,35,0.2,5])
    for i in TM3_index:
        viral_membranepot.addParticle(int(i),[5,35,0.2,5])
    sbm.system.addForce(viral_membranepot)

    #Centroid force between TM domains to avoid dissociation of the TM complex
    for pair in [[TM1_index,TM2_index],[TM1_index,TM3_index],[TM2_index,TM3_index]]:
        centroid_TMpot =  CustomCentroidBondForce(2,"0.5*20*distance(g1,g2)^2")
        centroid_TMpot.addGroup(pair[0])
        centroid_TMpot.addGroup(pair[1])
        centroid_TMpot.addBond([0, 1])
        sbm.system.addForce(centroid_TMpot)

    # HG membrane potential
    HG1_index = range(2490,3255)
    HG2_index = range(6582,7347)
    HG3_index = range(10674,11439)

    HGviral_membranepot = CustomExternalForce("step(vmem-z)*vk*(vmem-z)^2")
    HGviral_membranepot.addPerParticleParameter("vmem")
    HGviral_membranepot.addPerParticleParameter("vk")

    for i in HG1_index:
        HGviral_membranepot.addParticle(int(i),[37.5,2])
    for i in HG2_index:
        HGviral_membranepot.addParticle(int(i),[37.5,2])
    for i in HG3_index:
        HGviral_membranepot.addParticle(int(i),[37.5,2])
    sbm.system.addForce(HGviral_membranepot)
    print('***************************************************************')

    #Position restraints for heavy atoms
    #Find heavy atoms
    gro = GromacsGroFile("./sim_with_CV3.gro")
    ref = md.load("./sim_with_CV3.gro")

    heavy = ref.topology.select_atom_indices('heavy')
    pos=gro.positions
    positions = np.zeros([len(pos), 3]) * nanometer
    for idx, p_i in enumerate(pos):
        if idx < 15207:
            positions[idx, :] = p_i
    restraint_force = CustomExternalForce('0.5*(K_pr)*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)')
    # Adding the spring constant as a global parameter allows us to turn it off if desired
    restraint_force.addGlobalParameter('K_pr', 1000)
    restraint_force.addPerParticleParameter('x0')
    restraint_force.addPerParticleParameter('y0')
    restraint_force.addPerParticleParameter('z0')
    for index in heavy:
        if int(index) < 15207:
            parameters = positions[index, :].value_in_unit_system(md_unit_system)
            restraint_force.addParticle(int(index), parameters)
    sbm.system.addForce(restraint_force)

    # HR1 index
    HR11_index = list(range(9832-2*4092,9909-2*4092))
    HR12_index = list(range(9832,9909))
    HR13_index = list(range(9832-4092,9909-4092))
    HR1_index=np.concatenate([HR11_index,HR12_index,HR13_index])
    pull_force = CustomExternalForce('-PS*z')
    pull_force.addGlobalParameter('PS', 0)
    #Centroid force between TM domains to avoid dissociation of the TM complex
    for index in HR1_index:
        pull_force.addParticle(int(index))
    sbm.system.addForce(pull_force)

    return 0

#Simulation ID
SID = sys.argv[1]
# Force in pN
F_pN = float(sys.argv[2])

print("Simulation ID: " ,SID,' - Force: ',F_pN)
sbm = SBM(name='spike_protein', time_step=0.002, collision_rate=1.0, r_cutoff=0.65, temperature=70*0.00831446261815324, cmm=False)

sbm.setup_openmm(platform='hip',GPUindex='default')
sbm.saveFolder('output_'+str(F_pN)+'pN'+'_'+str(SID))

sbm_grofile = './sim_with_CV3.gro'
sbm_topfile = './sim_with_CV3.top'
sbm_xmlfile = './sim_with_CV3.xml'

sbm.loadSystem(Grofile=sbm_grofile, Topfile=sbm_topfile, Xmlfile=sbm_xmlfile)

#Change contact interactions to harmonic interactions for CV3-25 contacts
f=sbm.system.getForce(8)
print("Energy force 8:",f.getEnergyFunction(),f.getName())
f.setEnergyFunction("0.5*HS*(r-r0)^2")
f.addGlobalParameter('HS', 1)
print("Changing 8 to",f.getEnergyFunction(),f.getName())

# Add TM and HG forces (viral membrane) and position restraints for the begining of the simulation
add_extra_forces(sbm)

sbm.createSimulation()
sbm.minimize(tolerance=.001)

# Add reporters
interval=10**5
sbm.createReporters(trajectory=False, energies=True, energy_components=True, interval=interval)
sbm.simulation.reporters.append(StateDataReporter(sys.stdout, interval, potentialEnergy=True, step=True, speed=True, separator="\t"))
sbm.simulation.reporters.append(md.reporters.XTCReporter(sbm.folder+'/trajectory.xtc', interval))
sbm._createLogfile()

#Short simulation to allow relaxation of harmonic network of CV3-25 antibody
secs=10
for s in np.linspace(1, 1000, num=10):
    print('Spring const of harmonic potential: ',s)
    sbm.simulation.context.setParameter('HS',s)
    sbm.minimize(tolerance=1)
    sbm.simulation.runForClockTime(secs/3600)

print('***************************************************************')
print('Removing position restraints on Spike')
print('***************************************************************')
#Short simulation to release position restraints on Spike Protein
for s in np.linspace(1000,0,num=10):
    print('Position restraints strength k: ',s)
    sbm.simulation.context.setParameter('K_pr',s)
    sbm.minimize(tolerance=1)
    sbm.simulation.runForClockTime(secs/3600)

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

dist_HG_HR1=0
# conversion factor between total force, in pN, and force on each atom, in reduced units
a=(231 * 310/70 * 1.661)

for m in range(100):
    if dist_HG_HR1==0:
        print('HR1 has not formed at ',m)
        trj_all=md.load(sbm.folder+'/trajectory.xtc',top='./sim_with_CV3.gro')
        sbm.simulation.runForClockTime(1/60)
        CM_HG=np.mean(trj_all.xyz[:,HG_index,:],axis=1)
        CM_HR1=np.mean(trj_all.xyz[:,HR1_index,:],axis=1)
        dist_HG_HR1=np.sum(np.linalg.norm(CM_HG-CM_HR1,axis=1)>15)
    else:
        st=F_pN/a
        print('***************************************************************')
        print('HG-HR1 reached 15nm - Introducing pull force')
        print('***************************************************************')
        sbm.simulation.context.setParameter('PS',st)
        sbm.minimize(tolerance=1)
        break

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

#Using distance between TM and HR1 to assess whether the full transition happens
dist_TM_HR1=0
while dist_TM_HR1==0:
    print('No transition yet')
    sbm.simulation.runForClockTime(time=0.5,checkpointFile=sbm.folder+'/chkp.ck')
    trj_all=md.load(sbm.folder+'/trajectory.xtc',top='./sim_with_CV3.gro')
    CM_TM=np.mean(trj_all.xyz[:,TM_index,:],axis=1)
    CM_HR1=np.mean(trj_all.xyz[:,HR1_index,:],axis=1)
    dist_TM_HR1=np.sum(np.linalg.norm(CM_TM-CM_HR1,axis=1)<5)

print('Finish simulation')
