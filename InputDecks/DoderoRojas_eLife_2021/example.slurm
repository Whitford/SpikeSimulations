#!/bin/bash -l

#SBATCH --job-name=sim-wg-tmb
#SBATCH --partition=short
#SBATCH --constraint="E5-2680v4@2.40GHz"
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=7
#SBATCH --threads-per-core=1
#SBATCH --mem-per-cpu=2G
#SBATCH --export=ALL
#SBATCH --time=23:55:00
#SBATCH --mail-type=END
#SBATCH --exclusive

# load compilers used to build gromacs
module load gcc/6.4.0  openmpi/3.1.2  cmake/3.10.0

#Simulation set up
#ED - explain what cutoff is
CUTOFF=8
echo "$SLURM_ARRAY_TASK_ID"

# location of the gromacs executable. In this case, we are using an mpi-enabled version
# Important: you must use the modified (unofficial) version of Gromacs that is provided in the SpikeSim repo
GMX_EXE={PATH_TO_GROMACS}/bin/gmx_mpi

# Where to find the input files.  In this example, they are in the same directory that we are launching the job from
FILES_DIR='./'

# Create files related to the host membrane potential located at height=95 (~60nm from viral membrane)
[ ! -d "height_95" ] && mkdir height_95
cd height_95
height="95.000";
sed 's/HOST_MEMBRANE/'${height}'/g' ${FILES_DIR}/moveable_host_membrane.gro > rr_95.gro

mkdir rr_${SLURM_ARRAY_TASK_ID}
cd rr_${SLURM_ARRAY_TASK_ID}
cp ${FILES_DIR}/res.itp ./

##Energy minimization
srun -n 1 $GMX_EXE grompp -f ${FILES_DIR}/steep.mdp -p ${FILES_DIR}/postfusion_with_glycans_pre_TM.top -c ${FILES_DIR}/prefusion_with_glycans_bb.gro -o run.tpr -maxwarn 1 -r ${FILES_DIR}/height_95/rr_95.gro -n ${FILES_DIR}/index.ndx
srun $GMX_EXE mdrun -s run.tpr -v -maxh 24 -ntomp 8
mv confout.gro prefusion_with_glycans_bb_${SLURM_ARRAY_TASK_ID}.gro
rm run.tpr traj_comp.xtc traj.trr 

##SBM simulation - run for up to 1 hour. We are going to use 7 OMP threads and a single mpi rank
srun -n 1 $GMX_EXE grompp -f ${FILES_DIR}/base.mdp -p ${FILES_DIR}/postfusion_with_glycans_pre_TM.top -c prefusion_with_glycans_bb_${SLURM_ARRAY_TASK_ID}.gro -o run.tpr -maxwarn 1 -r ${FILES_DIR}/height_95/rr_95.gro -n ${FILES_DIR}/index.ndx
srun $GMX_EXE mdrun -s run.tpr -v -maxh 1 -ntomp 7 -dd 1 1 1 -nsteps -1

mv traj_comp.xtc traj_${SLURM_ARRAY_TASK_ID}.xtc 
mv traj.trr traj_${SLURM_ARRAY_TASK_ID}.trr 

# Compute distance between TM and HG to decide whether the transition finished 
srun -n 1 $GMX_EXE distance -f traj_${SLURM_ARRAY_TASK_ID}.xtc -n ${FILES_DIR}/index.ndx -oxyz -select 'cog of group "TM" plus cog of group "HG"'
awk '{print $1, ($3*$3 + $2*$2)^(1/2), $4}' distxyz.xvg > distrz.xvg
rd=`tail -q -n1 distrz.xvg | awk '{print $2}'`

# If transition has not completed, automatically submit a continuation run

m=`echo $rd'>'$CUTOFF | bc -l`
echo $rd $m $CUTOFF > status_run_${SLURM_ARRAY_TASK_ID}.dat

if [ "$m" -ne "1" ]
then
	jb1=$(sbatch ${FILES_DIR}/resub.slurm -height 95 -rep_id ${SLURM_ARRAY_TASK_ID} -fd ${FILES_DIR} | awk '{print $4}'); 
	echo $jb1 > jobs.txt
	mv traj_${SLURM_ARRAY_TASK_ID}.xtc traj_comp.xtc 
	mv traj_${SLURM_ARRAY_TASK_ID}.trr traj.trr 
fi
