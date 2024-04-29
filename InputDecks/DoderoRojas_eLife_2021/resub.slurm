#!/bin/bash -l

#SBATCH --job-name=rsim-wg-tmb
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
#SBATCH --mail-user=ed29@rice.edu
#SBATCH --exclusive


while test $# -gt 0; do
           case "$1" in
                -height)
                    shift
		    h=$1
                    shift
                    ;;	
		 -rep_id)
                    shift
		    rep_id=$1
                    shift
                    ;;
		 -fd)
                    shift
		    FILES_DIR=$1
                    shift
                    ;;
		*)
                   echo "$1 is not a recognized flag!"
                   return 1;
                   ;;
          esac
done

module load gcc/6.4.0  openmpi/3.1.2  cmake/3.10.0

#Simulation set up
CUTOFF=8
echo "$rep_id"
GMX_EXE=/home/e.doderorojas/BIN/gromacs_flat_GXYZ/bin/gmx_mpi


##SBM simulation
srun $GMX_EXE mdrun -s run.tpr -v -maxh 0.99 -ntomp 7 -dd 0 0 0  -nsteps -1 -cpi state.cpt

mv traj_comp.xtc traj_${rep_id}.xtc 
mv traj.trr traj_${rep_id}.trr 

#Distances
srun -n 1 $GMX_EXE distance -f traj_${rep_id}.xtc -n /scratch/e.doderorojas/index.ndx -oxyz -select 'cog of group "TM" plus cog of group "HG"'
awk '{print $1, ($3*$3 + $2*$2)^(1/2), $4}' distxyz.xvg > distrz.xvg
rd=`tail -q -n1 distrz.xvg | awk '{print $2}'`

#Extra submission if needed
m=`echo $rd'>'$CUTOFF | bc -l`
echo $rd $m $CUTOFF >> status_run_${rep_id}.dat
if [ "$m" -ne "1" ]
then
	jb1=$(sbatch ${FILES_DIR}/resub.slurm -height HEIGHT -rep_id ${rep_id} -fd ${FILES_DIR} | awk '{print $4}'); 
	echo $jb1 >> jobs.txt
	mv traj_${rep_id}.xtc traj_comp.xtc 
	mv traj_${rep_id}.trr traj.trr 
fi

