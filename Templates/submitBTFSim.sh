#!/bin/bash -f

# Created on 08.06.2023
# Edited on 27.09.2023
# Author: CCo

#SBATCH --partition=maxgpu,allgpu
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1      
#SBATCH --chdir=/beegfs/desy/user/cortesga/BTFSim/btf-simulation/{dirName}
#SBATCH --job-name=trackSim
#SBATCH --output=MAXOUT/btfSim-%j.out    # File to which STDOUT will be written
#SBATCH --error=MAXOUT/btfSim-%j.err     # File to which STDERR will be written
#SBATCH --mail-type=FAIL         # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=edgar.cristopher.cortes.garcia@desy.de  # Email to which notifications will be sent
#SBATCH --constraint=GPUx4

# Initialize module system
module load maxwell mamba/3.9
. mamba-init 
mamba activate DESY4

# Run BTF simulation
python3 run_btf.py
