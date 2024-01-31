#!/bin/bash
#SBATCH -t 3-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
##SBATCH -t 0-35:00
#SBATCH --gres=gpu:2
#SBATCH -n 1
#SBATCH -c 64
#SBATCH --account=iaifi_lab
#SBATCH -p iaifi_gpu_priority
##SBATCH --account=avillar_lab
##SBATCH -p itc_gpu
#SBATCH -o myoutput_64runNoParams_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_64runNoParams_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH -J vaeProf
#SBATCH --mem=0       # Memory pool for all cores (see also --mem-per-cpu)

# load modules
date
source ~/.bashrc
conda activate tf2.12_cuda11_sbi
# run code
#python trainVAE_wRotAngle.py 5
#python trainVAE_wRotAngle_mtadam.py 5
#python trainVAE_wRotAngle_KLdiv.py 5
#python trainVAE_noRot.py 4
#python trainVAE_uninformed.py 5
#python trainVAE_newArch.py 5
#srun python trainVAE_ssl.py 20
srun python trainVAE_noParams.py 20
date
