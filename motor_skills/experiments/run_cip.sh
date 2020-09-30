#!/bin/bash

#SBATCH --time=4:00:00

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -J cip-learn
#SBATCH --mem=8G

#SBATCH -o cip-learn-%j.out
#SBATCH -e cip-learn-%j.out

cd /users/babbatem/
source .bashrc
source load_mods.sh

cd motor_skills
python3 my_job_script.py --output experiments/cip/dev --config experiments/cip-learning-config.txt
