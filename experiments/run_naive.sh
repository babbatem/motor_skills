#!/bin/bash

#SBATCH --time=4:00:00

#SBATCH -N 1
#SBATCH -c 8
#SBATCH -J naive-learn
#SBATCH --mem=8G

#SBATCH -o naive-learn-%j.out
#SBATCH -e naive-learn-%j.out

cd /users/babbatem/
source .bashrc
source load_mods.sh
cd motor_skills
python3 my_job_script.py --output experiments/naive/dev --config experiments/naive-learning-config.txt
