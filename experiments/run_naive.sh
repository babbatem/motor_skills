#!/bin/bash

#SBATCH --time=4:00:00

#SBATCH -N 1
#SBATCH -c 4
#SBATCH -J naive-learn
#SBATCH --mem=8G

#SBATCH -o naive-learn-%j.out
#SBATCH -e naive-learn-%j.out

cd /users/babbatem/
source .bashrc
cd motor_skills
python my_job_script.py --output experiments/naive/%j --config experiments/naive-learning-config.txt
