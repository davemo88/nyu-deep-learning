#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -l walltime=10:00:00
#PBS -l mem=30GB
#PBS -N dk-rnn
module purge

module load torch-deps/7
module load torch/intel/20151009

cd /home/dk2353/4/materials/lstm
th main.lua

#torque can lose the last line apparently
