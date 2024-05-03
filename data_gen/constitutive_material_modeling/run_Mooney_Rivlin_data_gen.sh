#!/bin/bash

#start docker 
for ((i=0;i<500;i++))
do
  echo "$i"
  python3 -u  incomp_2d_Mooney_Rivlin_dirichlet_ubc.py $i >>MR_data_gen_log.log
  wait
done
wait
