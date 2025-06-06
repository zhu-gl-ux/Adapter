#!/bin/bash

GPU=0,1,2
BS=8
SAVEDIR='saved_voc'

TASKSETTING='overlap'
TASKNAME='15-1'
INIT_LR=0.001
LR=0.0001
MEMORY_SIZE=0 # 50 for Adapter-M

NAME='Adapter'
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 0 --lr ${INIT_LR} --bs ${BS}

python train_voc.py -c configs/config_voc.json \
-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 1 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}

#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 2 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
##
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 3 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 4 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}
#
#python train_voc.py -c configs/config_voc.json \
#-d ${GPU} --multiprocessing_distributed --save_dir ${SAVEDIR} --name ${NAME} \
#--task_name ${TASKNAME} --task_setting ${TASKSETTING} --task_step 5 --lr ${LR} --bs ${BS} --freeze_bn --mem_size ${MEMORY_SIZE}


