#!/bin/bash

M=100
L=10
for metric in ERR NDCG
do
	for ensemble in 100 500 1000
	do
		for lr in 0.05 0.3 0.5
		do
			for subsample in 0.3 0.5 1.0
			do
                for leaves in 10 40 70
                do
					python Optimization.py -r gbrt -g True -v ${metric} -e ${ensemble} -t ${lr} -u ${subsample} -x ${leaves} &> opt.log.${metric}.${ensemble}.${lr}.${subsample}.${leaves} &
				done
			done
		done
	done
done
