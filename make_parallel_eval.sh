#!/bin/bash

M=100
L=10

for metric in ERR NDCG
do
	for eval in tree lasso
	do
		for approach in OnPolicy IPS IPS_SN PI PI_SN
		do
			python Parallel.py -m ${M} -l ${L} -v ${metric} -e ${eval} -a ${approach} --start 0 --stop 25 &> eval.log.${metric}.${M}.${L}.${eval}.${approach} &
		done
	done
done

for metric in ERR NDCG
do
	for eval in tree lasso
	do
        for train in 1000 3000 10000 30000 100000 300000 1000000 3000000
        do
            for approach in DM_tree DM_lasso DMc_lasso
            do
                python Parallel.py -m ${M} -l ${L} -v ${metric} -e ${eval} -a ${approach} -z ${train} --start 0 --stop 25 &> eval.log.${metric}.${M}.${L}.${eval}.${approach}.${train} &
            done
        done
	done
done

for logger in tree lasso
do
    for temp in 0.5 1.0 1.5 2.0
    do
        for metric in ERR NDCG
        do
            for eval in tree lasso
            do
                for approach in OnPolicy IPS IPS_SN PI PI_SN
                do
                    python Parallel.py -m ${M} -l ${L} -v ${metric} -f ${logger} -e ${eval} -t ${temp} -a ${approach} --start 0 --stop 25 &> eval.log.${metric}.${M}.${L}.${logger}-${temp}.${eval}.${approach} &
                done
            done
        done
    done
done

for logger in tree lasso
do
    for temp in 0.5 1.0 1.5 2.0
    do
        for metric in ERR NDCG
        do
            for eval in tree lasso
            do
                for train in 1000 3000 10000 30000 100000 300000 1000000 3000000
                do
                    for approach in DM_tree DM_lasso DMc_lasso
                    do
                        python Parallel.py -m ${M} -l ${L} -v ${metric} -f ${logger} -e ${eval} -t ${temp} -a ${approach} -z ${train} --start 0 --stop 25 &> eval.log.${metric}.${M}.${L}.${logger}-${temp}.${eval}.${approach}.${train} &
                    done
                done
            done
        done
	done
done
