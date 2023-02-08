#!/bin/sh

for x1 in max avg; do
    for x2 in xgboost randomforest; do
        for x3 in subject; do
            for x4 in raw; do
            	for x5 in 7 14 21; do

				    python src/los_pred_xgboost_randomforest.py \
				        --num_workers 8 \
				        --savedir models/XGBoost/LOS/${x4}/${x3}/${x2}/${x1} \
				        --datatype ${x4} \
				        --timeseries_mode ${x1} \
				        --datasplit ${x3} \
				        --model ${x2} \
				        --estimator 5000 \
				        --gpu 1 \
				        --los ${x5} 
				        
				done
            done
        done
    done
done


for x1 in max avg; do
    for x2 in xgboost randomforest; do
        for x3 in subject; do
            for x4 in raw; do

                python src/los_pred_xgboost_randomforest.py \
                    --num_workers 8 \
                    --savedir models/XGBoost/mortality/${x4}/${x3}/${x2}/${x1} \
                    --datatype ${x4} \
                    --timeseries_mode ${x1} \
                    --datasplit ${x3} \
                    --model ${x2} \
                    --estimators 5000 \
                    --gpu 1 \
                    --los 0 \
                    --mortality True

            done
        done
    done
done
