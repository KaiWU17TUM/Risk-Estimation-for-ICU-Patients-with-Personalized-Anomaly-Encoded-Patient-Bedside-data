#!/bin/sh

for x1 in raw encoded; do
	for x2 in 7 14 21; do

		python src/los_pred_cnn.py \
			--num_workers 16 \
			--savedir models/CNN_LOS_PRED/los${x2} \
			--datatype ${x1} \
			--gpu 5 \
			--los ${x2}

	done
done


for x1 in raw encoded; do

		python src/mortality_pred_cnn.py \
			--num_workers 16 \
			--savedir models/CNN_LOS_PRED/mortality \
			--datatype ${x1} \
			--gpu 5 \

done
