LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--learning-rate 0.01 --learning-rate-decay-rate 0.2 --learning-rate-decay-epochs 20 25 --epochs 30
