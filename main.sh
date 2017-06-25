LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_manifest_speaker.csv --val_manifest data/mit_val_manifest_speaker.csv --cuda
