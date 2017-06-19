LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/an4_train_manifest.csv --val_manifest data/an4_val_manifest.csv --cuda
