LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python test.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda
