LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python rnn_train_0_0.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda \
--batch_size 20 --learning_anneal 1 --epochs 300
