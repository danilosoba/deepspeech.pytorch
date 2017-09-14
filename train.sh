# lr was 0.01 for reg and sum to fast 100% training accuracy...
# lr was 0.1 to full to fast more than 90% training accuracy...



LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 300 --hidden_layers 4 \
--kernel 11 --stride 2 --crop_begin 200 --crop_end 200 | tee logs/full_crop200x200_conv11x2_arch400x4_lr0.05.txt



LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 300 --hidden_layers 4 \
--kernel 11 --stride 2 --crop_begin 200 --crop_end 200 | tee logs/sum_crop200x200_conv11x2_arch400x4_lr0.005.txt



LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 300 --hidden_layers 4 \
--kernel 11 --stride 2 --crop_begin 200 --crop_end 200 | tee logs/reg_crop200x200_conv11x2_arch400x4_lr0.05.txt



LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg2 --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 300 --hidden_layers 4 \
--kernel 11 --stride 2 --crop_begin 200 --crop_end 200 | tee logs/reg2_crop200x200_conv11x2_arch400x4_lr0.05.txt
