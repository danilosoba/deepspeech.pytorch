# lr was 0.01 for reg and sum to fast 100% training accuracy...
# lr was 0.1 to full to fast more than 90% training accuracy...



# 300 appears to be ok since was better than 200 and 400...
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 2 --hidden_size 300 --cnn_features 300 \
--kernel 6 --stride 3 --crop_begin 100 --crop_end 100 --sample_proportion 0.8 | tee logs/reg_sample0.8_crop100x100_conv6x3_arch300x300x2_lr0.05.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 1 --hidden_size 300 --cnn_features 300 \
--kernel 6 --stride 3 --crop_begin 100 --crop_end 100 --sample_proportion 0.8 | tee logs/reg_sample0.8_crop100x100_conv6x3_arch300x300x1_lr0.05.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 3 --hidden_size 300 --cnn_features 300 \
--kernel 6 --stride 3 --crop_begin 100 --crop_end 100 --sample_proportion 0.8 | tee logs/reg_sample0.8_crop100x100_conv6x3_arch300x300x3_lr0.05.txt



# We can test above 400...
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 4 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv6x3_arch400x400x4_lr0.05.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 5 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv6x3_arch400x400x5_lr0.05.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv6x3_arch400x400x6_lr0.05.txt



# We can test above 400...
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 4 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv6x3_arch400x400x4_lr0.005.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 5 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv6x3_arch400x400x5_lr0.005.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 400 --cnn_features 400 \
--kernel 6 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv6x3_arch400x400x6_lr0.005.txt
