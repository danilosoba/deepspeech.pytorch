#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.10 --epochs 300 --cnn_features 32 --kernel 3 --stride 1 \
#--sample_proportion 0.8 --first_layer_type CONV --mfcc false | tee spect_full_lr0.10-0.99_arch-conv32-64-128.log

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --epochs 100 --cnn_features 600 --kernel 11 --stride 3 --hidden_layers 1 --hidden_size 600 \
--sample_proportion 0.8 --crop_begin 300 --crop_end 300 --first_layer_type CONV --mfcc false #| tee spect_full_lr0.05-0.99_arch-conv32-64-128.log

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.01 --epochs 300 --cnn_features 32 --kernel 3 --stride 1 \
#--sample_proportion 0.8 --first_layer_type CONV --mfcc false | tee spect_full_lr0.01-0.99_arch-conv32-64-128.log

# --hidden_layers 2 --hidden_size 600

#############################################################################

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 800 --cnn_features 800 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv11x3_arch800x800x6_lr0.05.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 1000 --cnn_features 1000 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv11x3_arch1000x1000x6_lr0.05.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.05 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 1200 --cnn_features 1200 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/full_sample0.8_crop200x200_conv11x3_arch1200x1200x6_lr0.05.txt

##############################################################################

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 800 --cnn_features 800 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv11x3_arch800x800x6_lr0.005.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 1000 --cnn_features 1000 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv11x3_arch1000x1000x6_lr0.005.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type sum --learning-rate 0.005 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 100 --hidden_layers 6 --hidden_size 1200 --cnn_features 1200 \
#--kernel 11 --stride 3 --crop_begin 200 --crop_end 200 --sample_proportion 0.8 | tee logs/sum_sample0.8_crop200x200_conv11x3_arch1200x1200x6_lr0.005.txt
