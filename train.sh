#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
#--sample_proportion 0.8 --first_layer_type CONV --mfcc false #| tee spect_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-conv600x11x3-rnn2x600.log

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
#--sample_proportion 0.8 --first_layer_type AVGPOOL --mfcc false #| tee spect_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-avgpool11x3-rnn2x600.log

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
#--sample_proportion 0.8 --first_layer_type NONE --mfcc false #| tee spect_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-none-rnn2x600.log

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
--sample_proportion 0.8 --first_layer_type CONV --mfcc true #| tee mfcc_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-conv600x11x3-rnn2x600.log

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
--sample_proportion 0.8 --first_layer_type AVGPOOL --mfcc true #| tee mfcc_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-avgpool11x3-rnn2x600.log

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.05 --epochs 300 --hidden_layers 2 --hidden_size 600 --cnn_features 600 --kernel 11 --stride 3 \
--sample_proportion 0.8 --first_layer_type NONE --mfcc true #| tee mfcc_reg_lr0.05-0.99_train-prop0.8_test-prop0.8_arch-none-rnn2x600.log

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
