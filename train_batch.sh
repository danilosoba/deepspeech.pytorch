# lr was 0.01 for reg and sum to fast 100% training accuracy...
# lr was 0.1 to full to fast more than 90% training accuracy...



#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type reg --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 1 | tee logs/reg_11x2_400_1_lr0.0003.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type reg --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 2 | tee logs/reg_11x2_400_2_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 3 | tee logs/reg_11x2_400_3_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 4 | tee logs/reg_11x2_400_4_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type reg --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 5 | tee logs/reg_11x2_400_5_lr0.0003.txt



#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type sum --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 1 | tee logs/sum_11x2_400_1_lr0.0003.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type sum --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 2 | tee logs/sum_11x2_400_2_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 3 | tee logs/sum_11x2_400_3_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 4 | tee logs/sum_11x2_400_4_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type sum --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 5 | tee logs/sum_11x2_400_5_lr0.0003.txt



#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 1 | tee logs/full_11x2_400_1_lr0.0003.txt

#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
#--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
#--loss_type full --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 2 | tee logs/full_11x2_400_2_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 3 | tee logs/full_11x2_400_3_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 4 | tee logs/full_11x2_400_4_lr0.0003.txt

LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1 python train.py \
--train_manifest data/mit_train_speaker_identification.csv --val_manifest data/mit_val_speaker_identification.csv --cuda --batch_size 20 \
--loss_type full --learning-rate 0.0003 --learning_rate_decay_rate 0.2 --learning_rate_decay_epochs 500 500 --epochs 500 --hidden_layers 5 | tee logs/full_11x2_400_5_lr0.0003.txt
