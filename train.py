########
"""
# insert this to the top of your scripts (usually main.py)
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
"""
########

import argparse
import time
import torch
import torch.nn as nn
import torchnet as tnt
import random

from torch.autograd import Variable
from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from model import DeepSpeech, supported_rnns

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR', help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR', help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample_rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch_size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels_path', default='labels.json', help='Contains all characters for prediction')
parser.add_argument('--window_size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window_stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden_size', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden_layers', default=4, type=int, help='Number of RNN layers')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning_anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint_per_batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--visdom_id', default='Deepspeech training', help='Identifier for visdom graph')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar', help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None, help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0, help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5, help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--no_bucketing', dest='no_bucketing', action='store_true', help='Turn off bucketing and sample from dataset based on sequence length (smallest to largest)')
########
parser.add_argument('--learning_rate_decay_rate', default=0.2, type=float, metavar='LRDR', help='learning rate decay rate')
parser.add_argument('--learning_rate_decay_epochs', default=None, nargs='+', type=int, metavar='LRDE', help='learning rate decay epochs')
parser.add_argument('--loss_type', default='reg', help='Type of the loss. reg|sum|full are supported')
parser.add_argument('--cnn_features', default=400, type=int, help='Hidden size of RNNs')
parser.add_argument('--kernel', default=11, type=int, help='Kernel width')
parser.add_argument('--stride', default=2, type=int, help='Stride in time')
parser.add_argument('--utterance_miliseconds', default=800, type=int, help='Miliseconds size of the utterances')
parser.add_argument('--sample_proportion', default=0.8, type=float, help='Sample proportion to train')
parser.add_argument('--crop_begin', default=40, type=int, help='Miliseconds to crop in the begning before training')
parser.add_argument('--crop_end', default=40, type=int, help='Miliseconds to crop in the end before training')
parser.add_argument('--first_layer_type', default='NONE', help='Type of first layer to be used. none|conv|avgpool are supported')
parser.add_argument('--mfcc', default='false', help='If "true", mfcc will be used')
########
parser.set_defaults(cuda=False, silent=False, checkpoint=False, visdom=False, augment=False, tensorboard=False, log_params=False, no_bucketing=False)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def main():
    args = parser.parse_args()
    torch.set_printoptions(profile="full")
    criterion = nn.CrossEntropyLoss()
    class_accu_reg = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    class_accu_sum = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)

    audio_conf = dict(sample_rate=args.sample_rate, window_size=args.window_size, window_stride=args.window_stride,
                      window=args.window, noise_dir=args.noise_dir, noise_prob=args.noise_prob, noise_levels=(args.noise_min, args.noise_max))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"

    #print("FIRST LAYER TYPE:\t", args.first_layer_type)
    #print("MFCC TRANSFORM:\t\t", args.mfcc)

    model = DeepSpeech(rnn_hidden_size=args.hidden_size, nb_layers=args.hidden_layers, rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf, bidirectional=True, cnn_features=args.cnn_features, kernel=args.kernel,
                       first_layer_type=args.first_layer_type, stride=args.stride, mfcc=args.mfcc)

    ########
    #print(list(model.rnns.modules()))
    #for rnn in model.rnns.modules():
    #    print(rnn)#.flatten_parameters()
    #def flat_model(model):
    #    for m in model.modules():
    #        if isinstance(m, nn.LSTM):
    #            m.flatten_parameters()
    ########

    parameters = model.parameters()
    optimizer = torch.optim.SGD(parameters, lr=args.lr,
                                momentum=args.momentum, nesterov=True)

    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    avg_loss = 0
    start_epoch = 0
    start_iter = 0
    best_train_accu_reg = 0
    best_train_accu_sum = 0
    best_test_accu_reg = 0
    best_test_accu_sum = 0
    best_avg_loss = float("inf") # sys.float_info.max # 1000000
    epoch_70 = None
    epoch_90 = None
    epoch_95 = None
    epoch_99 = None

    utterance_sequence_length = int(args.utterance_miliseconds / 10)

    loss_begin = round(args.crop_begin / (10 * args.stride))
    loss_end = -round(args.crop_end / (10 * args.stride)) or None
    #print("LOSS BEGIN:", loss_begin)
    #print("LOSS END:", loss_end)

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("Number of parameters: ", DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    #losses = AverageMeter()

    print(args, "\n")

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        scheduler.step()
        optim_state_now = optimizer.state_dict()
        print('\nLEARNING RATE: {lr:.6f}'.format(lr=optim_state_now['param_groups'][0]['lr']))
        class_accu_reg.reset()
        class_accu_sum.reset()
        model.train()
        end = time.time()

        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_loader):
                break

            inputs, input_percentages, speaker_labels, mfccs = data

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)

            ########
            mfccs = Variable(mfccs, requires_grad=False)
            if args.mfcc == "true" : inputs = mfccs # <<-- This line makes us to use mfccs...
            #print("INPUTS SIZE:", inputs.size())
            #print("MFCCS SIZE:", mfccs.size())
            ########

            speaker_labels = Variable(speaker_labels, requires_grad=False)
            speaker_labels = speaker_labels.cuda(async=True).long()

            if args.cuda:
                inputs = inputs.cuda()

            ########
            ########
            sizes = inputs.size()
            inputs = inputs.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # Collapse feature dimension
            #print("INPUTS SIZE: ====>>>>>\t", inputs.size())
            #start = 0
            #duration = 100
            start = random.randint(0, int((inputs.size(2)-1)*(1-args.sample_proportion)))
            duration = int((inputs.size(2))*(args.sample_proportion))
            #start = random.randint(0, (inputs.size(3)-1)-utterance_sequence_length)
            #duration = utterance_sequence_length
            utterances = inputs[...,start:start+duration] # <<<<<<====== THIS IS THE MOST IMPORTANT CODE OF THE PROJECT
            #print("UTTERS SIZE: ====>>>>>\t", utterances.size(), start, start+duration)
            out = model(utterances)
            #print("OUTPUT SIZE: ====>>>>>\t", out.size())
            out = out.transpose(0, 1)  # TxNxH
            ########
            ########

            # Prints the output of the model in a sequence of probabilities of char for each audio...
            #torch.set_printoptions(profile="full")
            ####print("OUT: " + str(out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()), "INPUT PERCENTAGES MEAN: " + str(input_percentages.mean()))
            #print(out[:,:,0])
            #print("SPEAKER LABELS: " + str(speaker_labels))
            #print(out[0][0])
            #softmax_output = F.softmax(out).data # This DOES NOT what I want...
            #softmax_output_alt = flex_softmax(out, axis=2).data # This is FINE!!! <<<===
            #print(softmax_output[0][0])
            #print(softmax_output_alt[0][0])
            ####new_out = torch.sum(out, 0)
            ####new_out = torch.sum(out[20:], 0)
            #print(out.size())
            #print(new_out.size())
            #print(out[-1].size())

            class_accu_reg.add(out[round(out.size(0)/2)].data, speaker_labels.data)
            class_accu_sum.add(torch.sum(out, 0).data, speaker_labels.data)
            #class_accu_reg.add(processed_out.data, processed_speaker_labels.data)

            if args.loss_type == "reg":
                processed_out = out[round(out.size(0)/2)]; processed_speaker_labels = speaker_labels
            elif args.loss_type == "sum":
                #processed_out = torch.sum(out[loss_begin:loss_end], 0); processed_speaker_labels = speaker_labels
                processed_out = torch.sum(out, 0); processed_speaker_labels = speaker_labels
            elif args.loss_type == "full":
                #processed_out = out.contiguous()[loss_begin:loss_end].view(-1,48); processed_speaker_labels = speaker_labels.repeat(out.size(0),1)[loss_begin:loss_end].view(-1) #speaker_labels = speaker_labels.expand(20, out.size(0))
                processed_out = out.contiguous().view(-1, 48); processed_speaker_labels = speaker_labels.repeat(out.size(0),1).view(-1)  # speaker_labels = speaker_labels.expand(20, out.size(0))
            #print("OUT: " + str(out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()))
            #print("PROC OUTPUT: ====>>>>>\t" + str(processed_out.size()))
            #print("PROC LABELS: ====>>>>>\t" + str(processed_speaker_labels.size()))

            loss = criterion(processed_out, processed_speaker_labels)
            loss = loss / inputs.size(0)  # average the loss by minibatch
            loss_sum = loss.data.sum()
            inf = float("inf")
            if loss_sum == inf or loss_sum == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0
            else:
                loss_value = loss.data[0]
            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            #accu_out3 = torch.sum(flex_softmax(out[20:], axis=2), 0)
            #print(classaccu.value()[0], classaccu.value()[1])
            # Cross Entropy Loss for a Sequence (Time Series) of Output?
            #output = output.view(-1,29)
            #target = target.view(-1)
            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(output,target)

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            #torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)

            # SGD step
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if not args.silent:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.8f} ({loss.avg:.8f})\t'
                      'CARR {carr:.2f}\t'
                      'CARS {cars:.2f}\t'
                      .format((epoch + 1), (i + 1), len(train_loader), batch_time=batch_time, data_time=data_time,
                              loss=losses, carr=class_accu_reg.value()[0], cars=class_accu_sum.value()[0]))

            if args.cuda:
                torch.cuda.synchronize()

            del loss
            del out
            del processed_out
            del speaker_labels
            del processed_speaker_labels

        avg_loss /= len(train_loader)

        if (best_avg_loss > avg_loss): best_avg_loss = avg_loss

        print("\nCURRENT EPOCH AVERAGE LOSS:\t", avg_loss)
        print("\nCURRENT EPOCH TRAINING RESULTS:\t", class_accu_reg.value()[0], "\t", class_accu_sum.value()[0], "\n")

        if (best_train_accu_reg < class_accu_reg.value()[0]): best_train_accu_reg = class_accu_reg.value()[0]
        if (best_train_accu_sum < class_accu_sum.value()[0]): best_train_accu_sum = class_accu_sum.value()[0]

        get_70 = (class_accu_reg.value()[0] > 70)
        if ((epoch_70 is None) and (get_70 == True)): epoch_70 = epoch + 1
        get_90 = (class_accu_reg.value()[0] > 90)
        if ((epoch_90 is None) and (get_90 == True)): epoch_90 = epoch + 1
        get_95 = (class_accu_reg.value()[0] > 95)
        if ((epoch_95 is None) and (get_95 == True)): epoch_95 = epoch + 1
        get_99 = (class_accu_reg.value()[0] > 99)
        if ((epoch_99 is None) and (get_99 == True)): epoch_99 = epoch + 1

        start_iter = 0  # Reset start iteration for next epoch
        model.eval()

        class_accu_reg.reset()
        class_accu_sum.reset()

        for i, (data) in enumerate(test_loader):  # test

            inputs, input_percentages, speaker_labels, mfccs = data

            inputs = Variable(inputs, volatile=True)

            ########
            mfccs = Variable(mfccs, requires_grad=False)
            if args.mfcc == "true" : inputs = mfccs # <<-- This line makes us to use mfccs...
            #print("INPUTS SIZE:", inputs.size())
            #print("MFCCS SIZE:", mfccs.size())
            ########

            speaker_labels = Variable(speaker_labels, requires_grad=False)
            speaker_labels = speaker_labels.cuda(async=True).long()

            if args.cuda:
                inputs = inputs.cuda()

            ########
            ########
            sizes = inputs.size()
            inputs = inputs.view(sizes[0], sizes[1]*sizes[2], sizes[3])  # Collapse feature dimension
            #print("INPUTS SIZE: ====>>>>>\t", inputs.size())
            start = 0
            duration = 100
            #start = random.randint(0, int((inputs.size(3)-1)*(1-args.sample_proportion)))
            #duration = int((inputs.size(3))*(args.sample_proportion))
            #start = random.randint(0, (inputs.size(3)-1)-utterance_sequence_length)
            #duration = utterance_sequence_length
            utterances = inputs#[...,start:start+duration] # <<<<<<====== THIS IS THE MOST IMPORTANT CODE OF THE PROJECT
            #print("UTTERS SIZE: ====>>>>>\t", utterances.size(), start, start+duration)
            out = model(utterances)
            #print("OUTPUT SIZE: ====>>>>>\t", out.size())
            out = out.transpose(0, 1)  # TxNxH
            ########
            ########

            # Prints the output of the model in a sequence of probabilities of char for each audio...
            #torch.set_printoptions(profile="full")
            ########print("OUT: " + str(out.size()), "NEW OUT:" + str(new_out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()), "INPUT PERCENTAGES MEAN: " + str(input_percentages.mean()))
            #print(out[:,:,0])
            #print("SPEAKER LABELS: " + str(speaker_labels))
            #print(out[0][0])
            #softmax_output = F.softmax(out).data # This DOES NOT what I want...
            #softmax_output_alt = flex_softmax(out, axis=2).data # This is FINE!!! <<<===
            #print(softmax_output[0][0])
            #print(softmax_output_alt[0][0])
            ########

            #if args.loss_type == "reg":
            #    processed_out = out[round(out.size(0)/2)]; processed_speaker_labels = speaker_labels
            #elif args.loss_type == "sum" or "full":
            #    #processed_out = torch.sum(out[loss_begin:loss_end], 0); processed_speaker_labels = speaker_labels
            #    processed_out = torch.sum(out, 0); processed_speaker_labels = speaker_labels
            #elif args.loss_type == "full":
            #    #processed_out = out.contiguous()[loss_begin:loss_end].view(-1,48); processed_speaker_labels = speaker_labels.repeat(out.size(0),1)[loss_begin:loss_end].view(-1) #speaker_labels = speaker_labels.expand(20, out.size(0))
            #    processed_out = out.contiguous().view(-1, 48); processed_speaker_labels = speaker_labels.repeat(out.size(0),1).view(-1)  # speaker_labels = speaker_labels.expand(20, out.size(0))
            #print("OUT: " + str(out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()))
            #print("PROC OUTPUT: ====>>>>>\t" + str(processed_out.size()))
            #print("PROC LABELS: ====>>>>>\t" + str(processed_speaker_labels.size()))

            class_accu_reg.add(out[round(out.size(0)/2)].data, speaker_labels.data)
            class_accu_sum.add(torch.sum(out, 0).data, speaker_labels.data)
            #class_accu_reg.add(processed_out.data, processed_speaker_labels.data)

            print('Validation Summary Epoch: [{0}]\t'
                  'CARR {carr:.2f}\t'
                  'CARS {cars:.2f}\t'
                  .format(epoch + 1, carr=class_accu_reg.value()[0], cars=class_accu_sum.value()[0]))

            if args.cuda:
                torch.cuda.synchronize()

            del out

        print("\nCURRENT EPOCH TEST RESULTS:\t", class_accu_reg.value()[0], "\t", class_accu_sum.value()[0], "\n")

        if (best_test_accu_reg < class_accu_reg.value()[0]): best_test_accu_reg = class_accu_reg.value()[0]
        if (best_test_accu_sum < class_accu_sum.value()[0]): best_test_accu_sum = class_accu_sum.value()[0]

        print("\nBEST AVERAGE LOSS:\t\t", best_avg_loss)
        print("\nBEST EPOCH TRAINING RESULTS:\t", best_train_accu_reg, "\t", best_train_accu_sum)
        print("\nBEST EPOCH TEST RESULTS:\t", best_test_accu_reg, "\t", best_test_accu_sum)
        print("\nEPOCHS 70%, 90%, 95%, 99%:\t", epoch_70, "\t", epoch_90, "\t", epoch_95, "\t", epoch_99, "\n")

        torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch), args.model_path)

        avg_loss = 0

        if not args.no_bucketing and epoch == 0:
            print("Switching to bucketing sampler for following epochs")
            train_dataset = SpectrogramDatasetWithLength(audio_conf=audio_conf, manifest_filepath=args.train_manifest, normalize=True, augment=args.augment)
            sampler = BucketingSampler(train_dataset)
            train_loader.sampler = sampler

if __name__ == '__main__':
    main()
