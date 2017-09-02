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
import errno
import json
import os
import time

import torch

########
import torch.nn as nn
import torchnet as tnt
########

from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from data.bucketing_sampler import BucketingSampler, SpectrogramDatasetWithLength
from data.data_loader import AudioDataLoader, SpectrogramDataset
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

########
from libs.utils import flex_softmax
########

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train_manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val_manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
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
parser.add_argument('--model_path', default='models/deepspeech_final.pth.tar',
                    help='Location to save best validation model')
parser.add_argument('--continue_from', default='', help='Continue from checkpoint model')
parser.add_argument('--rnn_type', default='lstm', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise_dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise_prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise_min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise_max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log_dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log_params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--no_bucketing', dest='no_bucketing', action='store_true',
                    help='Turn off bucketing and sample from dataset based on sequence length (smallest to largest)')
########
parser.add_argument('--learning-rate-decay-rate', default=0.2, type=float,
                    metavar='LRDR', help='learning rate decay rate')
parser.add_argument('--learning-rate-decay-epochs', default=None, nargs='+', type=int,
                    metavar='LRDE', help='learning rate decay epochs')
########
parser.set_defaults(cuda=False, silent=False, checkpoint=False, visdom=False, augment=False, tensorboard=False,
                    log_params=False, no_bucketing=False)


def to_np(x):
    return x.data.cpu().numpy()


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
    save_folder = args.save_folder

    ########
    """
    loss_results, cer_results, wer_results = torch.Tensor(args.epochs), torch.Tensor(args.epochs), torch.Tensor(
        args.epochs)
    best_wer = None
    if args.visdom:
        from visdom import Visdom
        viz = Visdom()

        opts = [dict(title=args.visdom_id + ' Loss', ylabel='Loss', xlabel='Epoch'),
                dict(title=args.visdom_id + ' WER', ylabel='WER', xlabel='Epoch'),
                dict(title=args.visdom_id + ' CER', ylabel='CER', xlabel='Epoch')]

        viz_windows = [None, None, None]
        epochs = torch.arange(1, args.epochs + 1)
    if args.tensorboard:
        from logger import TensorBoardLogger
        try:
            os.makedirs(args.log_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                print('Directory already exists.')
                for file in os.listdir(args.log_dir):
                    file_path = os.path.join(args.log_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        raise
            else:
                raise
        logger = TensorBoardLogger(args.log_dir)

    try:
        os.makedirs(save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise
    """
    ########

    ########
    """
    criterion = CTCLoss()
    """
    criterion = nn.CrossEntropyLoss()
    class_accu_train = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    class_accu_val = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    class_accu_val_sum = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    class_accu_val_sum_20 = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    class_accu_val_sum_20_softmax = tnt.meter.ClassErrorMeter(topk=[1], accuracy=True)
    ########

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))
    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, augment=False)
    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size,
                                   num_workers=args.num_workers)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=True)

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

    ########
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.learning_rate_decay_epochs, gamma=args.learning_rate_decay_rate)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    ########

    ########
    """
    decoder = GreedyDecoder(labels)
    """
    ########

    ########
    """
    if args.continue_from:
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from)
        model.load_state_dict(package['state_dict'])
        optimizer.load_state_dict(package['optim_dict'])
        start_epoch = int(package.get('epoch', 1)) - 1  # Python index start at 0 for training
        start_iter = package.get('iteration', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 0
        else:
            start_iter += 1
        avg_loss = int(package.get('avg_loss', 0))
        loss_results, cer_results, wer_results = package['loss_results'], package[
            'cer_results'], package['wer_results']
        if args.visdom and \
                        package['loss_results'] is not None and start_epoch > 0:  # Add previous scores to visdom graph
            x_axis = epochs[0:start_epoch]
            y_axis = [loss_results[0:start_epoch], wer_results[0:start_epoch], cer_results[0:start_epoch]]
            for x in range(len(viz_windows)):
                viz_windows[x] = viz.line(
                    X=x_axis,
                    Y=y_axis[x],
                    opts=opts[x],
                )
        if args.tensorboard and \
                        package['loss_results'] is not None and start_epoch > 0:  # Previous scores to tensorboard logs
            for i in range(start_epoch):
                info = {
                    'Avg Train Loss': loss_results[i],
                    'Avg WER': wer_results[i],
                    'Avg CER': cer_results[i]
                }
                for tag, val in info.items():
                    logger.scalar_summary(tag, val, i + 1)
        if not args.no_bucketing and epoch != 0:
            print("Using bucketing sampler for the following epochs")
            train_dataset = SpectrogramDatasetWithLength(audio_conf=audio_conf, manifest_filepath=args.train_manifest,
                                                         labels=labels,
                                                         normalize=True, augment=args.augment)
            sampler = BucketingSampler(train_dataset)
            train_loader.sampler = sampler
    else:
        avg_loss = 0
        start_epoch = 0
        start_iter = 0
    """
    avg_loss = 0
    start_epoch = 0
    start_iter = 0
    ########

    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for epoch in range(start_epoch, args.epochs):
        ########
        scheduler.step()
        optim_state_now = optimizer.state_dict()
        print('\nLEARNING RATE: {lr:.6f}'.format(lr=optim_state_now['param_groups'][0]['lr']))
        class_accu_train.reset()
        ########
        model.train()
        end = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_loader):
                break

            ########
            """
            inputs, targets, input_percentages, target_sizes = data
            """
            inputs, targets, input_percentages, target_sizes, speaker_labels = data
            ########

            # measure data loading time
            data_time.update(time.time() - end)
            inputs = Variable(inputs, requires_grad=False)

            ########
            """
            target_sizes = Variable(target_sizes, requires_grad=False)
            targets = Variable(targets, requires_grad=False)
            """
            speaker_labels = Variable(speaker_labels, requires_grad=False)
            ########

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            ########
            speaker_labels = speaker_labels.cuda(async=True).long()
            # Prints the output of the model in a sequence of probabilities of char for each audio...
            torch.set_printoptions(profile="full")
            ########print("OUT: " + str(out.size()), "NEW OUT:" + str(new_out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()), "INPUT PERCENTAGES MEAN: " + str(input_percentages.mean()))
            """
            seq_length = out.size(0)
            sizes = Variable(input_percentages.mul_(int(seq_length)).int(), requires_grad=False)

            loss = criterion(out, targets, sizes, target_sizes)
            """
            #print(out[:,:,0])
            #print("SPEAKER LABELS: " + str(speaker_labels))
            #print(out[0][0])
            #softmax_output = F.softmax(out).data # This DOES NOT what I want...
            #softmax_output_alt = flex_softmax(out, axis=2).data # This is FINE!!! <<<===
            #print(softmax_output[0][0])
            #print(softmax_output_alt[0][0])
            ########new_out = torch.sum(out, 0)
            ########new_out = torch.sum(out[20:], 0)
            #print(out.size())
            #print(new_out.size())
            #print(out[-1].size())
            loss_out = out[-1]
            loss = criterion(loss_out, speaker_labels)
            ########

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

            ########
            ########accu_out = torch.sum(out, 0)
            ########accu_out = torch.sum(out[20:], 0)
            accu_out = out[-1]
            class_accu_train.add(accu_out.data, speaker_labels.data)
            #print(classaccu.value()[0], classaccu.value()[1])
            # Cross Entropy Loss for a Sequence (Time Series) of Output?
            #output = output.view(-1,29)
            #target = target.view(-1)
            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(output,target)
            ########

            # compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            # SGD step
            optimizer.step()

            if args.cuda:
                torch.cuda.synchronize()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.silent:

                ########
                """
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                    (epoch + 1), (i + 1), len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
                """
                print('Epoch: [{0}][{1}/{2}]\t'
                      #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Average CAR {car:.3f}\t'
                      .format((epoch + 1), (i + 1), len(train_loader), batch_time=batch_time, data_time=data_time,
                              loss=losses, car=class_accu_train.value()[0]))
                ########

            ########
            """
            if args.checkpoint_per_batch > 0 and i > 0 and (i + 1) % args.checkpoint_per_batch == 0:
                file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth.tar' % (save_folder, epoch + 1, i + 1)
                print("Saving checkpoint model to %s" % file_path)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
                           file_path)
            """
            ########

            del loss
            del out

            ########
            del loss_out
            del accu_out
            ########

        avg_loss /= len(train_loader)

        ########
        """
        print('Training Summary Epoch: [{0}]\t'
              'Average Loss {loss:.3f}\t'.format(
            epoch + 1, loss=avg_loss))
        """
        print("\nFINAL EPOCH TRAINING RESULTS:", class_accu_train.value()[0], "\n")
        ########

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()

        ########
        class_accu_val.reset()
        class_accu_val_sum.reset()
        class_accu_val_sum_20.reset()
        class_accu_val_sum_20_softmax.reset()
        ########

        for i, (data) in enumerate(test_loader):  # test

            ########
            """
            inputs, targets, input_percentages, target_sizes = data
            """
            inputs, targets, input_percentages, target_sizes, speaker_labels = data
            ########

            inputs = Variable(inputs, volatile=True)

            ########
            speaker_labels = Variable(speaker_labels, requires_grad=False)
            speaker_labels = speaker_labels.cuda(async=True).long()
            """
            # unflatten targets
            split_targets = []
            offset = 0
            for size in target_sizes:
                split_targets.append(targets[offset:offset + size])
                offset += size
            """
            ########

            if args.cuda:
                inputs = inputs.cuda()

            out = model(inputs)
            out = out.transpose(0, 1)  # TxNxH

            ########
            speaker_labels = speaker_labels.cuda(async=True).long()
            # Prints the output of the model in a sequence of probabilities of char for each audio...
            torch.set_printoptions(profile="full")
            ########print("OUT: " + str(out.size()), "NEW OUT:" + str(new_out.size()), "SPEAKER LABELS:" + str(speaker_labels.size()), "INPUT PERCENTAGES MEAN: " + str(input_percentages.mean()))
            #print(out[:,:,0])
            #print("SPEAKER LABELS: " + str(speaker_labels))
            #print(out[0][0])
            #softmax_output = F.softmax(out).data # This DOES NOT what I want...
            #softmax_output_alt = flex_softmax(out, axis=2).data # This is FINE!!! <<<===
            #print(softmax_output[0][0])
            #print(softmax_output_alt[0][0])
            ########

            ########
            accu_out = out[-1]
            class_accu_val.add(accu_out.data, speaker_labels.data)
            accu_out1 = torch.sum(out, 0)
            class_accu_val_sum.add(accu_out1.data, speaker_labels.data)
            accu_out2 = torch.sum(out[20:], 0)
            class_accu_val_sum_20.add(accu_out2.data, speaker_labels.data)
            accu_out3 = torch.sum(flex_softmax(out[20:], axis=2), 0)
            class_accu_val_sum_20_softmax.add(accu_out3.data, speaker_labels.data)

            #print(classaccu.value()[0], classaccu.value()[1])
            # Cross Entropy Loss for a Sequence (Time Series) of Output?
            #output = output.view(-1,29)
            #target = target.view(-1)
            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(output,target)
            print('Validation Summary Epoch: [{0}]\t'
                  'Average CAR {car:.3f}\t'
                  'Average CAR_SUM {car_sum:.3f}\t'
                  'Average CAR_SUM_20 {car_sum_20:.3f}\t'
                  'Average CAR_SUM_20_SOFTMAX {car_sum_20_softmax:.3f}\t'
                  .format(epoch + 1, car=class_accu_val.value()[0], car_sum=class_accu_val_sum.value()[0],
                          car_sum_20=class_accu_val_sum_20.value()[0], car_sum_20_softmax=class_accu_val_sum_20_softmax.value()[0]))
            """
            seq_length = out.size(0)
            sizes = input_percentages.mul_(int(seq_length)).int()            
            decoded_output = decoder.decode(out.data, sizes)
            target_strings = decoder.process_strings(decoder.convert_to_strings(split_targets))
            wer, cer = 0, 0
            for x in range(len(target_strings)):
                wer += decoder.wer(decoded_output[x], target_strings[x]) / float(len(target_strings[x].split()))
                cer += decoder.cer(decoded_output[x], target_strings[x]) / float(len(target_strings[x]))
            total_cer += cer
            total_wer += wer
            """
            ########

            if args.cuda:
                torch.cuda.synchronize()
            del out

            ########
            del accu_out
            ########

        ########
        """
        wer = total_wer / len(test_loader.dataset)
        cer = total_cer / len(test_loader.dataset)
        wer *= 100
        cer *= 100
        loss_results[epoch] = avg_loss
        wer_results[epoch] = wer
        cer_results[epoch] = cer
        print('Validation Summary Epoch: [{0}]\t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(
            epoch + 1, wer=wer, cer=cer))
        """
        ########

        ########
        print("\nFINAL EPOCH TEST RESULTS:", class_accu_val.value()[0], class_accu_val_sum.value()[0],
              class_accu_val_sum_20.value()[0], class_accu_val_sum_20_softmax.value()[0], "\n")
        ########

        ########
        """
        if args.visdom:
            # epoch += 1
            x_axis = epochs[0:epoch + 1]
            y_axis = [loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]]
            for x in range(len(viz_windows)):
                if viz_windows[x] is None:
                    viz_windows[x] = viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        opts=opts[x],
                    )
                else:
                    viz.line(
                        X=x_axis,
                        Y=y_axis[x],
                        win=viz_windows[x],
                        update='replace',
                    )
        if args.tensorboard:
            info = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer,
                'Avg CER': cer
            }
            for tag, val in info.items():
                logger.scalar_summary(tag, val, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), epoch + 1)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), epoch + 1)
        if args.checkpoint:
            file_path = '%s/deepspeech_%d.pth.tar' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results),
                       file_path)

        # anneal lr
        optim_state = optimizer.state_dict()
        optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / args.learning_anneal
        optimizer.load_state_dict(optim_state)
        print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))

        if best_wer is None or best_wer > wer:
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results)
                       , args.model_path)
            best_wer = wer
        """
        ########

        avg_loss = 0
        if not args.no_bucketing and epoch == 0:
            print("Switching to bucketing sampler for following epochs")
            train_dataset = SpectrogramDatasetWithLength(audio_conf=audio_conf, manifest_filepath=args.train_manifest,
                                                         labels=labels,
                                                         normalize=True, augment=args.augment)
            sampler = BucketingSampler(train_dataset)
            train_loader.sampler = sampler


if __name__ == '__main__':
    main()
