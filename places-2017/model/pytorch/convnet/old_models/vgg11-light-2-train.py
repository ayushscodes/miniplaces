# Python builtin.
import time
import shutil
import datetime
import sys, os

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Changes
# Using Adam optimizer
# Store top 5 with checkpoints.
# Switched back to VGG11 feature detection.

# Files in this directory.
sys.path.append('../')
import vgg_pytorch as VGG
from miniplaces_dataset import *
from utils import accuracy, AverageMeter, save_checkpoint, log

def main():
    # Apply a series of transformations to the input data.
    DATA_MEAN = (0.45834960097, 0.44674252445, 0.41352266842)
    DATA_STD = (0.229, 0.224, 0.225)
    CROP_SIZE = 120
    batch_size = 200

    print('Batch size:', batch_size)
    print('Crop size:', CROP_SIZE)

    transform = transforms.Compose(
        [transforms.RandomSizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD)]
    )

    # Load in the training set.
    training_set = MiniPlacesDataset(os.path.abspath('./../../../data/train.txt'),
                                     os.path.abspath('./../../../data/images/'),
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load in the validation set.
    val_set = MiniPlacesDataset(os.path.abspath('./../../../data/val.txt'),
                                os.path.abspath('./../../../data/images/'),
                                transform=transforms.Compose([
                                transforms.CenterCrop(CROP_SIZE),
                                transforms.ToTensor(),
                                transforms.Normalize(DATA_MEAN, DATA_STD)]))

    val_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4)

    # Define the model, loss, and optimizer.
    model = VGG.vgg11(num_classes=100, dropout=0.5, light=True)
    model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    
    print("Model Parameters:", sum(param.numel() for param in model.parameters()))

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters())

    # Parameters
    start_epoch = 0
    epochs = 30
    print_freq = 10
    is_best = True
    best_prec1 = 0
    checkpoint_file = './model_best.pth.tar'

    # If checkpoint file is given, resume from there.
    if checkpoint_file != None:
        if os.path.isfile(checkpoint_file):
            print("Loading checkpoint '{}'".format(checkpoint_file))
            checkpoint = torch.load(checkpoint_file)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict']) # Get frozen weights.
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))
            log("Loaded checkpoint '{}' (epoch {})".format(checkpoint_file, checkpoint['epoch']))

        else:
            log("No checkpoint found at {}. Initializing from scratch.".format(checkpoint_file))
            print("No checkpoint found at {}. Initializing from scratch.".format(checkpoint_file))

    # Training + Validation + Saving loop.
    for epoch in range(start_epoch, epochs):

        # Train for one epoch.
        train(train_loader, model, criterion, optimizer, epoch, print_freq=print_freq)

        # Every epoch, test on the validation data.
        prec1, prec5 = validate(val_loader, model, criterion, print_freq=print_freq)
        is_best = prec1 > best_prec1 # Check if top1 precision has improved.
        best_prec1 = max(prec1, best_prec1)

        # Save a checkpoint file.
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'vgg',
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': prec5,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

def train(train_loader, model, criterion, optimizer, epoch, print_freq=1):
    print('Starting a new training epoch!')
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - end)

        # Get input and label tensors, wrap them in variables.
        inputs, target = data
        target = target.cuda(async=True)
        inputs_var = torch.autograd.Variable(inputs)
        targets_var = torch.autograd.Variable(target)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass, backward propagation, and optimization.
        outputs = model(inputs_var)
        loss = criterion(outputs, targets_var)
        loss.backward()
        optimizer.step()

        # Update metrics.
        prec1, prec5 = accuracy(outputs, targets_var, topk=(1, 5))
        losses.update(loss.data[0], inputs_var.size(0))
        top1.update(prec1.data[0], inputs_var.size(0))
        top5.update(prec5.data[0], inputs_var.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Print out metrics periodically.
        if i % print_freq == 0:
            print_str = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t' \
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           epoch, i, len(train_loader), batch_time=batch_time,
                           data_time=data_time, loss=losses, top1=top1, top5=top5)
            print(print_str)
            log(print_str)

    log('------------------ Finished Training epoch! -----------------------\n')
    print('Finished Training epoch!')

def validate(val_loader, model, criterion, print_freq=1):
    print('Starting validation!')
    log('------------------- Starting validation! -----------------------\n')
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print_str = 'Validation: [{0}/{1}]\t'\
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
               i, len(val_loader), batch_time=batch_time, loss=losses,
               top1=top1, top5=top5)
            print(print_str)
            log(print_str)

    print_str = ' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5)
    print(print_str)
    log(print_str)

    print('Finished validation!')
    log('--------------- Finished validation! ----------------- \n')
    return top1.avg, top5.avg

if __name__ == '__main__':
    main()
