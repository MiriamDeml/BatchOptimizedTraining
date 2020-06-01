from __future__ import print_function
import os
import random
import numpy as np
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.parallel
import torch.utils.data
import time
from statistics import mean
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
from buildModels.dataloader import get_dataset
from buildModels.dataloader import get_transform, AverageMeter, accuracy
import buildModels.models as models
import buildModels.model_loader as model_loader

# Thesis: Batch-Optimized Training for Neural Networks by Miriam Deml
# Developed relying on the code by Hoffer et al. (2017): https://github.com/eladhoffer/bigBatch and Li et al. (2018): https://github.com/tomgoldstein/loss-landscape/blob/master/cifar10/main.py

def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
   


    if not training:
        with torch.no_grad():
            for i, (inputs, target) in enumerate(data_loader):
                target = target.cuda(async=True)
                # measure data loading time
                data_time.update(time.time() - end)
                input_var = Variable(inputs.type(args.type))
                target_var = Variable(target)

        
                output = model(input_var)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target_var.data, topk=(1, 5))
                losses.update(loss.data, input_var.size(0))
                top1.update(prec1, input_var.size(0))
                top5.update(prec5, input_var.size(0))

    else:
        lossCollection = []
        meansOfLosses = []
        updateCounter = 0
        for i, (inputs, target) in enumerate(data_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                target = target.cuda(async=True)
                input_var = Variable(inputs.type(args.type))
                target_var = Variable(target)

                mini_inputs = input_var.chunk(batch_size // args.mini_batch_size)
                mini_targets = target_var.chunk(batch_size // args.mini_batch_size)
                

                optimizer.zero_grad()

                # iterate over virtual ghost batches
                for k, mini_input_var in enumerate(mini_inputs):
                    mini_target_var = mini_targets[k]
                    output = model(mini_input_var)
                    loss = criterion(output, mini_target_var)
                    lossCollection.append(loss.data.item())

                    prec1, prec5 = accuracy(output.data, mini_target_var.data, topk=(1, 5))
                    losses.update(loss.data, mini_input_var.size(0))
                    top1.update(prec1, mini_input_var.size(0))
                    top5.update(prec5, mini_input_var.size(0))

                    # compute gradient and do SGD step
                    loss.backward()

                if len(lossCollection) > 1:
                    meansOfLosses.append(mean(lossCollection))
                else:
                    meansOfLosses.append(lossCollection[0])
                lossCollection = []
               
                update = True
                # check if update is necessary
                if (not args.LRD) and len(meansOfLosses) > 1 and epoch > 2 and epoch < 120:
                    secondLast = meansOfLosses[-2]
                    lastOne = meansOfLosses[-1]
                    if lastOne < secondLast:
                        update = False
                    
                if update:
                    for p in model.parameters():
                        p.grad.data.div_(len(mini_inputs))
                    clip_grad_norm_(model.parameters(), 5.)
                    # actual update of the model
                    optimizer.step()
                    updateCounter = updateCounter + 1


        # measure elapsed time
        print('Number of updates: ' + str(updateCounter))
        f.write('Number of updates: ' + str(updateCounter) + '\n')
        batch_time.update(time.time() - end)
        end = time.time()


    return {'loss': losses.avg,
            'prec1': top1.avg,
            'prec5': top5.avg}


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def test(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)

if __name__ == '__main__':
    # Training options
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--mini_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--optimizer', default='sgd', help='optimizer: sgd | adagrad | adadelta | adam')
    parser.add_argument('--weight_decay', default=0.0005, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--filename', default='folder1',help='path to save trained nets')
    parser.add_argument('--save_epoch', default=10, type=int, help='save every save_epochs')
    parser.add_argument('--gpu', type=int, default=0, help='the GPU to use')
    parser.add_argument('--rand_seed', default=0, type=int, help='seed for random num generator')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
    parser.add_argument('--saveState', default=False, help='if the states should be saved to enable loss landscape visualization')
    
    # model parameters
    parser.add_argument('--model', '-m', default='wrn_164')
    parser.add_argument('--loss_name', '-l', default='crossentropy', help='loss functions: crossentropy | mse')

    # data parameters
    parser.add_argument('--dataset', default='CIFAR10')
    
    # individual parameters
    parser.add_argument('--LRD', default=False, help='if LRD (True): do the baseline')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # Set the seed for reproducing the results
    torch.cuda.manual_seed(123)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    global lr
    lr = args.lr  # initial learning rate
    start_epoch = 1


    save_folder = args.filename
    if not os.path.exists('files/trained_nets/' + save_folder):
        os.makedirs('files/trained_nets/' + save_folder)

    f = open('files/trained_nets/' + save_folder + '/log.out', 'a')
    print('\nLearning Rate: %f' % args.lr)
    f.write('\nLearning rate: ' + str(lr) + '\n')

    # Model
    net = model_loader.load(args.model)
    print(net)
    f.write('\nModel: %f' + str(args.model))

    num_parameters = sum([l.nelement() for l in net.parameters()])
    print("number of parameters: %d", num_parameters) 
    
   # Data loading code
    default_transform = {'train': get_transform(args.dataset, input_size=None, augment=True),
                         'eval': get_transform(args.dataset, input_size=None, augment= False)}
   
    transform = getattr(net, 'input_transform', default_transform)
    
    criterion = getattr(net, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    net.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
        
    testloader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    train_data = get_dataset(args.dataset, 'train', transform['train'])
    
    global trainloader
    global batch_size
    batch_size = args.batch_size
    trainloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    

    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=args.lr)
    elif args.optimizer == 'adadelta':
        optimizer = optim.Adadelta(net.parameters(), lr=args.lr)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    print(optimizer)
    f.write('\nOptimizer: %f' + str(optimizer))

    init_weights = [w.data.cpu().clone() for w in list(net.parameters())]


    # record the performance of initial model
    train_result = test(trainloader, net, criterion, 0)
    train_loss, train_top1, train_top5 = [train_result[r] for r in ['loss', 'prec1', 'prec5']]
    train_err = 100 - train_top1
    test_result = test(testloader, net, criterion, 0)
    test_loss, test_top1, test_top5 = [test_result[r] for r in ['loss', 'prec1', 'prec5']]
    test_err = 100 - test_top1
    status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (0, train_loss, train_err, test_err, test_loss)
    print(status)
    f.write(status)

    if args.saveState:
        state = {
            'acc': 100 - test_err,
            'epoch': 0,
             'state_dict': net.state_dict()
        }
        opt_state = {
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, 'files/trained_nets/' + save_folder + '/model_0.t7')
        torch.save(opt_state, 'files/trained_nets/' + save_folder + '/opt_state_0.t7')

    for epoch in range(start_epoch, args.epochs + 1):
        if int(epoch) in [60, 120, 160]:
            if int(epoch) in [120, 160]:
                if int(epoch) in [120]:
                    batch_size = 16384
                    for param_group in optimizer.param_groups:
                        lr = 0.035
                        param_group['lr'] = lr
                        print('Learning rate: ' + str(lr))
                        f.write('\nLearning rate: ' + str(lr) + '\n')
                else:
                    batch_size = 32768
                trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
                print('Batch Size: ' + str(batch_size))
                f.write('\nBatch Size: %f' + str(batch_size))
            else:
                for param_group in optimizer.param_groups:
                    lr = lr / 2
                    param_group['lr'] = lr
                    print('Learning rate: ' + str(lr))
                    f.write('\nLearning rate: ' + str(lr) + '\n')
                    
            
      
                
        train_result = train(trainloader, net, criterion, epoch, optimizer)
        loss, train_top1, train_top5 = [train_result[r] for r in ['loss', 'prec1', 'prec5']]
        train_err = 100 - train_top1
        test_result = test(testloader, net, criterion, epoch)
        test_loss, test_top1, test_top5 = [test_result[r] for r in ['loss', 'prec1', 'prec5']]
        test_err = 100 - test_top1
        
        status = 'e: %d loss: %.5f train_err: %.3f test_top1: %.3f test_loss %.5f \n' % (epoch, loss, train_err, test_err, test_loss)
        print(status)
        f.write(status)

        # Save checkpoint.
        if args.saveState:
            acc = train_top1
            if epoch == 1 or epoch % args.save_epoch == 0 or epoch == 150:
                state = {
                    'acc': acc,
                    'epoch': epoch,
                    'state_dict': net.state_dict(),
                }
                opt_state = {
                    'optimizer': optimizer.state_dict()
                }
                torch.save(state, 'files/trained_nets/' + save_folder + '/model_' + str(epoch) + '.t7')
                torch.save(opt_state, 'files/trained_nets/' + save_folder + '/opt_state_' + str(epoch) + '.t7') 

    f.close()
    
    
    

    
