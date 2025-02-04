import argparse
import math
import torch
import torch.nn as nn
import os
import utils.icbhi_dataset
import sys
import time
from utils.icbhi_dataset import AverageMeter
from copy import deepcopy
from utils.metrics import accuracy, update_moving_average, get_score
from utils.utils import save_model, adjust_learning_rate, warmup_learning_rate
from torchvision import transforms
from models.projector import Projector
from models.ast_model import ASTModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model on the ICBHI dataset')
    parser.add_argument('--datadir', type=str, default='data', help='Path to the ICBHI dataset')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--sample_rate', type=int, default=16000, help='Target sample rate for audio')
    parser.add_argument('--num_of_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for training')
    parser.add_argument('--lr_decay_rate', type=float, default=1e-6, help='Learning rate for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--log-interval', type=int, default=10, help='Number of batches to wait before logging training status')
    parser.add_argument('--checkpoint-path', type=str, default='model.pth', help='Path to save model checkpoints')
    parser.add_argument('--log-path', type=str, default='log.txt', help='Path to save training logs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--warm', action='store_true',help='warm-up for large batch training')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for data loading')
    parser.add_argument('--diagnosis', type=str, \
        default='/home/richard/workspace/ICBHI-chanllenge/meta_data/ICBHI_Challenge_diagnosis.csv', help='Diagnosis to train on') 
    parser.add_argument('--filelist', type=str, \
        default='/home/richard/workspace/ICBHI-chanllenge/meta_data/ICBHI_challenge_train_test.csv', help='Path to the filelist')
    parser.add_argument('--save_folder', type=str, default='exps', \
        help='Folder to save checkpoints')
    args = parser.parse_args()
    
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    args.ma_beta = 0
    if args.warm:
        args.warmup_from = args.lr * 0.1
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.lr * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.lr - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.lr
    args.h, args.w = 798, 128
    args.n_cls = 4
    # if args.dataset == 'icbhi':
    #     if args.class_split == 'lungsound':
    #         if args.n_cls == 4:
    #             args.cls_list = ['normal', 'crackle', 'wheeze', 'both']
    #         elif args.n_cls == 2:
    #             args.cls_list = ['normal', 'abnormal']
    #     elif args.class_split == 'diagnosis':
    #         if args.n_cls == 3:
    #             args.cls_list = ['healthy', 'chronic_diseases', 'non-chronic_diseases']
    #         elif args.n_cls == 2:
    #             args.cls_list = ['healthy', 'unhealthy']
    # else:
    #     raise NotImplementedError
    return args

def train(args, model, classifier, projector, train_loader, optimizer, criterion, scaler, epoch):
    losses = AverageMeter()
    date_time = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        # train
        with torch.no_grad():
            ma_ckpt = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
        images = images.squeeze(0)
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)
        
        with torch.amp.autocast('cuda'):
            # print(images.shape)
            features = model(images)
            output = classifier(features)
            loss = criterion(output, labels)
        
        bsz = images.size(0) # batch size
        losses.update(loss.item(), bsz)
        [acc1], _ = accuracy(output[:bsz], labels, topk=(1,)) # calculate accuracy
        top1.update(acc1[0], bsz) # update accuracy

        optimizer.zero_grad() # zero the parameter gradients
        scaler.scale(loss).backward() # backpropagation
        scaler.step(optimizer) # update the weights
        scaler.update() # update the scale for next iteration
    
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # moving average update, to smooth the process of model updates
        # with torch.no_grad():
        #     # exponential moving average update
        #     model = update_moving_average(args.ma_beta, model, ma_ckpt[0])
        #     classifier = update_moving_average(args.ma_beta, classifier, ma_ckpt[1])
                
        # print info
        if (idx + 1) % args.log_interval == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DateTime {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=date_time, loss=losses, top1=top1))
            sys.stdout.flush()
    return losses.avg, top1.avg
            
def validate(args, model: nn.Module, classifier: nn.Module, projector, \
    valid_loader, best_acc: tuple, criterion, best_model):
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    hits, counts = [0.0] * args.n_cls, [0.0] * args.n_cls
    end = time.time()
    for idx, (images, labels) in enumerate(valid_loader):
        # validate
        with torch.amp.autocast('cuda'):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            
            features = model(images)
            output = classifier(features)
            loss = criterion(output, labels)
            
            losses.update(loss.item(), bsz)
            [acc1], _ = accuracy(output, labels, topk=(1,))
            top1.update(acc1[0], bsz)

            _, preds = torch.max(output, 1)
            for idx in range(preds.shape[0]):
                counts[labels[idx].item()] += 1.0
                if labels[idx].item() == 0 and preds[idx].item() == labels[idx].item():
                    hits[labels[idx].item()] += 1.0
                elif labels[idx].item() != 0 and preds[idx].item() > 0:  # abnormal
                    hits[labels[idx].item()] += 1.0
            # sp: specificity, se: sensitivity, sc: score
            sp, se, sc = get_score(hits, counts)

            batch_time.update(time.time() - end)
            end = time.time()

            if (idx + 1) % args.log_interval == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(valid_loader), batch_time=batch_time,
                       loss=losses, top1=top1))
    
    if (sc > best_acc[-1] and se > 5) or best_model[0] is None:
        save_bool = True
        best_acc = [sp, se, sc]
        best_model = [deepcopy(model.state_dict()), deepcopy(classifier.state_dict())]
    else:
        save_bool = False

    print(' * S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f} (Best S_p: {:.2f}, S_e: {:.2f}, Score: {:.2f})'.format(sp, se, sc, best_acc[0], best_acc[1], best_acc[-1]))
    print(' * Acc@1 {top1.avg:.2f}'.format(top1=top1))

    return best_acc, best_model, save_bool

def main(args):
    nmels = 128
    segment_length = 798
    best_acc = [0, 0, 0]  # Specificity, Sensitivity, Score
    best_model = [None, None]
    train_transform = [\
        transforms.ToTensor(),\
        transforms.Resize(size=(int(args.h), int(args.w)))]
    val_transform = [\
        transforms.ToTensor(),\
        transforms.Resize(size=(int(args.h), int(args.w)))]         
    train_transform = transforms.Compose(train_transform)
    val_transform = transforms.Compose(val_transform)
    
    train_dataset = utils.icbhi_dataset.ICBHI_Dataset(args.diagnosis, args.filelist, args.datadir, type_mode='train',\
        n_mels=nmels, segment_length=segment_length, sample_rate=args.sample_rate, transform=train_transform)
    test_dataset = utils.icbhi_dataset.ICBHI_Dataset(args.diagnosis, args.filelist, args.datadir, type_mode='test',\
        n_mels=nmels, segment_length=segment_length, sample_rate=args.sample_rate, transform=val_transform)
    # train_transform.append(transforms.Normalize(mean=mean, std=std))
    # val_transform.append(transforms.Normalize(mean=mean, std=std))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                               num_workers=args.num_workers, pin_memory=True, sampler=None, drop_last=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True, sampler=None)
    
    scaler = torch.amp.GradScaler()
    criterion = nn.CrossEntropyLoss().cuda()
    model = ASTModel(\
        label_dim=args.n_cls, \
        fstride=10, tstride=10, input_fdim=nmels, input_tdim=segment_length, \
        imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True).cuda()
    
    classifier = deepcopy(model.mlp_head).cuda()
        
    optim_params = model.parameters()
    optimizer = torch.optim.Adam(optim_params, lr=args.lr)
    
    projector = nn.Identity().cuda()
    for epoch in range(args.num_of_epochs):
        loss, acc = train(args, model, classifier, projector, train_loader, optimizer, criterion, scaler, epoch)
        best_acc, best_model, save_bool = validate(args, model, classifier, projector, val_loader, best_acc, criterion, best_model)
        
        # save a checkpoint of model and classifier when the best score is updated
        if save_bool:            
            save_file = os.path.join(args.save_folder, 'best_epoch_{}.pth'.format(epoch))
            print('Best ckpt is modified with Score = {:.2f} when Epoch = {}'.format(best_acc[2], epoch))
            save_model(model, optimizer, args, epoch, save_file, classifier)
            
        if epoch % args.log_interval == 0:
            save_file = os.path.join(args.save_folder, 'epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, args, epoch, save_file, classifier)

    # save a checkpoint of classifier with the best accuracy or score
    save_file = os.path.join(args.save_folder, 'best.pth')
    model.load_state_dict(best_model[0])
    classifier.load_state_dict(best_model[1])
    save_model(model, optimizer, args, epoch, save_file, classifier)

if __name__ == '__main__':
    args = parse_args()
    main(args)
    