import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import time
import shutil

import torch

import data
from fasttext import load_model
from gensim.models import KeyedVectors
from model import VSE
from evaluation import c2c, AverageMeter, LogCollector, encode_data

import logging
import tensorboard_logger as tb_logger

import argparse


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_date_train', default='20-03-01', type=str,
                        help='Earliest post date of training tweets.')
    parser.add_argument('--num_train_tweets', default=100000, type=int,
                        help='Number of tweets to use for training.')
    parser.add_argument('--start_date_val', default='20-03-09', type=str,
                        help='Earliest post date of validation tweets.')
    parser.add_argument('--num_val_tweets', default=5000, type=int,
                        help='Number of tweets to use for validation.')
    parser.add_argument('--max_lt_tweets', default=0, type=int,
                        help='Maximum number of location/time tweets.')
    parser.add_argument('--text_model_path', default="./models/tweet_text.bin",
                        type=str, help='Path to saved word embedding.')
    parser.add_argument('--graph_model_path',
                        default="./models/graph_embedding.vec", type=str,
                        help='Path to saved graph embedding.')
    parser.add_argument('--precomp_images', action='store_true',
                        help='Use precomputed images.')
    parser.add_argument('--margin', default=0.2, type=float,
                        help='Rank loss margin.')
    parser.add_argument('--num_epochs', default=30, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--crop_size', default=224, type=int,
                        help='Size of an image crop as the CNN input.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lr_update', default=15, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--workers', default=10, type=int,
                        help='Number of data loader workers.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--val_step', default=500, type=int,
                        help='Number of steps to run validation.')
    parser.add_argument('--logger_name', default='runs/runX',
                        help='Path to save the model and Tensorboard log.')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--max_violation', action='store_true',
                        help='Use max instead of sum in the rank loss.')
    parser.add_argument('--img_dim', default=2048, type=int,
                        help='Dimensionality of the image embedding.')
    parser.add_argument('--finetune', action='store_true',
                        help='Fine-tune the image encoder.')
    parser.add_argument('--cnn_type', default='resnet152',
                        help="""The CNN used for image encoder
                        (e.g. vgg19, resnet152)""")
    parser.add_argument('--measure', default='cosine',
                        help='Similarity measure used (cosine|order)')
    parser.add_argument('--max_similarity', action='store_true',
                        help='Use max instead of sum for similarity scores.')
    parser.add_argument('--use_abs', action='store_true',
                        help='Take the absolute value of embedding vectors.')
    parser.add_argument('--no_imgnorm', action='store_true',
                        help='Do not normalize the image embeddings.')
    parser.add_argument('--components', type=str,
                        default='image,text,hashtags,user,location,time',
                        help='Tweet components to consider during training.')
    parser.add_argument('--target', type=str, default=None,
                        help='Optional target for loss function.')
    parser.add_argument('--k_vals', default='1,5,10',
                        help='Tuple containing values of k for recall@k.')

    opt = parser.parse_args()

    print(opt)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load data loaders
    opt.k_vals = [int(k) for k in opt.k_vals.split(",")]
    word_model = load_model(opt.text_model_path)
    graph_model = KeyedVectors.load_word2vec_format(opt.graph_model_path)
    # limit=581850)
    train_loader, val_loader = data.get_loaders(opt.start_date_train,
                                                opt.num_train_tweets,
                                                opt.start_date_val,
                                                opt.num_val_tweets, word_model,
                                                graph_model, opt.crop_size,
                                                opt.batch_size,
                                                opt.precomp_images,
                                                opt.max_lt_tweets,
                                                opt.workers, opt)

    # Construct the model
    components = ["image", "text", "hashtags", "user", "location", "time"]
    component_index = dict([(c, i) for i, c in enumerate(components)])
    components = opt.components.split(",")

    if opt.target is not None and opt.target != "average":
        opt.target = components.index(opt.target)

    model = VSE(opt, word_model, graph_model)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model, components)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0

    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader, components,
              component_index)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model, components, component_index)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        prefix = opt.logger_name + '/'

        save_checkpoint({'epoch': epoch + 1, 'model': model.state_dict(),
                         'best_rsum': best_rsum, 'opt': opt,
                         'Eiters': model.Eiters}, is_best, prefix=prefix)


def train(opt, train_loader, model, epoch, val_loader, components,
          component_index):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()
    end = time.time()

    for i, train_data in enumerate(train_loader):
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger
        data = [train_data[component_index[c]] for c in components]
        text_lengths = train_data[-4] if "text" in components else None
        l_lengths = train_data[-3] if "location" in components else None
        time_lengths = train_data[-2] if "time" in components else None

        # Update the model
        model.train_emb(data, components, text_lengths=text_lengths,
                        loc_lengths=l_lengths, time_lengths=time_lengths)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                .format(epoch, i, len(train_loader), batch_time=batch_time,
                        data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model, components, component_index)


def validate(opt, val_loader, model, components, component_index):
    embeddings = encode_data(model, val_loader, components, component_index,
                             opt.log_step, logging.info,
                             on_gpu=torch.cuda.is_available())
    currscore = 0
    pfx = ["r" + str(k) for k in opt.k_vals] + ["medr", "meanr"]

    for component in components:
        r = c2c(embeddings, components, component=component,
                measure=opt.measure, max_similarity=opt.max_similarity,
                k_vals=opt.k_vals)
        # sum of recalls to be used for early stopping
        currscore += sum(r[:-2])

        # record metrics in tensorboard
        logging.info(("Components to %s: " + ", ".join(["%.1f"] * len(r))) %
                     (component, *r))
        for i in range(len(r)):
            tb_logger.log_value(pfx[i] + component[0], r[i], step=model.Eiters)

    currscore /= len(components)

    tb_logger.log_value('rsum', currscore, step=model.Eiters)
    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    torch.save(state, prefix + filename)

    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
