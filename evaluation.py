from __future__ import print_function

import numpy
from data import get_test_loader
import argparse
import time
import numpy as np
import torch
from model import VSE, order_sim
from collections import OrderedDict
from fasttext import load_model
from gensim.models import KeyedVectors


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, components, component_index, log_step=10,
                logging=print, on_gpu=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    embs = [None for i in range(len(components))]

    with torch.no_grad():
        for i, batch_data in enumerate(data_loader):
            data = [batch_data[component_index[c]] for c in components]
            text_lengths = batch_data[-4] if "text" in components else None
            l_lens = batch_data[-3] if "location" in components else None
            time_lengths = batch_data[-2] if "time" in components else None
            ids = batch_data[-1]

            # make sure val logger is used
            model.logger = val_logger

            # compute the embeddings
            embeddings = model.forward_emb(data, components, text_lengths,
                                           l_lens, time_lengths)

            for j, embedding in enumerate(embeddings):
                if embs[j] is None:
                    len_dataset = len(data_loader.dataset)
                    embs[j] = np.zeros((len_dataset, embedding.size(1)))

                if on_gpu:
                    embs[j][ids] = embedding.data.cpu().numpy().copy()
                else:
                    embs[j][ids] = embedding.data.numpy().copy()

            model.forward_loss(embeddings)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        .format(
                            i, len(data_loader), batch_time=batch_time,
                            e_log=str(model.logger)))
            del embeddings

    return embs


def evalrank(model_path, start_date, num_tweets, fold5=False, on_gpu=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    device = 'cpu' if not on_gpu else 'cuda'
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    opt = checkpoint['opt']

    word_model = load_model(opt.text_model_path)
    graph_model = KeyedVectors.load_word2vec_format(opt.graph_model_path)

    # construct model
    model = VSE(opt, word_model, graph_model)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(start_date, num_tweets, word_model,
                                  graph_model, opt.precomp_images,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')

    rt = {}
    components = ["image", "text", "hashtags", "user", "location", "time"]
    component_index = dict([(c, i) for i, c in enumerate(components)])
    components = opt.components.split(",")
    embeddings = encode_data(model, data_loader, components, component_index,
                             on_gpu=on_gpu)
    fmt = ", ".join(["{}{}%d".format(c[0].upper(), c[1:]) for c in components])

    print(fmt % tuple([embedding.shape[0] for embedding in embeddings]))

    if not fold5:
        # no cross-validation, full evaluation
        rsum = 0

        for component in components:
            r, rtc = c2c(embeddings, components, component=component,
                         measure=opt.measure, k_vals=opt.k_vals,
                         max_similarity=opt.max_similarity, return_ranks=True)
            ar = sum(r[:-2]) / (len(r) - 2)
            rsum += sum(r[:-2])
            rt["rt" + ("x" if component == "text" else component[0])] = rtc

            print("Average c2%s Recall: %.1f" % (component[0], ar))
            print(("Components to %s: " + " ".join(["%.1f"] * len(r))) %
                  tuple([component] + list(r)))

        print("rsum: %.1f" % rsum)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        fmt = " ".join(["ar{}: %.1f".format(c[0]) for c in components])

        for i in range(5):
            start, end = i * 5000, (i + 1) * 5000
            ar, rsum = {}, 0

            results.append([])

            for component in components:
                r, rt0 = c2c([e[start:end] for e in embeddings], components,
                             component=component, measure=opt.measure,
                             return_ranks=True, k_vals=opt.k_vals)
                ar[component] = sum(r[:-2]) / (len(r) - 2)
                rsum += sum(r[:-2])
                results[-1] += list(r)

                if i == 0:
                    key = "rt" + ("x" if component == "text" else component[0])
                    rt[key] = rt0

                print(("Components to %s: " + " ".join(["%.1f"] * len(r))) %
                      tuple([component] + list(r)))

            results[-1] += [rsum] + [ar[component] for component in components]

            print("rsum: " + fmt % tuple(results[-1][-(len(components) + 1):]))

        print("-----------------------------------\nMean metrics: ")

        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        index = 5 * len(components)

        print("rsum: %.1f" % (mean_metrics[index] * 3 * len(components)))

        for i, c in enumerate(components):
            print("Average c2%s Recall: %.1f" %
                  (c[0], mean_metrics[index + 1 + i]))
            print(("Components to %s:" + " %.1f" * len(components)) %
                  tuple([c] + list(mean_metrics[i:i + len(components)])))

    torch.save(rt, 'ranks.pth.tar')


def c2c(embeddings, components, component="text", measure='cosine',
        max_similarity=False, return_ranks=False, k_vals=[1, 5, 10]):
    """
    Images, hashtags, users, location, time->Text (Annotation)
    Images: (5N, K) matrix of images
    Text: (5N, K) matrix of text
    Hashtags: (5N, K) matrix of hashtags
    Users: (5N, K) matrix of users
    Location: (5N, K) matrix of location
    Time: (5N, K) matrix of time
    """
    npts = embeddings[0].shape[0]
    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)

    for i, c in enumerate(components):
        if component == c:
            target = embeddings[i]
            break

    target = torch.Tensor(target).cuda() if measure == "order" else target

    for index in range(npts):
        query_components, scores = [], []

        for i, c in enumerate(components):
            if component != c:
                qc = embeddings[i][index].reshape(1, embeddings[i].shape[1])

                query_components.append(qc)

        # Compute scores
        if measure == 'order':
            bs = 100

            if index % bs == 0:
                mx = min(embeddings[0].shape[0], index + bs)

                for i in range(len(query_components)):
                    qc = torch.Tensor(query_components[i][index:mx:1]).cuda()

                    scores.append(order_sim(qc, target).cpu().numpy())

                if max_similarity:
                    d2 = numpy.max(scores, axis=0)
                else:
                    d2 = numpy.sum(scores, axis=0)

            d = d2[index % bs]
        else:
            for query_component in query_components:
                d = numpy.dot(query_component, target.T).flatten()

                scores.append(d)

            if max_similarity:
                d = numpy.max(scores, axis=0)
            else:
                d = numpy.sum(scores, axis=0)

        # Score
        inds = numpy.argsort(d)[::-1]
        ranks[index] = numpy.where(inds == index)[0][0]
        top1[index] = inds[0]

    # Compute metrics
    lr = len(ranks)
    recalls = [100.0 * len(numpy.where(ranks < k)[0]) / lr for k in k_vals]
    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return tuple(recalls + [medr, meanr]), (ranks, top1)
    else:
        return tuple(recalls + [medr, meanr])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path', default='runs/runX/model_best.pth.tar',
                        type=str, help='Path to trained model.')
    parser.add_argument('--start_date_test', default='20-03-10', type=str,
                        help='Earliest post date of test tweets.')
    parser.add_argument('--num_test_tweets', default=5000, type=int,
                        help='Number of tweets to use for testing.')

    opt = parser.parse_args()

    print(opt)
    evalrank(opt.model_path, opt.start_date_test, opt.num_test_tweets,
             on_gpu=torch.cuda.is_available())
