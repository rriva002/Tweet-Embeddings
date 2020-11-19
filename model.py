import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from collections import OrderedDict
from itertools import combinations


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def EncoderImage(img_dim, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False, precomp=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if precomp:
        return EncoderImagePrecomp(img_dim, embed_size, use_abs=use_abs,
                                   no_imgnorm=no_imgnorm)

    return EncoderImageFull(embed_size, finetune, cnn_type, use_abs,
                            no_imgnorm)


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):
    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                                embed_size)
            self.cnn.classifier = nn.Sequential(
                *list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def load_state_dict(self, state_dict):
        """
        Handle the models saved before commit pytorch/vision@989d52a
        """
        if 'cnn.classifier.1.weight' in state_dict:
            state_dict['cnn.classifier.0.weight'] = state_dict[
                'cnn.classifier.1.weight']
            del state_dict['cnn.classifier.1.weight']
            state_dict['cnn.classifier.0.bias'] = state_dict[
                'cnn.classifier.1.bias']
            del state_dict['cnn.classifier.1.bias']
            state_dict['cnn.classifier.3.weight'] = state_dict[
                'cnn.classifier.4.weight']
            del state_dict['cnn.classifier.4.weight']
            state_dict['cnn.classifier.3.bias'] = state_dict[
                'cnn.classifier.4.bias']
            del state_dict['cnn.classifier.4.bias']

        super(EncoderImageFull, self).load_state_dict(state_dict)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        self.fc = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):
    def __init__(self, wt, word_dim, embed_size, num_layers, use_abs=False):
        super(EncoderText, self).__init__()

        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding.from_pretrained(wt, freeze=True)

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True)

    def forward(self, x, lengths, normalize=True):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(len(lengths), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        if normalize:
            out = l2norm(out)

            # take absolute value, used by order embeddings
            if self.use_abs:
                out = torch.abs(out)

        return out


class EncoderHashtags(nn.Module):
    def __init__(self, word_dim, embed_size, use_abs=False):
        super(EncoderHashtags, self).__init__()

        self.use_abs = use_abs
        self.embed_size = embed_size
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(word_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)

        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        averages = []

        for hashtag in x:
            out = self.avg(torch.transpose(hashtag.unsqueeze(0), 1, 2))
            out = torch.transpose(out, 1, 2).squeeze(1).squeeze(0)

            averages.append(out)

        out = l2norm(self.fc(torch.stack(averages, 0)))
        return torch.abs(out) if self.use_abs else out


class EncoderUser(nn.Module):
    def __init__(self, graph_model, embed_size, use_abs=False):
        super(EncoderUser, self).__init__()

        self.use_abs = use_abs
        self.embed_size = embed_size
        self.fc = nn.Linear(graph_model.vector_size, embed_size)

        self.init_weights()

    def init_weights(self):
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)

        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        out = l2norm(self.fc(x))
        return torch.abs(out) if self.use_abs else out


class EncoderLocationTime(nn.Module):
    def __init__(self, wt, word_dim, embed_size, num_layers, use_abs=False):
        super(EncoderLocationTime, self).__init__()

        self.txt_enc = EncoderText(wt, word_dim, embed_size, num_layers)
        self.use_abs = use_abs
        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x, lengths):
        averages = []

        for i, length in enumerate(lengths):
            out = self.txt_enc(x[i], length, normalize=False)
            out = self.avg(torch.transpose(out.unsqueeze(0), 1, 2))

            averages.append(torch.transpose(out, 1, 2).squeeze(1))

        out = l2norm(torch.stack(averages, 0).squeeze(1))
        return torch.abs(out) if self.use_abs else out


def cosine_sim(a, b):
    """Cosine similarity between all the image and sentence pairs
    """
    return a.mm(b.t())


def order_sim(a, b):
    """Order embeddings similarity measure $max(0, b-a)$
    """
    YmX = (b.unsqueeze(1).expand(b.size(0), a.size(0), b.size(1))
           - a.unsqueeze(0).expand(b.size(0), a.size(0), b.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False,
                 target=None):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.sim = order_sim if measure == "order" else cosine_sim
        self.max_violation = max_violation
        self.target = target

    def __loss(self, a, b):
        scores = self.sim(a, b)
        diagonal = scores.diag().view(a.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)
        cost_b = (self.margin + scores - d1).clamp(min=0)
        cost_a = (self.margin + scores - d2).clamp(min=0)
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)

        if torch.cuda.is_available():
            I = I.cuda()

        cost_b = cost_b.masked_fill_(I, 0)
        cost_a = cost_a.masked_fill_(I, 0)

        if self.max_violation:
            cost_b = cost_b.max(1)[0]
            cost_a = cost_a.max(0)[0]

        return cost_b.sum() + cost_a.sum()

    def forward(self, embeddings):
        n_embeds = len(embeddings)

        if self.target == "average":
            target = torch.mean(torch.stack(embeddings), dim=0)
            ls = [self.__loss(target, embeddings[i]) for i in range(n_embeds)]
            return sum(ls) / len(embeddings)
        elif self.target is not None:
            embs = [embeddings[i] for i in range(n_embeds) if i != self.target]
            target = embeddings[self.target]
            ls = [self.__loss(target, embs[i]) for i in range(len(embs))]
            return sum(ls) / len(embs)

        combos = list(combinations(list(range(n_embeds)), r=2))
        ls = [self.__loss(embeddings[c[0]], embeddings[c[1]]) for c in combos]

        return sum(ls) / len(combos)


class VSE(object):
    """
    rkiros/uvs model
    """

    def __init__(self, opt, word_model, graph_model):
        # tutorials/09 - Image Captioning
        # Build Models
        wt = torch.FloatTensor([word_model[w] for w in word_model.get_words()])
        word_dim = word_model.get_dimension()
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size, opt.finetune,
                                    opt.cnn_type, use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm,
                                    precomp=opt.precomp_images)
        self.txt_enc = EncoderText(wt, word_dim, opt.embed_size,
                                   opt.num_layers, use_abs=opt.use_abs)
        self.ht_enc = EncoderHashtags(word_dim, opt.embed_size,
                                      use_abs=opt.use_abs)
        self.usr_enc = EncoderUser(graph_model, opt.embed_size,
                                   use_abs=opt.use_abs)
        self.loc_enc = EncoderLocationTime(wt, word_dim, opt.embed_size,
                                           opt.num_layers, use_abs=opt.use_abs)
        self.tme_enc = EncoderLocationTime(wt, word_dim, opt.embed_size,
                                           opt.num_layers, use_abs=opt.use_abs)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.ht_enc.cuda()
            self.usr_enc.cuda()
            self.loc_enc.cuda()
            self.tme_enc.cuda()

            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin, target=opt.target,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())

        params += list(self.ht_enc.parameters())
        params += list(self.usr_enc.parameters())
        params += list(self.loc_enc.parameters())
        params += list(self.tme_enc.parameters())
        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.ht_enc.state_dict(), self.usr_enc.state_dict(),
                      self.loc_enc.state_dict(), self.tme_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])
        self.ht_enc.load_state_dict(state_dict[2])
        self.usr_enc.load_state_dict(state_dict[3])
        self.loc_enc.load_state_dict(state_dict[4])
        self.tme_enc.load_state_dict(state_dict[5])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.ht_enc.train()
        self.usr_enc.train()
        self.loc_enc.train()
        self.tme_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.ht_enc.eval()
        self.usr_enc.eval()
        self.loc_enc.eval()
        self.tme_enc.eval()

    def forward_emb(self, data, components, text_lengths=None,
                    loc_lengths=None, time_lengths=None):
        embeddings = []

        for i, component in enumerate(components):
            if isinstance(data[i], list):
                embedding = [Variable(d) for d in data[i]]
            else:
                embedding = Variable(data[i])

            if torch.cuda.is_available():
                if isinstance(embedding, list):
                    embedding = [e.cuda() for e in embedding]
                else:
                    embedding = embedding.cuda()

            if component == "image":
                embeddings.append(self.img_enc(embedding))
            elif component == "text":
                embeddings.append(self.txt_enc(embedding, text_lengths))
            elif component == "hashtags":
                embeddings.append(self.ht_enc(embedding))
            elif component == "user":
                embeddings.append(self.usr_enc(embedding))
            elif component == "location":
                embeddings.append(self.loc_enc(embedding, loc_lengths))
            elif component == "time":
                embeddings.append(self.tme_enc(embedding, time_lengths))

        return embeddings

    def forward_loss(self, embeddings, **kwargs):
        loss = self.criterion(embeddings)

        self.logger.update('Le', loss.item(), embeddings[0].size(0))
        return loss

    def train_emb(self, data, components, text_lengths=None, loc_lengths=None,
                  time_lengths=None, ids=None, *args):
        self.Eiters += 1

        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        embeddings = self.forward_emb(data, components, text_lengths,
                                      loc_lengths, time_lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        loss = self.forward_loss(embeddings)

        # compute gradient and do SGD step
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)

        self.optimizer.step()
