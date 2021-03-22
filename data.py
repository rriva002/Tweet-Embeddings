import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from boto3 import resource
from data_helpers import load_lt, load_tweets, precompute_images
from numpy import mean
import json as jsonmod


class TwitterDataset(data.Dataset):
    def __init__(self, start_date, num_tweets, word_model, graph_model,
                 transform=None, precomp_images=False, max_lt_tweets=0):
        self.word_model = word_model
        self.graph_model = graph_model
        self.transform = transform
        self.precomp_images = precomp_images
        self.max_lt_tweets = max_lt_tweets
        self.dataset = {}
        self.location_tweets = {}
        self.time_tweets = {}
        self.ids = []
        self.bucket = resource("s3").Bucket("twitter-rriva002")
        self.end_date = None
        self.avg_user = mean([graph_model[k] for k in graph_model.vocab], 0)
        data_dir, data_ext, data_filename = "data", "data", None

        for filename in os.listdir(data_dir):
            if filename.endswith(data_ext):
                index = filename.find("_")
                date = filename[:index]

                try:
                    num = int(filename[index + 1:filename.rfind(".")])
                except ValueError:
                    continue

                if start_date == date and num >= num_tweets:
                    data_filename = "/".join([data_dir, filename])
                    break

        if data_filename is None:
            data_filename = "{}/{}_{}.{}".format(data_dir, start_date,
                                                 num_tweets, data_ext)
            self.dataset, self.ids, self.end_date = load_tweets(start_date,
                                                                num_tweets,
                                                                data_filename,
                                                                self.bucket,
                                                                word_model)

            if precomp_images:
                precompute_images(data_filename, self.bucket, self.transform)

                images = self.__load_images(data_filename, data_ext)

                for id_str, tweet in self.dataset.items():
                    tweet = list(tweet)
                    tweet[1] = images[id_str]
                    self.dataset[id_str][1] = tuple(tweet)
        else:
            if precomp_images:
                images = self.__load_images(data_filename, data_ext)

            with open(data_filename, "r") as file:
                for tweet in file:
                    tweet = jsonmod.loads(tweet)

                    if "end_date" in tweet:
                        self.end_date = tweet["end_date"]
                    elif len(self.ids) < num_tweets:
                        id_str = tweet["id_str"]
                        text = tweet["text"]
                        text = [word for word in text if word >= 0]
                        image = tweet["image"]
                        image = images[id_str] if precomp_images else image
                        ht = tweet["hashtags"]
                        user = tweet["user"]
                        lt = tweet["lt"]
                        self.dataset[id_str] = (text, image, ht, user, lt)

                        self.ids.append(id_str)

        load_lt(self.dataset.values(), self.bucket, self.location_tweets,
                self.time_tweets, self.max_lt_tweets)

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        id_str = self.ids[index]
        text = self.dataset[id_str][0]
        image = self.dataset[id_str][1]
        hashtags = self.dataset[id_str][2].split()
        hashtag_id_set = set([self.word_model.get_word_id(h) for h in hashtags
                              if h in self.word_model])
        text = torch.tensor([t for t in text if t not in hashtag_id_set])
        hashtags = [self.word_model[hashtag] for hashtag in hashtags]
        hashtags = torch.tensor(hashtags)

        if self.dataset[id_str][3] in self.graph_model:
            user = torch.tensor(self.graph_model[self.dataset[id_str][3]])
        else:
            user = torch.tensor(self.avg_user)

        lt_key_split = self.dataset[id_str][4].split("_")
        location = self.location_tweets[lt_key_split[-1]]
        time = self.time_tweets[lt_key_split[0]]

        if not self.precomp_images:
            img_fn = image

            self.bucket.download_file("/".join(["images", img_fn]), img_fn)

            image = Image.open(img_fn).convert('RGB')

            if self.transform is not None:
                image = self.transform(image)

            os.remove(img_fn)

        return image, text, hashtags, user, location, time, index

    def __len__(self):
        return len(self.ids)

    def __load_images(self, data_filename, data_ext):
        img_fn = data_filename.replace("." + data_ext, "_img_feat.pt")
        images = torch.load(img_fn)

        for id_str, image in images.items():
            images[id_str] = image.squeeze(0)

        return images


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by text length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, texts, hashtag_sets, users, locations, times, ids = zip(*data)

    # Merge images and users (convert tuple of 3D tensor to 4D tensor)
    images = images if isinstance(images[0], str) else torch.stack(images, 0)
    users = torch.stack(users, 0)

    def set_up_targets(tweet_texts, lengths):
        if len(lengths) > 0:
            targets = torch.zeros(len(tweet_texts), max(lengths)).long()

            for i, text in enumerate(tweet_texts):
                end = lengths[i]
                targets[i, :end] = text[:end]
        else:
            return torch.zeros(0)

        return targets

    # Merge texts (convert tuple of 1D tensor to 2D tensor)
    text_lengths = [len(text) for text in texts]
    text_targets = set_up_targets(texts, text_lengths)
    loc_lengths = [[len(text) for text in l_texts] for l_texts in locations]
    time_lengths = [[len(text) for text in t_texts] for t_texts in times]
    loc_targets, time_targets = [], []

    for i, l_texts in enumerate(locations):
        loc_targets.append(set_up_targets(l_texts, loc_lengths[i]))

    for i, t_texts in enumerate(times):
        time_targets.append(set_up_targets(t_texts, time_lengths[i]))

    return images, text_targets, hashtag_sets, users, loc_targets, \
        time_targets, text_lengths, loc_lengths, time_lengths, ids


def get_loader_single(start_date, num_tweets, word_model, graph_model,
                      transform, precomp_images, max_lt_tweets, batch_size=100,
                      shuffle=True, num_workers=4, collate_fn=collate_fn):
    ds = TwitterDataset(start_date, num_tweets, word_model, graph_model,
                        transform=transform, precomp_images=precomp_images,
                        max_lt_tweets=max_lt_tweets)
    return torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size,
                                       shuffle=shuffle, pin_memory=True,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn), ds.end_date


def get_transform(train, opt):
    if train:
        t_list = [transforms.RandomResizedCrop(opt.crop_size),
                  transforms.RandomHorizontalFlip()]
    else:
        t_list = [transforms.Resize(256), transforms.CenterCrop(224)]

    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])
    return transforms.Compose(t_list + [transforms.ToTensor(), normalizer])


def get_loaders(start_date_train, num_train_tweets, start_date_val,
                num_val_tweets, word_model, graph_model, crop_size, batch_size,
                precomp_images, max_lt_tweets, workers, opt):
    train_loader, end_date = get_loader_single(start_date_train,
                                               num_train_tweets, word_model,
                                               graph_model,
                                               get_transform(True, opt),
                                               precomp_images, max_lt_tweets,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               collate_fn=collate_fn)

    print("Training date range: {} to {}".format(start_date_train, end_date))

    val_loader, end_date = get_loader_single(start_date_val, num_val_tweets,
                                             word_model, graph_model,
                                             get_transform(False, opt),
                                             precomp_images, max_lt_tweets,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=workers,
                                             collate_fn=collate_fn)

    print("Validation date range: {} to {}".format(start_date_val, end_date))
    return train_loader, val_loader


def get_test_loader(start_date, num_tweets, word_model, graph_model,
                    precomp_images, batch_size, workers, opt):
    test_loader, _ = get_loader_single(start_date, num_tweets, word_model,
                                       graph_model, get_transform(False, opt),
                                       precomp_images, opt.max_lt_tweets,
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=workers,
                                       collate_fn=collate_fn)
    return test_loader
