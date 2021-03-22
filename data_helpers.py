import torch
import torchvision.models as models
from botocore.exceptions import ClientError
from itertools import groupby
from json import dumps, loads
from nltk.tokenize import word_tokenize
from os import remove
from os.path import exists
from PIL import Image, UnidentifiedImageError
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from random import sample
from re import sub
from queue import Queue
from threading import Lock, Thread
from torch.nn import DataParallel, Sequential
from zipfile import ZipFile


class ImagePreprocessor(Thread):
    def __init__(self, queue, bucket, cnn, image_dict, transform, *args,
                 **kwargs):
        self.__queue = queue
        self.__bucket = bucket
        self.__cnn = cnn
        self.__image_dict = image_dict
        self.__transform = transform

        super().__init__(*args, **kwargs)

    def __l2norm(self, X):
        return torch.div(X, torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt())

    def run(self):
        while not self.__queue.empty():
            tweet = self.__queue.get()

            if "id_str" in tweet:
                id_str = tweet["id_str"]

                filename = tweet["image"]
                key = "/".join(["images", filename])

                if self.__bucket.Object(key).content_length > 0:
                    self.__bucket.download_file(key, filename)

                    try:
                        image = Image.open(filename).convert("RGB")
                        image = self.__transform(image).unsqueeze(0)

                        if torch.cuda.is_available():
                            image = image.cuda()

                        features = self.__l2norm(self.__cnn(image))
                        features = features.clone().detach().cpu()
                        self.__image_dict[id_str] = features
                    except UnidentifiedImageError:
                        pass

                    if exists(filename):
                        remove(filename)

            self.__queue.task_done()


class LTLoader(Thread):
    def __init__(self, bucket, location_tweets, time_tweets, max_lt_tweets,
                 queue, location_lock, time_lock, dl=True, *args, **kwargs):
        self.bucket = bucket
        self.location_tweets = location_tweets
        self.time_tweets = time_tweets
        self.max_lt_tweets = max_lt_tweets
        self.queue = queue
        self.location_lock = location_lock
        self.time_lock = time_lock
        self.dl = dl

        super().__init__(*args, **kwargs)

    def __load_lt_text(self, lt_key):
        lt = []

        self.bucket.download_file("/".join(["lt", lt_key]), lt_key)

        with open(lt_key, "r") as file:
            for line in file:
                if len(line) > 0:
                    lt_text = loads(line)

                    if len(lt_text) > 0:
                        lt.append(lt_text)

        if exists(lt_key):
            remove(lt_key)

        return lt

    def run(self):
        dictionaries = [self.location_tweets, self.time_tweets]
        locks = [self.location_lock, self.time_lock]

        while not self.queue.empty():
            lt_key = self.queue.get()

            if self.dl:
                lt_key_split = lt_key.split("_")
                keys = [lt_key_split[-1], lt_key_split[0]]
                lt = self.__load_lt_text(lt_key)

                for i, dictionary in enumerate(dictionaries):
                    locks[i].acquire()

                    if keys[i] not in dictionary:
                        dictionary[keys[i]] = []

                    dictionary[keys[i]] += lt

                    locks[i].release()
            else:
                for i, dictionary in enumerate(dictionaries):
                    if lt_key in dictionary:
                        lt = dictionary[lt_key]
                        lt = [txt for txt, _ in groupby(sorted(lt))]

                        if self.max_lt_tweets > 0:
                            lt = sample(lt, min(len(lt), self.max_lt_tweets))

                        lt = sorted(lt, key=lambda x: len(x), reverse=True)
                        dictionary[lt_key] = [torch.tensor(txt) for txt in lt]

            self.queue.task_done()


def load_lt(tweets, bucket, location_tweets, time_tweets, max_lt_tweets):
    queue, location_lock, time_lock = Queue(), Lock(), Lock()
    lt_keys, num_threads = set(), 10

    for tweet in tweets:
        lt_keys.add(tweet[-1])

    for lt_key in lt_keys:
        queue.put(lt_key)

    del lt_keys

    for _ in range(num_threads):
        LTLoader(bucket, location_tweets, time_tweets, max_lt_tweets, queue,
                 location_lock, time_lock).start()

    queue.join()

    for l_key in location_tweets.keys():
        queue.put(l_key)

    for t_key in time_tweets.keys():
        queue.put(t_key)

    for _ in range(num_threads):
        LTLoader(bucket, location_tweets, time_tweets, max_lt_tweets, queue,
                 location_lock, time_lock, dl=False).start()

    queue.join()


def load_tweets(start_date, num_tweets, data_filename, bucket, word_model):
    drive = GoogleDrive(GoogleAuth().CommandLineAuth())
    query = "title contains 'geoTwitter' and trashed=false"
    files, skip = drive.ListFile({"q": query}).GetList(), True
    ids, dataset, end_date = [], {}, None

    def convert_text(text):
        text = sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str(text))
        text = sub(r"\'ve", " \'ve", sub(r"\'s", " \'s", text))
        text = sub(r"\'re", " \'re", sub(r"n\'t", " n\'t", text))
        text = sub(r"\'ll", " \'ll", sub(r"\'d", " \'d", text))
        text = sub(r"\(", " ( ", sub(r"!", " ! ", sub(r",", " , ", text)))
        text = sub(r"\s{2,}", " ", sub(r"\?", " ? ", sub(r"\)", " ) ", text)))
        return [word_model.get_word_id(t) for t in word_tokenize(text.lower())]

    def extract_hashtags(tweet):
        ht = " ".join([h["text"] for h in tweet["entities"]["hashtags"]])
        return ht.lower() if len(ht) > 0 else None

    def extract_image(tweet):
        url, ee, m = None, "extended_entities", "media"

        if ee in tweet and m in tweet[ee]:
            for media in tweet[ee][m]:
                if media["type"] == "photo":
                    url = media["media_url"]
                    break

        if url is not None:
            filename = tweet["id_str"] + url[url.rfind("."):]
            k = "/".join(["images", filename])

            try:
                if bucket.Object(k).content_length > 0:
                    return filename
            except ClientError:
                pass

        return None

    def extract_location(tweet):
        u = (-171.791110603, 18.91619, -66.96466, 71.3577635769)
        p, b, c = "place", "bounding_box", "coordinates"

        if p not in tweet or tweet[p] is None:
            return None
        elif b not in tweet[p] or tweet[p][b] is None:
            return None
        elif c not in tweet[p][b] or tweet[p][b][c] is None:
            return None

        bb = tweet[p][b][c][0]
        min_x, max_x, min_y, max_y = bb[0][0], bb[0][0], bb[0][1], bb[0][1]

        for c in bb:
            min_x, max_x = min(min_x, c[0]), max(min_x, c[0])
            min_y, max_y = min(min_y, c[1]), max(min_y, c[1])

        if min_x > u[0] and min_y > u[1] and max_x < u[2] and max_y < u[3]:
            return tweet[p]["id"]

        return None

    def extract_text(tweet):
        et, ft, t = "extended_tweet", "full_text", "text"

        if et in tweet and ft in tweet[et] and len(tweet[et][ft]) > 0:
            return tweet[et][ft]

        return tweet[t] if t in tweet and len(tweet[t]) > 0 else None

    def extract_user(tweet):
        return tweet["user"]["id_str"] if "user" in tweet else None

    for gdrive_file in sorted(files, key=lambda file: file["title"]):
        title = gdrive_file["title"]
        skip = skip and start_date not in title

        if skip:
            continue

        drive.CreateFile({"id": gdrive_file["id"]}).GetContentFile(title)

        with ZipFile(title, "r") as zip_file:
            for filename in zip_file.namelist():
                end_date = filename[-14:-6]

                with zip_file.open(filename) as file:
                    for tweet in file:
                        tweet = loads(tweet.decode("utf-8"))
                        id_str = tweet["id_str"]
                        text = extract_text(tweet)
                        image = extract_image(tweet)
                        ht = extract_hashtags(tweet)
                        user = extract_user(tweet)
                        loc = extract_location(tweet)
                        invalid = text is None or image is None
                        invalid = invalid or ht is None or user is None
                        invalid = invalid or loc is None
                        invalid = invalid or len(ids) >= num_tweets

                        if invalid:
                            continue

                        text = convert_text(text)
                        text = [word for word in text if word >= 0]
                        lt = "_".join([filename[-14:], loc])
                        dataset[id_str] = (text, image, ht, user, lt)

                        ids.append(id_str)

                if len(ids) >= num_tweets:
                    break

        remove(title)

        if len(ids) >= num_tweets:
            break

    with open(data_filename, "w") as file:
        keys = ["id_str", "text", "image", "hashtags", "user", "lt"]

        for id_str in ids:
            tweet = dict(zip(keys, [id_str] + list(dataset[id_str])))

            file.write(dumps(tweet) + "\n")

        file.write(dumps({"end_date": end_date}))

    return dataset, ids, end_date


def precompute_images(filename, bucket, transform):
    cnn = DataParallel(models.__dict__["resnet152"](pretrained=True))

    if torch.cuda.is_available():
        cnn = cnn.cuda()

    cnn.module.fc = Sequential()
    queue = Queue()

    with open(filename, "r") as file:
        for line in file:
            queue.put(loads(line))

    image_dict, fn = {}, filename[:filename.rfind(".")] + "_img_feat.pt"

    for _ in range(10):
        ImagePreprocessor(queue, bucket, cnn, image_dict, transform).start()

    queue.join()
    torch.save(image_dict, fn)
