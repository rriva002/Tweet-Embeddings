import json as jsonmod
import numpy as np
from boto3 import resource
from os import remove
from os.path import exists
from pickle import load, dump
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def batchMaker(images, texts, tags, ids, batch_size, num_tags):
    shape = texts.shape[0]
    text_copy = texts.copy()
    tag_copy = tags.copy()
    ids_copy = ids.copy()
    indices = np.arange(shape)

    np.random.shuffle(indices)

    text_copy = list(text_copy[indices])
    tag_copy = list(tag_copy[indices])
    i = 0

    while True:
        if i + batch_size <= shape:
            img_train = []
            tmp_train_text = []
            tmp_train_tag = []

            for index in range(i, i + batch_size):
                data = images[ids_copy[index]]
                np_data = np.array(data)

                if np_data.shape != ():
                    img_train.append(np_data)
                    tmp_train_text.append(np.array(text_copy[index],
                                                   dtype=np.int32))
                    tmpArray = np.zeros(num_tags)
                    tmpArray[np.array(tag_copy[index], dtype=np.int32)] = 1
                    tmp_train_tag.append(tmpArray)

            text_train = np.array(tmp_train_text)
            tag_train = np.array(tmp_train_tag)
            img_train = np.squeeze(np.array(img_train))

            yield [img_train, text_train], tag_train
            i += batch_size
        else:
            i = 0
            indices = np.arange(shape)
            np.random.shuffle(indices)
            text_copy = np.array(text_copy)
            tag_copy = np.array(tag_copy)
            text_copy = list(text_copy[indices])
            tag_copy = list(tag_copy[indices])
            continue


def load_data_file(data_filename, word_model, precomp_features=False):
    hashtag_ids = set()
    dataset = {}

    if precomp_features:
        images = {}

        with open(data_filename.replace(".data", ".htr.pkl"), "rb") as file:
            data = load(file)

        for tweet in data:
            hashtags = tweet[3]

            if len(hashtags) > 0:
                dataset[tweet[0]] = (tweet[2], hashtags)
                images[tweet[0]] = tweet[1]

                for hashtag in hashtags:
                    hashtag_ids.add(hashtag)

        return images, dataset, hashtag_ids

    bucket = resource("s3").Bucket("twitter-rriva002")
    index_from = 1000

    with open(data_filename, "r") as file:
        for tweet in file:
            tweet = jsonmod.loads(tweet)

            if "end_date" not in tweet:
                hashtags = tweet["hashtags"].split()
                # hashtags = [word_model.get_word_id(h) for h in
                #             hashtags if h in word_model]

                if len(hashtags) > 0:
                    id_str = tweet["id_str"]
                    hashtag_id_set = set([word_model.get_word_id(h) for h in
                                          hashtags if h in word_model])
                    # hashtag_id_set = set(hashtags)
                    text = [word for word in tweet["text"] if word >= 0]
                    text = [t + 1 for t in text if t not in hashtag_id_set]
                    index_from = min([index_from] + text)
                    dataset[id_str] = (text, hashtags, tweet["image"])

                    for hashtag in hashtags:
                        hashtag_ids.add(hashtag)

    image_filename = data_filename.replace(".data", ".vgg.pkl")

    if exists(image_filename):
        with open(image_filename, "rb") as file:
            images = load(file)
    else:
        vgg = VGG16(weights="imagenet", include_top=False)
        images = {}

        for id_str, data in dataset.items():
            filename = data[-1]
            key = "images/" + filename

            if bucket.Object(key).content_length == 0:
                continue

            bucket.download_file(key, filename)

            img = load_img(filename, target_size=(224, 224))

            if exists(filename):
                remove(filename)

            x = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
            images[id_str] = vgg.predict(x)

        with open(image_filename, "wb") as file:
            dump(images, file)

    for id_str, image in images.items():
        images[id_str] = image.squeeze(0)

    seq_length = max([len(tweet[0]) for tweet in dataset.values()])
    return images, dataset, index_from, seq_length, hashtag_ids


def load_all_data(train_fn, test_fn, word_model, batch_size,
                  precomp_features=False):
    if precomp_features:
        img_tr, data_tr, ht_ids = load_data_file(train_fn, word_model, True)
        img_test, data_test, ht_id_test = load_data_file(test_fn, word_model,
                                                         True)
    else:
        img_tr, data_tr, if_tr, s_ln_tr, ht_ids = load_data_file(train_fn,
                                                                 word_model)
        img_test, data_test, ift, slt, ht_id_test = load_data_file(test_fn,
                                                                   word_model)
        index_from = min(if_tr, ift)
        seq_length = max(s_ln_tr, slt)

    hashtag_ids = sorted(list(ht_ids.union(ht_id_test)))
    num_tags = len(hashtag_ids)
    hashtag_index = {}
    text_data, id_data, hashtag_data = [], [], []

    for i, hashtag_id in enumerate(hashtag_ids):
        hashtag_index[hashtag_id] = i

    for id_str in data_tr.keys():
        text = data_tr[id_str][0]

        if not precomp_features:
            text += [0 for _ in range(seq_length - len(text))]

        text_data.append(text)
        id_data.append(id_str)

    for id_str in data_tr.keys():
        hashtag_data.append([hashtag_index[h] for h in data_tr[id_str][1]
                             if h in hashtag_index])

    generator = batchMaker(img_tr, np.array(text_data), np.array(hashtag_data),
                           id_data, batch_size, num_tags)
    len_data = len(text_data)

    image_test, text_data, hashtag_data = [], [], []

    for id_str in data_test.keys():
        text = data_test[id_str][0]

        if not precomp_features:
            text += [0 for _ in range(seq_length - len(text))]

        image_test.append(img_test[id_str])
        text_data.append(text)

        tmpArray = np.zeros(num_tags)
        hashtags = [hashtag_index[h] for h in data_test[id_str][1] if h in
                    hashtag_index]

        tmpArray[np.array(hashtags, dtype=np.int32)] = 1
        hashtag_data.append(tmpArray)

    if precomp_features:
        return generator, len_data, np.array(image_test), \
            np.array(text_data), np.array(hashtag_data), num_tags

    return generator, len_data, index_from, np.array(image_test), \
        np.array(text_data), np.array(hashtag_data), seq_length, num_tags
