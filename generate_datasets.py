from boto3 import resource
from botocore.exceptions import ClientError
from fastnode2vec import Graph, Node2Vec
from fasttext import load_model, train_unsupervised
from gensim.models import KeyedVectors
from json import dumps, loads
from networkx import DiGraph
from os import remove
from os.path import exists
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from re import sub
from requests import get
from zipfile import ZipFile


def extract_hashtags(tweet, word_model=None):
    hashtags = [h["text"].lower() for h in tweet["entities"]["hashtags"]]

    if word_model is not None:
        for hashtag in hashtags:
            if hashtag not in word_model:
                return None

    return hashtags if len(hashtags) > 0 else None


def extract_image(tweet, s3, bucket_name):
    url, ee, m = None, "extended_entities", "media"

    if ee in tweet and m in tweet[ee]:
        for media in tweet[ee][m]:
            if media["type"] == "photo":
                url = media["media_url"]
                break

    if url is not None:
        filename = tweet["id_str"] + url[url.rfind("."):]
        key = "/".join(["images", filename])

        if not key_exists(s3, bucket_name, key):
            try:
                response = get(url, timeout=60)
            except Exception:
                return None

            with open(filename, "wb") as file:
                file.write(response.content)

            s3.Object(bucket_name, key).upload_file(Filename=filename)
            remove(filename)

        try:
            if s3.Bucket(bucket_name).Object(key).content_length > 0:
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

    if t in tweet and len(tweet[t]) > 0:
        text = sub(r"[^A-Za-z0-9(),!?\'\`]", " ", str(tweet[t]))
        text = sub(r"\'ve", " \'ve", sub(r"\'s", " \'s", text))
        text = sub(r"\'re", " \'re", sub(r"n\'t", " n\'t", text))
        text = sub(r"\'ll", " \'ll", sub(r"\'d", " \'d", text))
        text = sub(r"\(", " ( ", sub(r"!", " ! ", sub(r",", " , ", text)))
        text = sub(r"\s{2,}", " ", sub(r"\?", " ? ", sub(r"\)", " ) ", text)))
        return text.strip().lower()

    return None


def extract_user(tweet, graph_model=None):
    if "user" in tweet:
        user = tweet["user"]["id_str"]

    if graph_model is None or (user is not None and user in graph_model):
        return user

    return None


def key_exists(s3, bucket_name, key):
    try:
        s3.Object(bucket_name, key).load()
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise

    return True


def make_dataset(start_date, num_tweets, train=False):
    s3 = resource("s3")
    bucket_name = "twitter-rriva002"
    drive = GoogleDrive(GoogleAuth().CommandLineAuth())
    query = "title contains 'geoTwitter' and trashed=false"
    files, skip = drive.ListFile({"q": query}).GetList(), True
    files = sorted(files, key=lambda file: file["title"])
    dataset, end_date = {}, None
    mdls_dir, data_dir = "models", "data"
    data_filename = "{}/{}_{}.data".format(data_dir, start_date, num_tweets)
    tweet_text_fn = "tweet_text.txt"
    word_model_fn = "/".join([mdls_dir, tweet_text_fn.replace(".txt", ".bin")])
    graph_model_fn = "/".join([mdls_dir, "graph_embedding.vec"])
    train = train and not exists(word_model_fn) and not exists(graph_model_fn)

    if train:
        tweet_text_file = open(tweet_text_fn, "w", encoding="utf-8")
        graph = DiGraph()
        word_model, graph_model = None, None
    else:
        word_model = load_model(word_model_fn)
        graph_model = KeyedVectors.load_word2vec_format(graph_model_fn)
        graph = None

    for gdrive_file in files:
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

                        # Skip if retweet

                        id_str = tweet["id_str"]
                        text = extract_text(tweet)
                        image = extract_image(tweet, s3, bucket_name)
                        ht = extract_hashtags(tweet, word_model)
                        user = extract_user(tweet, graph_model)
                        loc = extract_location(tweet)
                        data = [text, image, ht, user, loc]

                        if train:
                            if text is not None:
                                tweet_text_file.write(text + "\n")

                            if user is not None:
                                if user not in graph:
                                    graph.add_node(user)

                                for u in tweet["entities"]["user_mentions"]:
                                    if "id_str" in user:
                                        m_id = u["id_str"]

                                        if m_id not in graph:
                                            graph.add_node(m_id)

                                        if m_id in graph[user]:
                                            graph[user][m_id]["weight"] += 1
                                        else:
                                            graph.add_edge(user, m_id)

                                            graph[user][m_id]["weight"] = 1

                        if len(dataset) < num_tweets and None not in data:
                            data[-1] = "_".join([filename[-14:], loc])
                            dataset[id_str] = data

                if len(dataset) >= num_tweets:
                    break

        remove(title)

        if len(dataset) >= num_tweets:
            break

    if train:
        tweet_text_file.close()
        print("Training fastText model.")

        word_model = train_unsupervised(tweet_text_fn, dim=300, minCount=1,
                                        model="skipgram", thread=10, verbose=0)

        remove(tweet_text_fn)
        word_model.save_model(word_model_fn)

    for id_str, t in dataset.items():
        dataset[id_str][0] = [word_model.get_word_id(w) for w in t[0].split()]
        dataset[id_str][2] = [word_model.get_word_id(h) for h in t[2]]

    save_lt_files(dataset, start_date, end_date, drive, files, word_model, s3,
                  bucket_name)
    del word_model

    print("Saving dataset to " + data_filename)
    save_dataset(dataset, data_filename, end_date)
    del dataset

    if train:
        print("Training node2vec model.")

        graph = Graph(list(graph.edges.data("weight")), directed=True,
                      weighted=True)
        n2v = Node2Vec(graph, dim=300, walk_length=80, context=10, workers=10)

        n2v.train(epochs=10, progress_bar=False)
        n2v.wv.save_word2vec_format(graph_model_fn)


def save_dataset(dataset, filename, end_date):
    with open(filename, "w") as file:
        keys = ["id_str", "text", "image", "hashtags", "user", "lt"]

        for id_str, tweet in dataset.items():
            file.write(dumps(dict(zip(keys, [id_str] + tweet))) + "\n")

        file.write(dumps({"end_date": end_date}))


def save_lt_files(dataset, start_date, end_date, drive, files, word_model, s3,
                  bucket_name):
    lt_set, skip = set([tweet[-1] for tweet in dataset.values()]), True

    for gdrive_file in files:
        title = gdrive_file["title"]
        skip = skip and start_date not in title

        if skip:
            continue

        drive.CreateFile({"id": gdrive_file["id"]}).GetContentFile(title)

        with ZipFile(title, "r") as zip_file:
            for filename in zip_file.namelist():
                lt_tweets = {}

                with zip_file.open(filename) as file:
                    for tweet in file:
                        tweet = loads(tweet.decode("utf-8"))
                        loc = extract_location(tweet)
                        lt = "_".join([filename[-14:], loc])

                        if lt in lt_set:
                            text = extract_text(tweet)

                            if text is not None:
                                if lt in lt_tweets:
                                    lt_tweets[lt].append(text)
                                else:
                                    lt_tweets[lt] = [text]

                for lt, tweets in lt_tweets.items():
                    key = "/".join(["lt", lt])

                    if True:  # not key_exists(s3, bucket_name, key):
                        with open(lt, "w") as file:
                            for text in tweets:
                                txt = text.split()
                                txt = [word_model.get_word_id(w) for w in txt]

                                file.write(dumps(txt) + "\n")

                        s3.Object(bucket_name, key).upload_file(Filename=lt)
                        remove(lt)

        remove(title)

        if end_date in title:
            break


if __name__ == "__main__":
    # Fix training and validation start dates
    start_date_train = "20-03-01"
    start_date_validation = make_dataset(start_date_train, 100000, True)
    start_date_test = make_dataset(start_date_validation, 5000)
    end_date_test = make_dataset(start_date_test, 5000)
