import numpy as np
import torch
from data import get_test_loader
from evaluation import encode_data
from fasttext import load_model
from gensim.models import KeyedVectors as KV
from json import loads
from model import VSE


def evaluation(y_true, y_pred, top_K):
    acc_count = 0
    precision_K = []
    recall_K = []
    f1_K = []

    for i in range(y_pred.shape[0]):
        top_indices = y_pred[i].argsort()[-top_K:]
        if np.sum(y_true[i, top_indices]) >= 1:
            acc_count += 1
        p = np.sum(y_true[i, top_indices]) / top_K
        r = np.sum(y_true[i, top_indices]) / np.sum(y_true[i, :])
        precision_K.append(p)
        recall_K.append(r)
        if p != 0 or r != 0:
            f1_K.append(2 * p * r / (p + r))
        else:
            f1_K.append(0)

    acc_K = acc_count * 1.0 / y_pred.shape[0]

    return acc_K, np.mean(np.array(precision_K)), \
        np.mean(np.array(recall_K)), np.mean(np.array(f1_K))


def all_hashtags(model, word_model, training_filename, test_filename,
                 on_gpu=False):
    hashtags, hashtag_embeddings, num_tweets = {}, [], 0

    for filename in [training_filename, test_filename]:
        with open(filename, "r") as file:
            with torch.no_grad():
                for line in file:
                    tweet = loads(line)

                    if "end_date" not in tweet:
                        num_tweets += 1 if filename == test_filename else 0

                        for hashtag in tweet["hashtags"].split():
                            if hashtag not in hashtags:
                                hashtags[hashtag] = len(hashtags)

                                if filename == training_filename:
                                    w_emb = torch.tensor([word_model[hashtag]])
                                    w_emb = w_emb.cuda() if on_gpu else w_emb
                                    ht_emb = model.ht_enc([w_emb])

                                    if on_gpu:
                                        ht_emb = ht_emb.clone().detach().cpu()

                                    hashtag_embeddings.append(ht_emb)

    hashtag_embeddings = torch.stack(hashtag_embeddings, dim=0).squeeze(1)
    y_test = np.zeros((num_tweets, len(hashtags)), dtype=np.int8)

    with open(test_filename, "r") as file:
        index = 0

        for line in file:
            tweet = loads(line)

            if "end_date" not in tweet:
                for hashtag in tweet["hashtags"].split():
                    y_test[index, hashtags[hashtag]] = 1

                index += 1

    if on_gpu:
        hashtag_embeddings = hashtag_embeddings.data.cpu().numpy().copy()
    else:
        hashtag_embeddings = hashtag_embeddings.data.numpy().copy()

    return hashtag_embeddings, y_test


def format_result(result, decimal_places=2):
    return int(result * 100 ** max(decimal_places, 0)) / 100


if __name__ == "__main__":
    model_path = "runs/runX/model_best.pth.tar"
    on_gpu = torch.cuda.is_available()
    device = "cpu" if not on_gpu else "cuda"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    opt = checkpoint["opt"]
    word_model = load_model(opt.text_model_path)
    graph_model = KV.load_word2vec_format(opt.graph_model_path)
    model = VSE(opt, word_model, graph_model)
    data_loader = get_test_loader("20-03-10", 5000, word_model, graph_model,
                                  opt.precomp_images, opt.batch_size,
                                  opt.workers, opt)

    model.load_state_dict(checkpoint["model"])

    components = ["image", "text", "hashtags", "user", "location", "time"]
    component_index = dict([(c, i) for i, c in enumerate(components)])
    components = [c for c in opt.components.split(",") if c != "hashtags"]
    embeddings = encode_data(model, data_loader, components, component_index,
                             on_gpu=on_gpu)
    hashtag_embeddings, y_test = all_hashtags(model, word_model,
                                              "data/20-03-01_100000.data",
                                              "data/20-03-10_5000.data",
                                              on_gpu=on_gpu)
    y_pred = []

    for i in range(len(data_loader.dataset)):
        scores = []

        for embedding_array in embeddings:
            scores.append(np.dot(embedding_array[i], hashtag_embeddings.T))

        if opt.max_similarity:
            y_pred.append(np.max(scores, axis=0))
        else:
            y_pred.append(np.sum(scores, axis=0))

    y_pred = np.array(y_pred)

    print("k\tAccuracy\tPrecision\tRecall\t\tF1")

    for k in opt.k_vals:
        acc_K, precision_K, recall_K, f1_K = evaluation(y_test, y_pred, k)
        print("{}\t{}\t\t{}\t\t{}\t\t{}".format(k, format_result(acc_K),
                                                format_result(precision_K),
                                                format_result(recall_K),
                                                format_result(f1_K)))
