import torch
from data import get_test_loader
from evaluation import encode_data
from fasttext import load_model
from gensim.models import KeyedVectors as KV
from json import loads
from model import VSE
from pickle import dump

if __name__ == "__main__":
    model_path = "runs/runX/model_best.pth.tar"
    training_filename = "data/20-03-01_100000.data"
    test_filename = "data/20-03-10_5000.data"
    on_gpu = torch.cuda.is_available()
    device = "cpu" if not on_gpu else "cuda"
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    opt = checkpoint["opt"]
    word_model = load_model(opt.text_model_path)
    graph_model = KV.load_word2vec_format(opt.graph_model_path)
    model = VSE(opt, word_model, graph_model)
    components = ["image", "text", "hashtags", "user", "location", "time"]
    component_index = dict([(c, i) for i, c in enumerate(components)])
    components = ["image", "text", "hashtags"]

    for filename in ["data/20-03-01_100000.data", "data/20-03-10_5000.data"]:
        start_date = filename[len("data/"):len("data/") + len("yy-mm-dd")]
        num_tweets = int(filename[1 + filename.rfind("_"):filename.rfind(".")])
        data_loader = get_test_loader(start_date, num_tweets, word_model,
                                      graph_model, opt.precomp_images,
                                      opt.batch_size, opt.workers, opt)
        embeddings = encode_data(model, data_loader, components,
                                 component_index, on_gpu=on_gpu)
        embeddings = [(embeddings[0][i], embeddings[1][i])
                      for i in range(embeddings[0].shape[0])]
        ids, ht = [], []

        with open(filename, "r") as file:
            for line in file:
                tweet = loads(line)

                if "end_date" not in tweet:
                    ids.append(tweet["id_str"])
                    ht.append(tweet["hashtags"].split())

        embeddings = [(ids[i], embeddings[0][0], embeddings[0][1], ht[i])
                      for i in range(len(ht))]

        with open(filename.replace(".data", ".htr.pkl"), "wb") as file:
            dump(embeddings, file)
