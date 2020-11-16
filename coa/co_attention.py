'''
hierarchical Co-attention model based on IJCAI article
'''
from tensorflow import expand_dims
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Reshape

from selfDef import coAttention_alt, myLossFunc
from data import load_all_data
from fasttext import load_model
import argparse
import numpy as np

# num_tags = 3896
# num_words = 212000
# index_from = 3
# seq_length = 30
batch_size = 128
cnn_dimension = 512
embedding_size = 200
hidden_size = 100
attention_size = 200
dim_k = 100
num_region = 7*7
drop_rate = 0.5
TopK = [1, 5, 10, 100]


def imageFeature(inputs):
    features = Reshape(target_shape=(num_region, cnn_dimension))(inputs)
    features = Dense(embedding_size, activation="tanh",
                     use_bias=False)(features)
    return features


def textFeature(X):
    embeddings = Embedding(input_dim=num_words + index_from,
                           output_dim=embedding_size, mask_zero=True,
                           input_length=seq_ln)(X)
    tFeature = LSTM(units=embedding_size, return_sequences=True)(embeddings)

    return tFeature


def modelDef(precomputed_features=False):
    if precomputed_features:
        embedding_size = 1024
        inputs_img = Input(shape=(embedding_size,))
        inputs_text = Input(shape=(embedding_size,))
        iFeature = expand_dims(inputs_img, axis=1)
        tFeature = expand_dims(inputs_text, axis=1)
    else:
        embedding_size = 200
        inputs_img = Input(shape=(7, 7, 512))
        inputs_text = Input(shape=(seq_ln,))
        iFeature = imageFeature(inputs_img)
        tFeature = textFeature(inputs_text)

    co_feature = coAttention_alt(dim_k=dim_k, output_dim=embedding_size,
                                 precomp_features=precomputed_features)

    co_feature.build([iFeature.get_shape().as_list(),
                      tFeature.get_shape().as_list()])

    co_feature = co_feature([iFeature, tFeature])
    dropout = Dropout(drop_rate)(co_feature)
    Softmax = Dense(n_tags, activation="softmax", use_bias=True)(dropout)
    model = Model(inputs=[inputs_img, inputs_text], outputs=[Softmax])
    model.compile(optimizer="adam", loss=myLossFunc)
    return model


def format_result(result, decimal_places=2):
    return int(result * 100 ** max(decimal_places, 0)) / 100


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    train_fn, test_fn = "data/20-03-01_100000.data", "data/20-03-10_5000.data"

    parser.add_argument('--precomp_features', action='store_true',
                        help='Use precomputed feature embeddings.')
    opt = parser.parse_args()
    word_model = load_model("./models/tweet_text.bin")

    if opt.precomp_features:
        gen, len_data, img_test, txt_test, tag_test, n_tags = \
            load_all_data(train_fn, test_fn, word_model, batch_size, True)
    else:
        # prepare the following data. img data is the output of VGG-16
        num_words = len(word_model.get_words())
        gen, len_data, index_from, img_test, txt_test, tag_test, seq_ln, \
            n_tags = load_all_data(train_fn, test_fn, word_model, batch_size)

    del word_model

    model = modelDef(opt.precomp_features)
    history = model.fit(x=gen, epochs=30, verbose=1,
                        steps_per_epoch=int(len_data / batch_size))
    y_pred = model.predict(x=[img_test, txt_test])

    print("k\tAccuracy\tPrecision\tRecall\t\tF1")

    for k in TopK:
        acc_K, precision_K, recall_K, f1_K = evaluation(tag_test, y_pred, k)
        print("{}\t{}\t\t{}\t\t{}\t\t{}".format(k, format_result(acc_K),
                                                format_result(precision_K),
                                                format_result(recall_K),
                                                format_result(f1_K)))
