# Tweet-Embeddings

**Files for tweet embedding model:**
* data.py - Loads data from files and prepares it for input to the model.
* data_helpers.py - Additional functionality for loading and preprocessing.
* evaluation.py - Functionality for evaluating model performance (can be run as a standalone program).
* model.py - Code for tweet embedding model framework.
* train.py - Trains a tweet embedding model.

**Files for hashtag recommendation:**
* feature_extraction.py - Code for generating text and image embeddings from a trained tweet embedding model for input to hashtag recommendation baseline (see below).
* hashtag_recommendation.py - Tests retrieval-based hashtag recommendation using a trained tweet embedding model.

**Files for hashtag recommendation baseline:**
* coa/co_attention.py - Trains and tests hashtag recommendation baseline.
* coa/data.py - Loads data from files and prepares it for input to the model.
* coa/selfDef.py - Code for hashtag recommendation baseline co-attention network and loss function.

**To train tweet embedding model:**
train.py [options]

**To test retrieval-based hashtag recommendation:**
train.py [options]  
hashtag_recommendation.py

**To test hashtag recommendation baseline:**
coa/co_attention.py

**To test hashtag recommendation baseline with image and text vectors from tweet embedding model:**
train.py [options]  
feature_extraction.py  
coa/co_attention.py --precomp_features

**Latest results:**
Tweet embedding model trained as follows:  
train.py --precomp_images --max_lt_tweets 50 --components image,text,hashtags,user,location --k_vals 1,5,10 --max_similarity --target hashtags  
  
Retrieval-based hashtag recommendation:
| k   | Accuracy | Precision | Recall | F1   |
| --- | -------- | --------- | ------ | ---- |
| 1   | 9.68     | 9.68      | 7.06   | 7.72 |
| 5   | 15.1     | 3.15      | 10.98  | 4.63 |
| 10  | 17.32    | 1.85      | 12.54  | 3.1  |

Baseline with tweet embedding vectors:
| k   | Accuracy | Precision | Recall | F1   |
| --- | -------- | --------- | ------ | ---- |
| 1   | 0.44     | 0.44      | 0.13   | 0.2  |
| 5   | 8.8      | 1.76      | 6.83   | 2.67 |
| 10  | 9.03     | 0.93      | 7.04   | 1.58 |

Baseline:
| k   | Accuracy | Precision | Recall | F1   |
| --- | -------- | --------- | ------ | ---- |
| 1   | 10.28    | 10.28     | 7.88   | 8.52 |
| 5   | 14.38    | 3.68      | 11.84  | 5.28 |
| 10  | 16.36    | 2.11      | 13.44  | 3.48 |
