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

Retrieval-based hashtag recommendation:  
train.py --precomp_images --max_lt_tweets 50 --components image,text,hashtags,user,location --k_vals 1,5,10 --max_violation --max_similarity --target average  
hashtag_recommendation.py
| k   | Accuracy | Precision | Recall | F1    |
| --- | -------- | --------- | ------ | ----- |
| 1   | 14.14    | 14.14     | 10.15  | 11.15 |
| 5   | 20.1     | 4.58      | 15.04  | 6.61  |
| 10  | 21.74    | 2.56      | 16.32  | 4.23  |
  
Retrieval-based hashtag recommendation:  
train.py --precomp_images --max_lt_tweets 50 --components image,text,hashtags,user,location --k_vals 1,5,10 --max_violation --max_similarity --target hashtags  
hashtag_recommendation.py
| k   | Accuracy | Precision | Recall | F1    |
| --- | -------- | --------- | ------ | ----- |
| 1   | 21.78    | 21.78     | 15.32  | 16.97 |
| 5   | 27.02    | 6.22      | 20.08  | 8.94  |
| 10  | 29.75    | 3.7       | 22.68  | 6.09  |

Baseline with tweet embedding vectors (target hashtags):  
train.py --precomp_images --max_lt_tweets 50 --components image,text,hashtags,user,location --k_vals 1,5,10 --max_violation --max_similarity --target hashtags  
feature_extraction.py  
coa/co_attention.py --precomp_features
| k   | Accuracy | Precision | Recall | F1   |
| --- | -------- | --------- | ------ | ---- |
| 1   | 0.44     | 0.44      | 0.13   | 0.2  |
| 5   | 8.86     | 1.77      | 6.86   | 2.68 |
| 10  | 9.68     | 1.03      | 7.67   | 1.76 |

Baseline:  
coa/co_attention.py
| k   | Accuracy | Precision | Recall | F1   |
| --- | -------- | --------- | ------ | ---- |
| 1   | 10.28    | 10.28     | 7.88   | 8.52 |
| 5   | 14.38    | 3.68      | 11.84  | 5.28 |
| 10  | 16.36    | 2.11      | 13.44  | 3.48 |
