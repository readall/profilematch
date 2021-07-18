
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords', download_dir='/workspace/app/data/nltk_data')
nltk.download('punkt', download_dir='/workspace/app/data/nltk_data')
nltk.data.path.append("/workspace/app/data/nltk_data")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer, util

import torch
import io

import spacy

#spacy.cli.download("en_core_web_trf", "/workspace/app/data/spacy/")
#spacy.load("en_core_web_sm")
#spacy.load("en_core_web_trf", "/workspace/app/data/spacy/")
spacy.load("en_core_web_trf")
#spacy.load( "/workspace/app/data/spacy//en_core_web_trf")
#spacy.load("en_core_web_trf") #, "/workspace/app/data/spacy/")


JD_FILE = "/workspace/app/code/jd.txt"
RESUME_FILE = "/workspace/app/code/resume.txt"
unwanted_chars = ['\n', '\n\n', '\n\n\n', '\t','\t\t', '\t\t\t']

def load_file(name=JD_FILE):
    content_list = []

    with io.open(name, 'rt') as f:
        content_list = f.readlines()
    return content_list


def extract_keywords(input_list):
    from sklearn.feature_extraction.text import CountVectorizer
    #print(len(input_list) , input_list[:5])
    text_all = []
    sentence_wo_sw = ""
    for line in input_list:
        line_ = line.split(' ')
        tokens_without_ = [word for word in line_ if not word in stopwords.words() ]
        tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
        # print(tokens_without_sw)
        sentence_wo_sw = ' '.join(tokens_without_sw)
        text_all.append(sentence_wo_sw)

    full_sentence = ' '.join(text_all)
    # print("*==="*50)
    # print(sentence_wo_sw)
    # print("*==="*50)
    n_gram_range = (1, 1)
    stop_words = "english"

    # Extract candidate words/phrases
    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([full_sentence])
    candidates = count.get_feature_names()
    #print(candidates)
    return candidates



def missing_keywords(key_jd, key_resume):
    missing_in_resume = [word for word in key_jd if not word in key_resume]
    missing_percent = 100*len(missing_in_resume)/len(key_jd)
    return missing_percent, missing_in_resume






def profile_matching_v1():
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2', cache_folder="/workspace/app/data/" )
    input_jd = load_file(JD_FILE)
    input_resume = load_file(RESUME_FILE)
    joined_list = []
    joined_list.append(input_jd)
    joined_list.append(input_resume)
    final_list = []
    
    for line in joined_list:
        # print(line)
        # print("#"*50)
        text_tokens = word_tokenize(line[0])
        # print("*"*50)
        # print(text_tokens)
        # print("*"*50)
        tokens_without_ = [word for word in text_tokens if not word in stopwords.words() ]
        tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
        sentence_wo_sw = ' '.join(tokens_without_sw)
        # print("#"*50)
        # print(sentence_wo_sw)
        # print("#"*50)
        final_list.append(sentence_wo_sw)
    
    paraphrases = util.paraphrase_mining(model,final_list )

    for paraphrase in paraphrases:
        score, i, j = paraphrase
        print("{} \t\t {} \t\t Score: {:.4f}".format(final_list[i][:20], final_list[j][:20], score))

    # #keyword extraction
    # print("#"*50)
    # jd_sentence_wo_sw = ""
    # for line in input_jd:
    #     jd_text_tokens = word_tokenize(line[0])
    #     jd_tokens_without_ = [word for word in text_tokens if not word in stopwords.words() ]
    #     jd_tokens_without_sw = [word for word in tokens_without_ if not word in unwanted_chars ]
    #     jd_sentence_wo_sw = ' '.join(tokens_without_sw)

    # model = SentenceTransformer('sentence-transformers/distilbert-base-nli-mean-tokens')
    # embeddings = model.encode(jd_sentence_wo_sw)
    # print(embeddings, embeddings.size)  
    # print("#"*50, "End Embeddings")
    print("#$"*30 )
    keyword_jd = extract_keywords(input_jd)
    print("#$"*30, type(keyword_jd))
    keyword_resume = extract_keywords(input_resume)
    print("#$"*30)
    print(missing_keywords(keyword_jd, keyword_resume))


profile_matching_v1()

#compare_documents()
