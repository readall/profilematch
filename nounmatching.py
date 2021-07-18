
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
from sklearn.feature_extraction.text import CountVectorizer
import io

import spacy

#spacy.cli.download("en_core_web_trf", "/workspace/app/data/spacy/")
#spacy.load("en_core_web_sm")
#spacy.load("en_core_web_trf", "/workspace/app/data/spacy/")
nlp = spacy.load("en_core_web_trf")
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



def compare_documents():
    # key_jd = extract_keywords(JD_FILE)
    # key_resume = extract_keywords(RESUME_FILE)
    #nlp = spacy.load("en_core_web_trf")
    input_jd = ' '.join(load_file(JD_FILE))

    doc_jd = nlp(input_jd)
    noun_phrases_jd = set(chunk.text.strip().lower() for chunk in doc_jd.noun_chunks)
    nouns_jd = set()
    for token in doc_jd:
        if token.pos_ == "NOUN":
            nouns_jd.add(token.text)
    print("#nouns_jd"*10, '\n', nouns_jd, '\n', "#"*100)

    input_resume = ' '.join(load_file(RESUME_FILE))
    doc_resume = nlp(input_resume)
    noun_phrases_resume = set(chunk.text.strip().lower()
                              for chunk in doc_resume.noun_chunks)
    nouns_resume = set()
    for token in doc_resume:
        if token.pos_ == "NOUN":
            nouns_resume.add(token.text)
    print("#nouns_resume"*10, '\n', nouns_resume, '\n', "#"*100)
    print(missing_keywords(nouns_jd, nouns_resume))
    #print(compare_nouns(nouns_jd, nouns_resume))




compare_documents()

