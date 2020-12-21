# -*- coding: utf-8 -*-

import codecs, sys
import collections, csv
import spacy, nltk
import numpy as np
from nltk.tokenize import TweetTokenizer
from scipy.stats import entropy

def preprocess(text):
    # extract tokens
    tweet_tok = TweetTokenizer()
    tokens = tweet_tok.tokenize(text)
    tokens_count = len(tokens)
    if text == "x" or text == 'text_speeches' or text == 'speakers' or tokens_count == 0:
        return 'error'

    # stemming - skipping for now

    # tokens count
    speech_length = len(tokens)
    word_lengths = list(map(lambda x: len(x), tokens))

    # mean word length, std word length
    mean_word_length = np.mean(word_lengths)
    std_word_length = np.std(word_lengths)

    # unique tokens
    unique_tokens = np.unique(tokens)
    unique_tokens_count = len(unique_tokens)
    lexical_diversity = unique_tokens_count / tokens_count

    pos_tagged_tokens = nltk.pos_tag(tokens)

    # find noun tokens, adjective tokens, verb tokens
    noun_tokens = list(filter(lambda x: x[1] in ('NN', 'NNS', 'NNP', 'NNPS'), pos_tagged_tokens))
    adjective_tokens = list(filter(lambda x: x[1] in ('JJ', 'JJR', 'JJS'), pos_tagged_tokens))
    verb_tokens = list(filter(lambda x: x[1] in ('VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'), pos_tagged_tokens))

    # count nouns, adjectives, verbs
    noun_count = len(noun_tokens)
    adjective_count = len(adjective_tokens)
    verb_count = len(verb_tokens)
    pos_tags_entropy = entropy([ noun_count, adjective_count, verb_count ])

    # normalise them
    noun_count = noun_count / speech_length
    adjective_count = adjective_count / speech_length
    verb_count = verb_count / speech_length

    # name entities count
    sp = spacy.load('en_core_web_sm')
    sen = sp(text)
    person_entities_count = len([ent for ent in sen.ents if ent.label_ == 'PERSON'])
    person_entities_count /= tokens_count

    feature_list = [
        ('mean_word_length', mean_word_length),
        ('std_word_length', std_word_length),
        ('lexical_diversity', lexical_diversity),
        ('noun_count', noun_count),
        ('adjective_count', adjective_count),
        ('verb_count', verb_count),
        ('pos_tags_entropy', pos_tags_entropy),
        ('person_entities_count', person_entities_count)
    ]

    return feature_list

sys.stdout = codecs.getwriter("cp1251")(sys.stdout, 'xmlcharrefreplace')
nltk.download('averaged_perceptron_tagger')
logFileName = "features_calculation.log"
speeches_file_name = "speeches.txt"
features_file_name = "stylometric_features.csv"
delimter = ','
featureVector = []  # all features collected throughout the whole document
lineCounter = 0

log = codecs.open(logFileName, 'w', encoding = "utf-8")
f = codecs.open(speeches_file_name, encoding = "utf-8")
feature_file = codecs.open(features_file_name, 'w', encoding = "utf-8")

speeches_file = open(speeches_file_name, encoding='latin-1')
csv_reader = csv.reader(speeches_file, delimiter=',')

feature_names = ['mean_word_length', 'std_word_length', 'lexical_diversity', 'noun_count', 'adjective_count', 'verb_count',
                 'pos_tag_entropy', 'person_entities_count', 'speaker']
feature_file.write(','.join(feature_names) + "\n")

for row in csv_reader:
    text_speech = row[0]
    speaker = row[1]
    feature_list = preprocess(text_speech)
    if (feature_list == 'error'):
        continue
    lineCounter += 1
    if lineCounter >= 990:
        log.write("ended\n")
    feature_row = ','.join(map(lambda x: str(x[1]), feature_list))
    feature_file.write(feature_row + "," + speaker + "\n")