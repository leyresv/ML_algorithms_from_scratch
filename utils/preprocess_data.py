import string
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def process_sentence(sentence):
    """
    Remove stopwords, punctuations and lemmatize

    :param sentence: Input sentence (String)
    :return: tokenized sentence (list)
    """
    stopwords_eng = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    clean_sentence = [lemmatizer.lemmatize(token) for token in sentence if
                      token not in stopwords_eng and token not in string.punctuation]

    return clean_sentence


def create_word_freqs_dict(sentences, labels, verbose=False):
    """
    Create frequencies dictionary

    :param sentences: list of sentences
    :param labels: list of sentences' labels (0 or 1)
    :param verbose: print beginning and end of the process
    :return: vocabulary frequencies dictionary
    """
    if verbose:
        print("Creating vocabulary frequencies dictionary...")
    tok_sentences = [process_sentence(sentence) for sentence in sentences]
    word_freqs = {}
    for sentence, label in zip(tok_sentences, labels):
        for word in sentence:
            if not (word, label[0]) in word_freqs:
                word_freqs[(word, label[0])] = 0
            word_freqs[(word, label[0])] += 1

    if verbose:
        print(f"FINISHED. Vocab size: {len(word_freqs)} tokens")
        print()
    # return dictionary sorted by values
    return dict(sorted(word_freqs.items(), key=lambda x:x[1], reverse=True))


def extract_freq_feature(sentences, vocab, verbose=False):
    """
    Create frequency features vector for each sequence

    :param sentences: input sentences (list of strings)
    :param vocab: vocabulary frequencies dictionary
    :param verbose: print beginning and end of the process
    :return: numpy array of frequency freatures vectors
    """
    if verbose:
        print("Extracting frequency features...")
    freq_feature = []
    for sentence in sentences:
        sentence = process_sentence(sentence)
        pos = 0
        neg = 0
        # Ignore repeated words
        for word in list(set(sentence)):
            pos += vocab.get((word, 1), 0)
            neg += vocab.get((word, 0), 0)
        # Add 1 for bias
        freq_feature.append([1, pos, neg])

    if verbose:
        print("FINISHED")
        print()
    return np.array(freq_feature)