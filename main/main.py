import nltk
import numpy as np

from utils.preprocess_data import create_word_freqs_dict, extract_freq_feature
from utils.evaluation_metrics import evaluate_accuracy
from models.logistic_regression import LogisticRegressionClassifier
from models.naive_bayes import NaiveBayesClassifier
from nltk.corpus import sentence_polarity

nltk.download("sentence_polarity")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")
nltk.download("punkt")


def print_results(test_sentences, y_test, y_pred):
    print("Accuracy on test set:", evaluate_accuracy(y_test, y_pred))
    print()
    # Print some test sentences and their predicted and true label
    for sentence, label, pred_label in zip(test_sentences[:5], y_test[:5], y_pred[:5]):
        print(sentence)
        print(f"Predicted label: {pred_label} ------- True label: {label[0]}")
        print()


def lr_classification(train_sentences, test_sentences, Y_train, Y_test, verbose=False):
    # Crete frequencies dictionary
    vocab_dict = create_word_freqs_dict(train_sentences, Y_train, verbose=verbose)

    # Extract input features from sequences
    X_train = extract_freq_feature(train_sentences, vocab_dict, verbose=verbose)
    X_test = extract_freq_feature(test_sentences, vocab_dict, verbose=verbose)

    # Instantiate the classifier
    lr_classifier = LogisticRegressionClassifier()

    # Hyper-parameters
    alpha = 5e-6
    num_iter = 500

    # Train the classifier
    lr_classifier.train(X_train, Y_train, alpha, num_iter, verbose=verbose)

    # Predict labels for test set and evaluate accuracy
    Y_pred = lr_classifier.predict(X_test)
    if verbose:
        print_results(test_sentences, Y_test, Y_pred)


def nb_classification(train_sentences, test_sentences, Y_train, Y_test, verbose=False):
    # Train the classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(train_sentences, Y_train, verbose=verbose)

    # Predict labels for test set and evaluate accuracy
    y_pred = nb_classifier.predict(test_sentences)

    if verbose:
        print_results(test_sentences, Y_test, y_pred)

    # Get word positiveness/negativeness ratio
    print("Ratio of word 'movie':", nb_classifier.get_ratio("movie"))

    # Get words with a positive ratio higher than 10:
    print(nb_classifier.get_words_by_threshold(1, 10))

    # Get words with a negative ratio lower than 0.5:
    print(nb_classifier.get_words_by_threshold(0, 0.1))


def main():
    # Import sentence polarity corpus and split data in train/test
    pos_ids = sentence_polarity.fileids('pos')
    neg_ids = sentence_polarity.fileids('neg')
    pos_sentences = sentence_polarity.sents(pos_ids)
    neg_sentences = sentence_polarity.sents(neg_ids)
    train_pos, train_neg = pos_sentences[:4500], neg_sentences[:4500]
    test_pos, test_neg = pos_sentences[4500:], neg_sentences[4500:]
    train_sentences = train_pos + train_neg
    test_sentences = test_pos + test_neg

    # Create sentiment labels
    Y_train = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
    Y_test = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

    # Logistic Regression
    lr_classification(train_sentences, test_sentences, Y_train, Y_test, verbose=True)

    # Naïve Bayes
    nb_classification(train_sentences, test_sentences, Y_train, Y_test, verbose=True)


if __name__=="__main__":
    main()