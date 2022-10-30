import numpy as np
from utils.preprocess_data import create_word_freqs_dict, process_sentence


class NaiveBayesClassifier:

    def __init__(self):
        self.freqs_dict = {}
        self.log_prior = 0
        self.log_likelihood = {}

    @staticmethod
    def get_conditional_probability(word, label, freqs_dict, n_class, vocab_size):
        """
        Compute the probability of a word given a class: P(word|class)

        :param word: string
        :param label: class label (0 or 1)
        :param freqs_dict: frequencies dictionary
        :param n_class: int class frequency
        :param vocab_size: int number of words in the vocabulary
        :return: conditional probability value (float)
        """
        return (freqs_dict.get((word, label), 0) + 1) / (n_class + vocab_size)

    @staticmethod
    def get_log_prior(labels):
        """
        Calculate te log prior

        :param labels: list of training labels
        :return: log prior value (float)
        """
        p_pos = sum(labels)
        p_neg = len(labels) - p_pos
        return np.log(p_pos) - np.log(p_neg)

    def train(self, x_train, y_train, verbose=False):
        """
        Train the classifier

        :param x_train: list of training tokenized sentences (list of lists of strings)
        :param y_train: training labels
        :param verbose: print beginning and end of the process
        :return: reference to the instance object
        """
        if verbose:
            print("Training Naïve Bayes classifier")

        self.freqs_dict = create_word_freqs_dict(x_train, y_train)

        # Get classes frequency
        num_pos = np.sum(np.array([self.freqs_dict[(word, label)] for (word, label) in self.freqs_dict.keys() if label == 1]))
        num_neg = np.sum(np.array([self.freqs_dict[(word, label)] for (word, label) in self.freqs_dict.keys() if label == 0]))

        # Get vocab size
        vocab = set([word for (word, _) in self.freqs_dict.keys()])
        vocab_size = len(vocab)

        # Get dataset log prior
        self.log_prior = self.get_log_prior(y_train)

        # Get log likelihood of each word
        for word in vocab:
            prob_pos = self.get_conditional_probability(word, 1, self.freqs_dict, num_pos, vocab_size)
            prob_neg = self.get_conditional_probability(word, 0, self.freqs_dict, num_neg, vocab_size)
            self.log_likelihood[word] = np.log(prob_pos) - np.log(prob_neg)

        if verbose:
            print("Training finished")
        return self

    def predict(self, sentences):
        """
        Predict polarity labels for the input sequences (0=Negative, 1=Positive)

        :param sentences: list of tokenized sentences
        :return: list of predicted polarity labels for the input sentences
        """
        sentences = [process_sentence(sentence) for sentence in sentences]
        predictions = []

        for sentence in sentences:
            pred = self.log_prior + np.sum(np.array([self.log_likelihood.get(word, 0) for word in sentence]))
            if pred > 0:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def get_ratio(self, word):
        """
        Calculate ratio of positive/negative frequency of a word in the training set

        :param word: string
        :return: ratio (float)
        """
        return (self.freqs_dict.get((word, 1), 0) + 1) / (self.freqs_dict.get((word, 0), 0) + 1)

    def get_words_by_threshold(self, label, threshold):
        """
        Get vocabulary words that have a minimum level of positiveness/negativeness

        :param label: 1 for positive, 0 for negative
        :param threshold: that will be used as the cutoff for including a word in the returned dictionary
        :return: dictionary of filtered words (key) and their ratio (value)
        """
        filtered_words = {}
        for (word, _) in self.freqs_dict.keys():
            ratio = self.get_ratio(word)
            if label == 1 and ratio >= threshold:
                filtered_words[word] = ratio
            elif label == 0 and ratio <= threshold:
                filtered_words[word] = ratio
        return filtered_words


