import numpy as np
from utils.preprocess_data import create_word_freqs_dict, process_sentence

class NaiveBayesClassifier:

    def __init__(self):
        self.freqs_dict = {}
        self.log_prior = 0
        self.log_likelihood = {}

    def __get_conditional_probability(self, word, label, freqs_dict, n_class, vocab_size):
        return (freqs_dict.get((word, label), 0) + 1) / (n_class + vocab_size)

    def __get_log_prior(self, labels):
        p_pos = sum(labels)
        p_neg = len(labels) - p_pos
        return np.log(p_pos) - np.log(p_neg)

    def train(self, x_train, y_train, verbose=False):
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
        self.log_prior = self.__get_log_prior(y_train)

        # Get log likelihood of each word
        for word in vocab:
            prob_pos = self.__get_conditional_probability(word, 1, self.freqs_dict, num_pos, vocab_size)
            prob_neg = self.__get_conditional_probability(word, 0, self.freqs_dict, num_neg, vocab_size)
            self.log_likelihood[word] = np.log(prob_pos) - np.log(prob_neg)

        if verbose:
            print(f"Training finished")
        return self

    def predict(self, sentences):
        sentences = [process_sentence(sentence) for sentence in sentences]
        predictions = []

        for sentence in sentences:
            pred = self.log_prior + np.sum(np.array([self.log_likelihood.get(word, 0) for word in sentence]))
            if pred > 0:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions
