import nltk
import numpy as np

from preprocess_data import create_word_freqs_dict, extract_freq_feature
from logistic_regression import LogisticRegressionClassifier
from nltk.corpus import sentence_polarity

nltk.download("sentence_polarity")
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download("stopwords")
nltk.download("punkt")


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

    # Crete frequencies dictionary
    vocab_dict = create_word_freqs_dict(train_sentences, Y_train, verbose=True)

    # Extract input features from sequences
    X_train = extract_freq_feature(train_sentences, vocab_dict, verbose=True)
    X_test = extract_freq_feature(test_sentences, vocab_dict, verbose=True)

    # Instantiate the classifier
    lr_classifier = LogisticRegressionClassifier()

    # Hyper-parameters
    alpha = 5e-6
    num_iter = 500

    # Train the classifier
    lr_classifier.train(X_train, Y_train, alpha, num_iter, verbose=True)

    # Predict labels for test set and evaluate accuracy
    Y_pred = lr_classifier.predict(X_test)
    print("Accuracy on test set:", lr_classifier.evaluate_accuracy(Y_test, Y_pred))

    # Print some test sentences and their predicted and true label
    print()
    for sentence, label, pred_label in zip(test_sentences[:5], Y_test[:5], Y_pred[:5]):
        print(sentence)
        print(f"Predicted label: {pred_label} ------- True label: {label[0]}")
        print()


if __name__=="__main__":
    main()