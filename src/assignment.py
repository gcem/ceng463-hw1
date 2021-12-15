import nltk
import pickle

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier


def read_file(filename) -> list[str]:
    """
    Read the file and return the lines.

    Parameters
    -----------
    filename: str
        The name of the file to read.

    Returns
    --------
    lines: list[str]
        The lines of the file.
    """

    with open(filename, "r") as f:
        # read the lines
        lines = f.readlines()
    # trim whitespace
    lines = [line.strip() for line in lines]
    return lines


def preprocess(lines) -> list[nltk.FreqDist]:
    """
    Preprocess the lines and return the frequency distribution of words.

    Parameters
    -----------
    lines: list[str]
        The text to preprocess.

    Returns
    --------
    features: list[nltk.FreqDist]
        The frequency distribution of words.
    """

    # duplicate title lines
    lines = [' '.join([lines[i]] * 2 + [lines[i + 1]])
             for i in range(0, len(lines), 2)]

    tokens = [nltk.word_tokenize(line) for line in lines]
    features = [nltk.FreqDist(line) for line in tokens]

    # TODO: convert to lowercase?

    # count the number of occurrences of each word
    # word_frequency = nltk.FreqDist(tokens)

    return features


def create_training_data() -> list[(nltk.FreqDist, str)]:
    """
    Create training data from the training files.
        Return a list with word and category frequency distributions.

    Returns
    --------
    labeled_documents: list[(nltk.FreqDist, str)]
        Features in the document and the category.
    """

    training_documents = [
        "philosophy_train.txt", "sports_train.txt",
        "mystery_train.txt", "religion_train.txt",
        "science_train.txt", "romance_train.txt",
        "horror_train.txt", "science-fiction_train.txt"]

    labeled_documents = []

    for filename in training_documents:
        # read the file
        lines = read_file("data/train/" + filename)
        # get document count
        document_count = len(lines) / 2
        # get category name
        category = filename.split("_")[0]

        # preprocess the lines
        labeled_documents += [(document, category)
                              for document in preprocess(lines)]

    return labeled_documents


def create_dev_data() -> list[(nltk.FreqDist, str)]:
    """
    Create development data from the development files.
        Return a list with word and category frequency distributions.

    Returns
    --------
    labeled_documents: list[(nltk.FreqDist, str)]
        Features in the document and the category.
    """

    dev_documents = [
        "philosophy_dev.txt", "sports_dev.txt",
        "mystery_dev.txt", "religion_dev.txt",
        "science_dev.txt", "romance_dev.txt",
        "horror_dev.txt", "science-fiction_dev.txt"]

    labeled_documents = []

    for filename in dev_documents:
        # read the file
        lines = read_file("data/dev/" + filename)
        # get document count
        document_count = len(lines) / 2
        # get category name
        category = filename.split("_")[0]

        # preprocess the lines
        labeled_documents += [(document, category)
                              for document in preprocess(lines)]

    return labeled_documents


def test_classifier(classifier, test_data: list[(nltk.FreqDist, str)]):
    """
    Test the classifier on the test data.

    Parameters
    -----------
    classifier: NaiveBayesClassifier
        The classifier to test.
    test_data: list[(nltk.FreqDist, str)]
        The test data.
    """

    not_labeled = [item[0] for item in test_data]
    correct_labels = [item[1] for item in test_data]
    predicted_labels = classifier.classify_many(not_labeled)

    correct = [a[0] == a[1] for a in zip(predicted_labels, correct_labels)]
    accuracy = sum(correct) / len(correct)
    print('Accuracy:', accuracy)

    confusion_matrix = nltk.ConfusionMatrix(correct_labels, predicted_labels)
    print(confusion_matrix)


def build_bayes_classifier(training_data: list[(nltk.FreqDist, str)]) -> NaiveBayesClassifier:
    """
    Build a Naive Bayes classifier from the training data.

    Parameters
    -----------
    training_data: list[(nltk.FreqDist, str)]
        The training data.

    Returns
    --------
    classifier: NaiveBayesClassifier
        The classifier.
    """

    classifier = NaiveBayesClassifier.train(training_data)
    return classifier


if __name__ == "__main__":
    filename = 'cache/classifier_bayes_simple.pickle'

    # try to load the classifier from the cache
    try:
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        print('Loaded classifier from cache.')
    except FileNotFoundError:
        # create the classifier
        training_data = create_training_data()
        classifier = build_bayes_classifier(training_data)
        # save the classifier to the cache
        with open(filename, 'wb') as f:
            pickle.dump(classifier, f)
        print('Created classifier and saved to cache.')

    dev_data = create_dev_data()
    test_classifier(classifier, dev_data)
