import nltk
import pickle
import os

from nltk.classify import NaiveBayesClassifier
from nltk.classify import SklearnClassifier

from sklearn.svm import SVC

log_file = ''


def log(item):
    """
    Log the item to the log file and print it to the console.

    Parameters
    -----------
    item: str
        The item to log.
    """

    print(item)
    with open(log_file, "a") as f:
        f.write(item)
        f.write("\n")


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

    # duplicate title lines and convert everything to lowercase
    lines = [' '.join([lines[i]] * 2 + [lines[i + 1]]).lower()
             for i in range(0, len(lines), 2)]

    tokens = [nltk.word_tokenize(line) for line in lines]
    features = [nltk.FreqDist(line) for line in tokens]

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
    log('Accuracy: ' + str(accuracy))

    confusion_matrix = nltk.ConfusionMatrix(correct_labels, predicted_labels)
    log(confusion_matrix.pretty_format())

    log('\n\n%15s  | %10s  | %10s  | %10s \n-------------------------------------------------------------' % ('',
        'Precision', 'Recall', 'F1-Measure'))
    for label in classifier.labels():
        true_positives = sum([(a[0] == label) & (a[1] == label)
                             for a in zip(predicted_labels, correct_labels)])
        false_positives = sum([(a[0] == label) & (a[1] != label)
                              for a in zip(predicted_labels, correct_labels)])
        false_negatives = sum([(a[0] != label) & (a[1] == label)
                              for a in zip(predicted_labels, correct_labels)])

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1_measure = 2 * precision * recall / (precision + recall)

        log('%15s  | %10.4f  | %10.4f  | %10.4f' % (
            label, precision, recall, f1_measure))


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


def build_svc_classifier(training_data: list[(nltk.FreqDist, str)]) -> SklearnClassifier:
    """
    Build a Naive Bayes classifier from the training data.

    Parameters
    -----------
    training_data: list[(nltk.FreqDist, str)]
        The training data.

    Returns
    --------
    classifier: SklearnClassifier
        The classifier.
    """

    classifier = SklearnClassifier(SVC()).train(training_data)
    return classifier


if __name__ == "__main__":
    filename = 'cache/classifier_svc_lowercase.pickle'

    name = filename.split('.')[0].split('/')[-1]
    log_file = 'logs/' + name + '.log'
    # delete the log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)

    # try to load the classifier from the cache
    try:
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        log('Loaded classifier from cache.')
    except FileNotFoundError:
        # create the classifier
        training_data = create_training_data()
        classifier = build_svc_classifier(training_data)
        # save the classifier to the cache
        with open(filename, 'wb') as f:
            pickle.dump(classifier, f)
        log('Created classifier and saved to cache.')

    dev_data = create_dev_data()
    test_classifier(classifier, dev_data)
