import nltk

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
    """

    with open(filename, "r") as f:
        # read the lines
        lines = f.readlines()
        # trim whitespace
        lines = [line.strip() for line in lines]

        return lines


def preprocess(lines) -> nltk.FreqDist:
    """
    Preprocess the lines and return the frequency distribution of words.

    Parameters
    -----------
    lines: list[str]
        The lines to preprocess.

    Returns
    --------
    word_frequency: nltk.FreqDist
        The frequency distribution of words.
    """

    # remove punctuation
    lines = [line.translate(str.maketrans(
        '', '', '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')) for line in lines]

    # convert to lowercase
    lines = [line.lower() for line in lines]

    # split into words
    lines = " ".join(lines).split()

    # count the number of occurrences of each word
    word_frequency = nltk.FreqDist(lines)

    return word_frequency


def create_training_data() -> list[nltk.FreqDist]:
    """
    Create training data from the training files.
        Return a list with word and category frequency distributions.

    Returns
    --------
    training_data: list[nltk.FreqDist]
        Frequency of each category, frequencies of words in the whole training set, and the frequency of each word in each category respectively.
    """

    training_documents = [
        "philosophy_train.txt", "sports_train.txt",
        "mystery_train.txt", "religion_train.txt",
        "science_train.txt", "romance_train.txt",
        "horror_train.txt", "science-fiction_train.txt"]

    category_frequency = nltk.FreqDist()
    total_word_frequency = nltk.FreqDist()
    category_word_frequency = nltk.ConditionalFreqDist()

    for filename in training_documents:
        # read the file
        lines = read_file("data/train/" + filename)
        # get document count
        document_count = len(lines) / 2
        # get category name
        category = filename.split("_")[0]

        # preprocess the lines
        word_frequency = preprocess(lines)

        category_frequency[category] += document_count
        total_word_frequency += word_frequency
        category_word_frequency[category] += word_frequency

    return {
        "c": category_frequency,
        "w": total_word_frequency,
        "cw": category_word_frequency
    }


if __name__ == "__main__":
    training_result = create_training_data()
