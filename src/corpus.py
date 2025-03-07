import io
import numpy as np
import re

# This class contains all the corpus data from the files we provide to it 
class Language():
    def __init__(self):
        self.vocabulary = []
        self.corpus_size = 0
        self.delimiters = r"[,.:;\s]"

    # Read in a file and identify each unique word 
    def read(self, file):

        try:
            with io.open(file, 'r', encoding='utf-8') as file:
                text = file.read().lower()
                text = ''.join([c if c.isalnum() else ' ' for c in text])
                words = text.split()

                vocab = {}

                for word in words:
                    items = re.split(self.delimiters, word)

                    for item in items:
                        if item not in vocab:
                            vocab[word] = True
                
                self.vocabulary = list(vocab.keys())
                self.corpus_size = len(self.vocabulary)

        except Exception as e:
            raise Exception(f"Error in Language.read(): {e}")

        print(f"Generated a new vobulary of size {self.corpus_size}")

    # Get the index of a word in the vocab
    def get_word_index(self, x):
        index = 0
        try:
            index = self.vocabulary.index(x)
        except ValueError:
            raise Exception("item '{x}' not in vocabulary")

        return index

    # generate skip-gram dataset
    def build_dataset(self, file, window_size):
        try:
            with open(file, 'r') as file:
                text = file.read().lower()
                text = ''.join([c if c.isalnum() else ' ' for c in text])
                words = text.split()
        except Exception as e:
            raise Exception(f"Language.build_dataset error: {e}")

        word_indices = [self.get_word_index(word) for word in words]

        data = []
        for i in range(len(word_indices)):
            center = word_indices[i]
            contexts = []

            start = max(0, i - window_size)
            end = min(len(word_indices), i + window_size + 1)

            for j in range(start, end):
                if j != i:
                    contexts.append(word_indices[j])

            if contexts:
                data.append((center, contexts))

        return data



