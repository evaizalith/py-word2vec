import io
import numpy as np

# This class contains all the corpus data from the files we provide to it 
class Language():
    def __init__(self):
        self.vocabulary = []
        self.corpus_size = 0
    
    # Read in a file and identify each unique word 
    def read(self, file):
        try:
            with io.open(file, 'r', encoding='utf-8') as file:
                text = file.read()
                words = text.split()

                vocab = {}

                for word in words:
                    if word not in vocab:
                        vocab[word] = True
                
                self.vocabulary = list(vocab.keys())
                self.corpus_size = len(self.vocabulary)

        except Exception as e:
            raise Exception(f"Error in Language.read(): {e}")

        print(f"Generated a new vobulary of size {self.corpus_size}")

    # One-hot encode a given word based on the existing known vobulary
    # Returns an exception if the word provided to it is unknown
    def hot_encode(self, x):
        try:
            index = self.vocabulary.index(x)
        except ValueError:
            raise Exception(f"Item '{x}' is not in the vocabulary")

        hot = np.zeros(shape=(self.corpus_size, 1))
        hot[index] = 1

        return hot

    # Pass the softmaxed results of the NN to randomly select a word from the vocabulary treating the value of each element in the output as a probability
    def get_likely_word(self, x):
        x = x.flatten()
        i = np.random.choice(len(x), p=x)
        return self.vocabulary[i]

