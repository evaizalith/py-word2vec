from nn import net
import numpy as np
from corpus import Language
import sys
import io
import traceback

def train(file, lang, nn, lr=0.01):
    try:
        with io.open(file, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()

    except Exception as e:
        print(f"Error in train: {e}")
        traceback.print_exec()

    try:
        prev = ""

        losses = []

        for word in words:
            if prev == "":
                prev = word

                x = lang.hot_encode(prev)
                y_t = lang.hot_encode(word)

                y_p = nn.forward(x)
                loss = nn.cross_entropy(y_p, y_t)

                nn.gradient(x, y_p, y_t, loss, lr)
    
                losses.append(loss)

                pred = lang.get_likely_word(y_p)
                print(f"pred: {pred}, true: {word}, loss: {loss}")

                prev = word

        avg_loss = sum(losses) / len(losses)
        return avg_loss

    except Exception as e:
        print(f"Error in lower train: {e}")
        traceback.print_exc()

def main():
    #np.set_printoptions(threshold=sys.maxsize)
    training_file = "scene1.txt"

    lang = Language()
    lang.read(training_file)

    #nn = net(lang.corpus_size, lang.corpus_size // 2, 10)
    nn = net(lang.corpus_size, lang.corpus_size, lang.corpus_size)

    for i in range(10):
        epoch_loss = train(training_file, lang, nn, 10)
        print(f"Average epoch loss for epoch {i} is: {epoch_loss}")
    
if __name__ == "__main__":
    main()
