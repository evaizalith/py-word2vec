from nn import net
import numpy as np
from corpus import Language
import sys
import io
import traceback
import matplotlib.pyplot as plt

def train(file, corpus_size, data, nn, epochs, lr=0.01, batch_size=32):
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            targets, contexts = zip(*batch)

            batch_size_actual = len(targets)
            x = np.zeros((batch_size_actual, corpus_size))
            x[np.arange(batch_size_actual), targets] = 1

            y = nn.forward(x)
            vec = nn.softmax(y)

            y_true = np.zeros_like(y)
            for j in range(batch_size_actual):
                y_true[j, contexts[j]] = 1.0 / len(contexts[j])

            loss = -np.sum(y_true * np.log(vec + 1e-8)) / batch_size_actual
            epoch_loss += loss

            d = (vec - y_true) / batch_size_actual

            nn.gradient(d, lr)

        print(f"epoch {epoch + 1}/{epochs}, loss: {epoch_loss / len(data)}")
        losses.append((epoch_loss / len(data)))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs. Epochs')
        ax.set_xlim(0, epochs)

        plt.plot(range(len(losses)), losses, color='blue', linewidth=2, label='Training Loss')

        ax.legend()
        ax.grid(True)

        #plt.tight_layout()
        plt.savefig('loss_epoch_graph.png')
        plt.close()

def main():
    #np.set_printoptions(threshold=sys.maxsize)
    training_file = "scene1.txt"
    epochs = 10
    window_size = 10

    lang = Language()
    lang.read(training_file)
    data = lang.build_dataset(training_file, 2)

    nn = net(lang.corpus_size, lang.corpus_size, lang.corpus_size * 2)

    train(training_file, lang.corpus_size, data, nn, epochs, lr=0.1, batch_size=32)

    #person1 = nn.forward()
    
if __name__ == "__main__":
    main()
