# py-word2vec

This is my implementation of Word2Vec in Python using only Numpy, no pytorch or tensorflow.

## Dataset

The dataset used to test the model is Scene 1 from Shakespeare's play Hamlet. It contains 562 unique words and is primarily dialogue between characters. 

## Gradient Descent and Loss

The initial gradient is equal to the derivative of softmax + the derivative of cross-entropy, which is equivalent to $predictions - true values$

Each linear layer in the model can be represented with the formula $xA + b = y$ where x is the input, $A$ is a weights matrix, $b$ is a bias vector, and $y$ is the layer output.

Gradients are calculated from the end to the front, multiplying the gradient of the current layer by the gradient of the next ($d$).

Each layer contains the following function to calculate and apply gradient descent:
```python
    # Calculate and apply gradient for this layer then pass on new derivative to next layer 
    def backward(self, d, lr, l):
        d = np.atleast_2d(d)
        grad_A = self.x.T @ d
        grad_b = d.sum(axis=0, keepdims=True)
        
        dx = d @ self.A.T

        self.A -= grad_A * lr
        self.b -= grad_b * lr

        return dx
```
This code takes in the loss from the next layer, multiplies the by the layer's input as stored during the forward step, and then applies this result to the weights (represented using the variable A)

The loss decreases over time as the model learns the encoding of words. Below, you can see the loss over time in a training session with 10 epochs.

![loss_epoch_graph](https://github.com/user-attachments/assets/a1c0e50c-3492-4bf1-8e78-7880264e2e7f)

## Word Similarity Analysis

In my implementation using my dataset, words tended to have similar embeddings. I tested the embeddings using 3 pairs of words: (Horatio, Marcellus), (Horatio, says), and (fair, gross). Horatio and Marcellus had a cosine of -0.04. These are distinct characters in the play. Horatio and "says" have a cosine of -0.02, while "fair" and "gross" have a cosine of 0.02. "Fair" and "gross" are used in the same way in the dataset, both being used as adjectives followed immediately by "and" and another adjective.  

## Limitations

The utility of Word2Vec and its limitations have been made clear. It is a very good implementation of an LM based on a simple fully connected model architecture, but the skip-gram model is ineffective at handling linguistic information over long distances. 
