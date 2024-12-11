**ABBIE: Attention-Based BI-Encoders for Predicting Where to Split Compound Sanskrit Words**

This project implements the prediction of the split location in the Sanskrit compound words. 

Recognizing the exact location of splits in a compound word is difficult since several possible splits can be found, but only a few are semantically meaningful.
 
We propose a novel deep-learning method that uses **two bi-encoders and a multi-head attention** module to predict the valid split location in Sanskrit compound words.

ABBIE has been implemented in Python 3.10.14 with Keras API running on the TensorFlow backend. 

We used 2 attention heads in the multi-head attention layer.

We used batches of size 64 and trained the model for 40 epochs by the Adam optimizer.

We adopted the mean squared error (MSE) as a loss function.
