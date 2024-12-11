                  ![image](https://github.com/user-attachments/assets/c43b80a6-fb99-4be1-8eb7-4cd131da44c6)

**ABBIE: Attention-Based BI-Encoders for Predicting Where to Split Compound Sanskrit Words**

This project implements the prediction of the split location in the Sanskrit compound words. 

Recognizing the exact location of splits in a compound word is difficult since several possible splits can be found, but only a few are semantically meaningful.
 
We propose a novel deep-learning method that uses **two bi-encoders to encode character-level contextual information for direct and reverse sequence and multi-head attention** module to predict the valid split location in Sanskrit compound words.

ABBIE has been implemented in Python 3.10.14 with Keras API running on the TensorFlow backend. 

We adopted the mean squared error (MSE) as a loss function.
