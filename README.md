<div style="position: relative; height: 200px;">
  <div style="position: absolute; top: 0; left: 0;">
    <img src="ITSERR.png" alt="Alt Text" width="200">
  </div>
</div>

# ABBIE: Attention-Based BI-Encoders for Predicting Where to Split Compound Sanskrit Words

This project implements the prediction of the split location in the Sanskrit compound words. 

Recognizing the exact location of splits in a compound word is difficult since several possible splits can be found, but only a few are semantically meaningful.
 
We propose a novel deep-learning method that uses **two bi-encoders to encode character-level contextual information for direct and reverse sequence and multi-head attention** module to predict the valid split location in Sanskrit compound words.

ABBIE has been implemented in Python 3.10.14 with Keras API running on the TensorFlow backend. 

We adopted the mean squared error (MSE) as a loss function.
