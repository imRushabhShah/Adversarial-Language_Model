

## DataSet - kdnuggets-fake-news
This is a further development of the kdnuggets article on fake news classification by George McIntyre:

https://www.kdnuggets.com/2017/04/machine-learning-fake-news-accuracy.html

In his article McIntyre approaches document classification from a very classical perspective: applying a vector-model 
to the corpus and then using a Naive Bayes classifier.  

## Neural Models
Here we take it into the deep learning realm: we apply a deep convolutional network with a 
traininable word-embedding layer.
Also we are using LSTM model for variation purposes.

## Attack
The attack over the fake news dataset takes a fake truely classified sample and find all the synonymes around it.
Average Sentence Len: 486.48
	
#### Glove LSTM
```
	Model 
		test Accuracy: 84.96%
		
	attack Success Rate: 100% 
	Avg amount of pertube: 16.01%
	Std dev for pertube: 12.899
	min pertube : 0.1051 %
	max pertube : 66.66 %
	min count altered = 1 word out of 952 words
```

#### CNN with trainable embedding
```
	Model 
		test Accuracy: 84.96%
		
	attack Success Rate: 43.5 % 
	Avg amount of pertube: 19.19 %
	Std dev for pertube: 11.195 %
	min pertube : 0.2159 %
	max pertube : 70 %
	min count altered = 121 word out of 926 words
```

### Create model
```
from fake-news-classification import *

#for CNN model
cnn = run_CNN_model_creation()

for LSTM_model
lstm = run_LSTM_model_creation()
```
### Attack model
```
replace X with last success epoc weight saved and model with cnn or lstm
attack(X,model,SAVE_MODEL_PATH = './save/model/')

```
