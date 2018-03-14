####A0176605B   Wang Jia#####
Lab 1: Microblog Clssification
Deadline: 19 Feb 2018

### Environment Setting
1. Python 3.6

### Installation 
1. nltk 
2. simplejson
3. pickle
4. numpy
5. scipy
6. scikit-learn


### Usage
## Early Fusion Classifier 
1. Run 'early_processor.py' to prepocess tweet attributes, including feature extraction and combination, data cleaning (e.g., remove url, punctuations, time), word tokenize, stemming, remove low-frequency words and stopping words. 
2. Run 'early_classifier.py', which adopts a random forest classifier. You are supposed to see performances (classification score, average percision, average recall, f1 score) printing into the screen.

## Late Fusion Classifier
1. Run 'late_processor.py' to prepocess tweet attributes, including data cleaning (e.g., remove url, punctuations, time), word tokenize, stemming, remove low-frequency words and stopping words. The three features(tweets texts, description, hashtag) will extracted and 
processed seperately.
2. Run 'late_classifier.py', which adopts 3 individual classifier and a combined model. You are supposed to see performances (classification score, average percision, average recall, f1 score) of the 3 single model and the final combined model printing into the screen.