import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier


data_dir = './data' 

print("Loading data...")
with open(os.path.join(data_dir, 'early_samples_processed.txt'), 'r', encoding='utf-8') as f:
	x = f.readlines()
with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
	y = np.array(f.readlines())



print("Extract features...")
x_feats = TfidfVectorizer().fit_transform(x)
print(x_feats.shape)



print("Start training and predict...")
kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
macro_f1 = 0
micro_f1 = 0
for train, test in kf.split(x_feats):
	model = MultinomialNB().fit(x_feats[train], y[train])
    # model = RandomForestClassifier(n_estimators = 200, max_features = 7, random_state=0).fit(x_feats[train], y[train])
	predicts = model.predict(x_feats[test])
	print(classification_report(y[test],predicts))
	avg_p   += precision_score(y[test],predicts, average='macro')
	avg_r   += recall_score(y[test],predicts, average='macro')
	macro_f1  += f1_score(y[test],predicts, average='macro')
	micro_f1  += f1_score(y[test],predicts, average='micro')
	
print('Average Precision of early fusion classifer is %f.' %(avg_p/10.0))
print('Average Recall of early fusion classifer is %f.' %(avg_r/10.0))
print('Average Macro-F1 of early fusion classifer is %f.' %(macro_f1/10.0))
print('Average Micro-F1 of early fusion classifer is %f.' %(micro_f1/10.0))