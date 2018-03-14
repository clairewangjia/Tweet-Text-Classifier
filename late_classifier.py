import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score


data_dir = './data' 

print("Loading data...")
with open(os.path.join(data_dir, 'tweets_processed.txt'), 'r', encoding='utf-8') as f_tweets:
	x_tweets = f_tweets.readlines()

with open(os.path.join(data_dir, 'description_processed.txt'), 'r', encoding='utf-8') as f_description:
	x_description = f_description.readlines()

with open(os.path.join(data_dir, 'hashtag_processed.txt'), 'r', encoding='utf-8') as f_hashtag:
	x_hashtag = f_hashtag.readlines()

with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding='utf-8') as f:
	y = np.array(f.readlines())
    
print("Extract features...")
x_tweets_feats = TfidfVectorizer().fit_transform(x_tweets)
x_description_feats = TfidfVectorizer().fit_transform(x_description)
x_hashtag_feats = TfidfVectorizer().fit_transform(x_hashtag)

print(x_tweets_feats.shape)
print(x_description_feats.shape)
print(x_hashtag_feats.shape)


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("Start training and predict...")
n = 10
prob_tweets= np.empty(shape=[0, n])
prob_description= np.empty(shape=[0, n])
prob_hashtag= np.empty(shape=[0, n])
kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
macro_f1 = 0
micro_f1 = 0
for train, test in kf.split(x_tweets_feats):
#     model = MultinomialNB().fit(x_tweets_feats[train], y[train])
#     model = KNeighborsClassifier(n_neighbors=3).fit(x_tweets_feats[train], y[train])
    model = RandomForestClassifier(n_estimators = 200, max_features = 7, random_state=0).fit(x_tweets_feats[train], y[train])
#     model = LogisticRegression().fit(x_tweets_feats[train], y[train])
    # model = svm.SVC(probability=True).fit(x_tweets_feats[train], y[train])
    prob = model.predict_proba(x_tweets_feats[test])
    predicts = model.predict(x_tweets_feats[test])
    print(classification_report(y[test],predicts))
    prob_tweets = np.concatenate((prob_tweets, prob))
    avg_p	+= precision_score(y[test],predicts, average='macro')
    avg_r	+= recall_score(y[test],predicts, average='macro')
    macro_f1  += f1_score(y[test],predicts, average='macro')
    micro_f1  += f1_score(y[test],predicts, average='micro')

print('Average Precision of tweets text classifer is %f.' %(avg_p/10.0))
print('Average Recall of tweets text classifer is %f.' %(avg_r/10.0))
print('Average Macro-F1 of tweets text classifer is %f.' %(macro_f1/10.0))
print('Average Micro-F1 of tweets text classifer is %f.' %(micro_f1/10.0))



kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
macro_f1 = 0
micro_f1 = 0
for train, test in kf.split(x_description_feats):
#     model = MultinomialNB().fit(x_description_feats[train], y[train])
#     model = KNeighborsClassifier(n_neighbors=3).fit(x_description_feats[train], y[train])
    model = RandomForestClassifier(n_estimators = 200, max_features = 7, random_state=0).fit(x_description_feats[train], y[train])
#     model = LogisticRegression().fit(x_description_feats[train], y[train])
#     model = SVC(probability=True).fit(x_description_feats[train], y[train])
    prob = model.predict_proba(x_description_feats[test])
    predicts = model.predict(x_description_feats[test])
    print(classification_report(y[test],predicts))
    prob_description = np.concatenate((prob_description, prob))

    avg_p   += precision_score(y[test],predicts, average='macro')
    avg_r   += recall_score(y[test],predicts, average='macro')
    macro_f1  += f1_score(y[test],predicts, average='macro')
    micro_f1  += f1_score(y[test],predicts, average='micro')

print('Average Precision of description classifer is %f.' %(avg_p/10.0))
print('Average Recall of description classifer is %f.' %(avg_r/10.0))
print('Average Macro-F1 of description classifer is %f.' %(macro_f1/10.0))
print('Average Macro-F1 of description classifer is %f.' %(macro_f1/10.0))


kf = KFold(n_splits=10)
avg_p = 0
avg_r = 0
macro_f1 = 0
micro_f1 = 0
for train, test in kf.split(x_hashtag_feats):
#     model = MultinomialNB().fit(x_hashtag_feats[train], y[train])
#     model = KNeighborsClassifier(n_neighbors=3).fit(x_hashtag_feats[train], y[train])
    model = RandomForestClassifier(n_estimators=200, max_features=7, random_state=0).fit(x_hashtag_feats[train], y[train])
#     model = LogisticRegression().fit(x_hashtag_feats[train], y[train])    
#     model = SVC(probability=True).fit(x_hashtag_feats[train], y[train])
    prob = model.predict_proba(x_hashtag_feats[test])
    predicts = model.predict(x_hashtag_feats[test])
    print(classification_report(y[test],predicts))
    prob_hashtag = np.concatenate((prob_hashtag, prob))
    avg_p   += precision_score(y[test],predicts, average='macro')
    avg_r   += recall_score(y[test],predicts, average='macro')
    macro_f1  += f1_score(y[test],predicts, average='macro')
    micro_f1  += f1_score(y[test],predicts, average='micro')

print('Average Precision of hashtag classifer is %f.' %(avg_p/10.0))
print('Average Recall of hashtag classifer is %f.' %(avg_r/10.0))
print('Average Macro-F1 of hashtag classifer is %f.' %(macro_f1/10.0))
print('Average Micro-F1 of hashtag classifer is %f.' %(micro_f1/10.0))



weights = []
for w1 in np.arange(0,0.7,0.01):
    for w2 in np.arange(0,0.7,0.01):
        w3 = 1-w1-w2
        weights.append([w1, w2, w3])


y_num = np.array([int(num[:-1]) for num in y])
precisions = []
recalls = []
macrof1 = []
microf1 = []

for i in range(len(weights)):
    w_tweets, w_description, w_hashtag  = weights[i]
    result_prob = w_tweets*prob_tweets + w_description*prob_description + w_hashtag*prob_hashtag
    result = np.argmax(result_prob, axis=1)  
    avg_p = precision_score(y_num, result, average='macro')
    avg_r = recall_score(y_num, result, average='macro')
    macro_f1  = f1_score(y_num,result, average='macro')
    micro_f1  = f1_score(y_num,result, average='micro')

    precisions.append(avg_p)
    recalls.append(avg_r)
    macrof1.append(macro_f1)
    microf1.append(micro_f1)


opt_id = np.argmax(macrof1)
print(weights[opt_id])
print('Optimal Precision of late fusion classifer is %f.' %precisions[opt_id])
print('Optimal Recall of late fusion classifer is %f.' %recalls[opt_id])
print('Optimal Macro-F1 of late fusion classifer is %f.' %macrof1[opt_id])
print('Optimal Micro-F1 of late fusion classifer is %f.' %microf1[opt_id])