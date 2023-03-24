#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
d = pd.read_csv("C:/Users/dridi/Downloads/Youtube01-Psy.csv")


# In[71]:


d.tail(10)


# In[72]:


len(d.query('CLASS == 1'))


# In[73]:


len(d.query('CLASS == 0'))


# In[74]:


len(d)


# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()


# In[76]:


dvec = vectorizer.fit_transform(d['CONTENT'])


# In[77]:


dvec


# In[78]:


analyze = vectorizer.build_analyzer()


# In[79]:


print(d['CONTENT'][349])
analyze(d['CONTENT'][349])


# In[80]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train = ['the',
 'first',
 'billion',
 'viewed',
 'this',
 'because',
 'they',
 'thought',
 'it',
 'was',
 'really',
 'cool',
 'the',
 'other',
 'billion',
 'and',
 'half',
 'came',
 'to',
 'see',
 'how',
 'stupid',
 'the',
 'first',
 'billion',
 'were']
vectorizer.fit(X_train)

feature_names = vectorizer.get_feature_names()
print(feature_names)


# In[81]:


dshuf = d.sample(frac=1)


# In[82]:


d_train = dshuf[:300]
d_test = dshuf[300:]
d_train_att = vectorizer.fit_transform(d_train['CONTENT']) # fit bag-of-words on training set
d_test_att = vectorizer.transform(d_test['CONTENT']) # reuse on testing set
d_train_label = d_train['CLASS']
d_test_label = d_test['CLASS']


# In[83]:


d_train_att


# In[84]:


d_test_att


# In[85]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=80)


# In[86]:


clf.fit(d_train_att, d_train_label)


# In[87]:


clf.score(d_test_att, d_test_label)


# In[88]:


from sklearn.metrics import confusion_matrix
pred_labels = clf.predict(d_test_att)
confusion_matrix(d_test_label, pred_labels)


# In[89]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, d_train_att, d_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[90]:


# load all datasets and combine them
d = pd.concat([pd.read_csv("C:/Users/dridi/Downloads/Youtube01-Psy.csv"),
               pd.read_csv("C:/Users/dridi/Downloads/Youtube02-KatyPerry.csv"),
               pd.read_csv("C:/Users/dridi/Downloads/Youtube03-LMFAO.csv"),
               pd.read_csv("C:/Users/dridi/Downloads/Youtube04-Eminem.csv"),
               pd.read_csv("C:/Users/dridi/Downloads/Youtube05-Shakira.csv")])


# In[91]:


len(d)


# In[92]:


len(d.query('CLASS == 1'))


# In[93]:


len(d.query('CLASS == 0'))


# In[94]:


dshuf = d.sample(frac=1)
d_content = dshuf['CONTENT']
d_label = dshuf['CLASS']


# In[95]:


# set up a pipeline
from sklearn.pipeline import Pipeline, make_pipeline
pipeline = Pipeline([
    ('bag-of-words', CountVectorizer()),
    ('random forest', RandomForestClassifier()),
])
pipeline


# In[96]:


# or: pipeline = make_pipeline(CountVectorizer(), RandomForestClassifier())
make_pipeline(CountVectorizer(), RandomForestClassifier())


# In[97]:


pipeline.fit(d_content[:1500],d_label[:1500])


# In[98]:


pipeline.score(d_content[15:], d_label[15:])


# In[30]:


pipeline.predict(["what a neat video!"])


# In[31]:


pipeline.predict(["plz subscribe to my channel"])


# In[32]:


scores = cross_val_score(pipeline, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[33]:


# add tfidf
from sklearn.feature_extraction.text import TfidfTransformer
pipeline2 = make_pipeline(CountVectorizer(),
                          TfidfTransformer(norm=None),
                          RandomForestClassifier())


# In[34]:


scores = cross_val_score(pipeline2, d_content, d_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[35]:


pipeline2.steps


# In[36]:


# parameter search
parameters = {
    'countvectorizer__max_features': (None, 1000, 2000),
    'countvectorizer__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'countvectorizer__stop_words': ('english', None),
    'tfidftransformer__use_idf': (True, False), # effectively turn on/off tfidf
    'randomforestclassifier__n_estimators': (20, 50, 100)
}
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(pipeline2, parameters, n_jobs=-1, verbose=1)


# In[37]:


grid_search.fit(d_content, d_label)


# In[38]:


print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[ ]:




