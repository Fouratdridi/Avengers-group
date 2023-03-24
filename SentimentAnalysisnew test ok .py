#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gensim, logging


# In[3]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[4]:


gmodel = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)


# In[5]:


gmodel['cat']


# In[6]:


gmodel['dog']


# In[7]:


gmodel['spatula']


# In[8]:


gmodel.similarity('cat', 'dog')


# In[9]:


gmodel.similarity('cat', 'spatula')


# In[10]:


from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# In[11]:


def extract_words(sent):
    sent = sent.lower()
    sent = re.sub(r'<[^>]+>', ' ', sent) # strip html tags
    sent = re.sub(r'(\w)\'(\w)', '\1\2', sent) # remove apostrophes
    sent = re.sub(r'\W', ' ', sent) # remove punctuation
    sent = re.sub(r'\s+', ' ', sent) # remove repeated spaces
    sent = sent.strip()
    return sent.split()


# In[12]:


# unsupervised training data
import re
import os
unsup_sentences = []

# source: http://ai.stanford.edu/~amaas/data/sentiment/, data from IMDB
for dirname in ["train/pos", "train/neg", "train/unsup", "test/pos", "test/neg"]:
    for fname in sorted(os.listdir(f"C:/Users/dridi/Downloads/aclImdb")):
        if fname[-4:] == '.txt':
            with open(f"C:/Users/dridi/Downloads/aclImdb" + dirname + "/" + fname, encoding='UTF-8') as f:
                sent = f.read()
                words = extract_words(sent)
                unsup_sentences.append(TaggedDocument(words, [dirname + "/" + fname]))

# source: http://www.cs.cornell.edu/people/pabo/movie-review-data/
for dirname in ["review_polarity/txt_sentoken/pos", "review_polarity/txt_sentoken/neg"]:
    for fname in sorted(os.listdir(f"C:/Users/dridi/Downloads/review_polarity")):
        if fname[-4:] == '.txt':
            with open(f"C:/Users/dridi/Downloads/review_polarity" + dirname + "/" + fname, encoding='UTF-8') as f:
                for i, sent in enumerate(f):
                    words = extract_words(sent)
                    unsup_sentences.append(TaggedDocument(words, ["%s/%s-%d" % (dirname, fname, i)]))
                
# source: https://nlp.stanford.edu/sentiment/, data from Rotten Tomatoes
with open (f"C:/Users/dridi/Downloads/stanfordSentimentTreebank/original_rt_snippets.txt" , encoding='UTF-8') as f:
    for i, line in enumerate(f):
        words = extract_words(line)
        unsup_sentences.append(TaggedDocument(words, ["rt-%d" % i]))


# In[13]:


len(unsup_sentences)


# In[14]:


unsup_sentences[0:10]


# In[21]:


import random
class PermuteSentences(object):
    def __init__(self, sents):
        self.sents = sents
        
    def __iter__(self):
        shuffled = list(self.sents)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent


# In[22]:


permuter = PermuteSentences(unsup_sentences) 
model = Doc2Vec(permuter, dm=0, hs=1, vector_size=50)


# In[24]:


# done with training, free up some memory
del model


# In[26]:


from gensim.models.doc2vec import Doc2Vec

model = Doc2Vec.load('reviews.d2v')
# in other program, we could write: model = Doc2Vec.load('reviews.d2v')


# In[27]:


model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
    [model.infer_vector(extract_words("This place is not worth your time, let alone Vegas."))],
    [model.infer_vector(extract_words("Service sucks."))])


# In[ ]:


cosine_similarity(
    [model.infer_vector(extract_words("Highly recommended."))],
    [model.infer_vector(extract_words("Service sucks."))])


# In[ ]:


sentences = []
sentvecs = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]: 
    with open("sentiment labelled sentences/%s_labelled.txt" % fname, encoding='UTF-8') as f:
        for i, line in enumerate(f):
            line_split = line.strip().split('\t')
            sentences.append(line_split[0])
            words = extract_words(line_split[0])
            sentvecs.append(model.infer_vector(words, steps=10)) # create a vector for this document
            sentiments.append(int(line_split[1]))
            
# shuffle sentences, sentvecs, sentiments together
combined = list(zip(sentences, sentvecs, sentiments))
random.shuffle(combined)
sentences, sentvecs, sentiments = zip(*combined)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

clf = KNeighborsClassifier(n_neighbors=9)
clfrf = RandomForestClassifier()


# In[ ]:


scores = cross_val_score(clf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[ ]:


scores = cross_val_score(clfrf, sentvecs, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[ ]:


# bag-of-words comparison
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
pipeline = make_pipeline(CountVectorizer(), TfidfTransformer(), RandomForestClassifier())


# In[ ]:


scores = cross_val_score(pipeline, sentences, sentiments, cv=5)
np.mean(scores), np.std(scores)


# In[ ]:




