#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim, logging


# In[2]:


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[3]:


from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec


# In[4]:


with open("C:/Users/dridi/Downloads/yelp_labelled.txt") as f:
    for item_no, line in enumerate(f):
        print(item_no, line)


# In[5]:


sentences = []
sentiments = []
with open ("C:/Users/dridi/Downloads/yelp_labelled.txt") as f:
    for item_no, line in enumerate(f):
        line_split = line.strip().split('\t')
        sentences.append((line_split[0], "yelp_%d" % item_no))
        sentiments.append(int(line_split[1]))


# In[6]:


len(sentences), sentences


# In[7]:


from gensim.models.doc2vec import TaggedDocument
import re

sentences = []
sentiments = []
for fname in ["yelp", "amazon_cells", "imdb"]:
    with open (f"C:/Users/dridi/Downloads/{fname}_labelled.txt") as f:
        for item_no, line in enumerate(f):
            line_split = line.strip().split('\t')
            sent = line_split[0].lower()
            sent = re.sub(r'\'', '', sent)
            sent = re.sub(r'\W', ' ', sent)
            sent = re.sub(r'\s+', ' ', sent).strip()
            sentences.append(TaggedDocument(words=sent.split(), tags=["%s_%d" % (fname, item_no)]))
            sentiments.append(int(line_split[1]))


# In[8]:


sentences


# In[9]:


import random
class PermuteSentences(object):
    def __iter__(self):
        shuffled = list(sentences)
        random.shuffle(shuffled)
        for sent in shuffled:
            yield sent
permuter = PermuteSentences()
        
model = Doc2Vec(permuter, min_count=1)


# In[10]:


model.wv.most_similar('tasty')


# In[ ]:




