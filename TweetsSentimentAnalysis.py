# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:26:14 2020

@author: SaniyaJaswani
"""

import os
#import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.pipeline import Pipeline
import spacy
nlp = spacy.load('en')
import re
#import nltk
#from nltk.corpus import stopwords 
#nltk.download('stopwords')                  #Stopwords corpus
#nltk.download('wordnet')
#stop_words = set(stopwords.words("english"))
os.chdir('C:/Users/SaniyaJaswani/Downloads')
#from nltk.stem import WordNetLemmatizer
#lemmatizer = WordNetLemmatizer()
data = pd.read_csv('tweet.csv',engine = 'python')
