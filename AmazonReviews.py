# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 11:21:33 2020

@author: SaniyaJaswani
"""
import os
#import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
nlp = spacy.load('en')
import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')                  #Stopwords corpus

os.chdir('C:/Users/SaniyaJaswani/Downloads')
data = pd.read_csv('Reviews.csv')


DescriveCat = data.describe(include=['O'])
grp = data.groupby('ProfileName')[['ProfileName','HelpfulnessNumerator','HelpfulnessDenominator']]
#.sort_values('HelpfulnessDenominator',ascending = True,inplace = True)
grp.sort('HelpfulnessDenominator',ascending = True,inplace = True)
data.dtypes
#import datetime
#data['TimeStamp'] = pd.to_datetime(data['Time'])
#datetime.datetime.fromtimestamp(data['Time'])
#data['label']= data['Score'].apply(lambda c:'Positive' if c>3 else if )

def partition(x):
    if x > 3:
        return 'positive'
    else:
        return 'negative'

data['label']=data['Score'].map(partition)

dataClean = data.drop_duplicates(subset = ['UserId','ProfileName','Time','Text'])
x = dataClean['Text']
y= dataClean['Score']

import re
temp =[]
snow = nltk.stem.SnowballStemmer('english')
for sentence in dataClean['Text']:
    #print(sentence)
    sentence = sentence.lower()                 # Converting to lowercase
    cleanr = re.compile('<.*?>')
    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations
    #print(sentence)
    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')] 
    #print(words)# Stemming and removing stopwords
    temp.append(words)
    
dataClean['Text'] = temp    


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state = 42)

sia = SentimentIntensityAnalyzer()
sia.polarity_scores(x_train[0])
dataClean['Score_pred'] = dataClean['Text'].apply(lambda review:sia.polarity_scores(review))
dataClean['compound'] = dataClean['Score_pred'].apply(lambda score: score['compound'])
dataClean['pred_label'] = dataClean['compound'].apply(lambda c:'positive' if c>=0 else 'negative')  


print(metrics.classification_report(dataClean['label'],dataClean['pred_label']))
print(metrics.accuracy_score(dataClean['label'],dataClean['pred_label']))

vect = TfidfVectorizer()
x_train_vect = vect.fit_transform(x_train)
model = MultinomialNB()
#model2= LinearSVC()
model.fit(x_train_vect,y_train)
y_pred = model.predict(x_test)
print(metrics.confusion_matrix(y_test,y_pred))
print(metrics.accuracy_score(y_test,y_pred))