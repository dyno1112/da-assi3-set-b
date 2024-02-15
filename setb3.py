import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

# Read the dataset
df = pd.read_csv('covid.csv')
print(df)
'''ct=df["comment_text"]
print("\n",ct,"\n")'''
#clean daatset 
print(df.isnull().sum())
