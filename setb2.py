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
ct=df["comment_text"]
print("\n",ct,"\n")
#clean daatset 
print(df.isnull().sum())
sw=set(stopwords.words("english"))
print("\n Stopwords..... \n")
print(sw)
print("\n Word Tokens..... \n")
'''
wd={}
for w in ct:
    wd=word_tokenize(w)
    print(wd)

'''#sentiment ana.
vd=SentimentIntensityAnalyzer()
wd={}
for w in ct:
    wd=word_tokenize(w)
    print(wd)
    print("\n",vd.polarity_scores(w))
    


