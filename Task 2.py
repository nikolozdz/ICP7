import requests
from bs4 import BeautifulSoup
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk import ne_chunk

nltk.download()


html = requests.get("https://en.wikipedia.org/wiki/Google")

bsObj=BeautifulSoup(html.content,"html.parser")

print(bsObj.title.string)
finalString=''
filterSentence=''
bracket=0

file = open('input.txt','w')
for link in bsObj.find_all('p'):#Searching all the paragraphs in html
    stringTags=str(link) #convert the paragraph into a string and start parsing
    stringTags=stringTags.replace('<p>',' ') #Remove all the p tags
    stringTags = stringTags.replace('</p>', ' ')
    for chr in stringTags: # But yet there are more tags so we need to take out the content in the brackets
        if chr=='<': #if we see the enclosing < then increment bracket count
            bracket+=1
        elif chr=='>': # if we see the closing > then decrement
            bracket-=1
        #This elif statements makes sure that text is not enclosed in <> makes sure text is either alphabetic,numeric or is an element of tuple below
        elif bracket==0 and (chr.isalpha() or chr.isnumeric() or chr in (' ', '-', ',', '$', '.', '!', '?', ';', ':', "'", '"', '\n')):
            filterSentence+=chr #add the character to the filtered sentence
    finalString+=filterSentence #after loop add filltered sentence to a final string
    filterSentence='' #clear filter
file.write(finalString)
file.close()

###################################################################
ps = PorterStemmer()

lemmatizer = WordNetLemmatizer()
file = open('input.txt','r')
text=file.read()

wordTokens =word_tokenize(text)
sentenceTokens = sent_tokenize(text)
print("Word tokens:",wordTokens)
print("Sentence tokens:",sentenceTokens)
print('\n')
trigrams = ngrams(wordTokens,3)
print("Trigrams: ",list(trigrams))
print('\n')
lemmatized_output = ' '.join([lemmatizer.lemmatize(w) for w in wordTokens])
print("Lemmatization:")
print(lemmatized_output)
print('\n')
stemmed_output = ' '.join([ps.stem(chr) for chr in wordTokens])
print("Stemming:")
print(stemmed_output)
print('\n\n')

n_pos = nltk.pos_tag(wordTokens)
print("Parts of Speech :")
print(n_pos)
print('\n\n\n')

ner = ne_chunk(n_pos)
print("Named Entity Recognition :", ner)
print('\n')