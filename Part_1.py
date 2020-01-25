#PART 1 : TOKENISATION, PART-OF-SPEECH TAGGING

# # Question 1: Reading the text from the given website url and identifying all the tokens and types before and after lowercasing and lemmatization


from urllib import request
from bs4 import BeautifulSoup

#The following steps for extracting text from any url has been referred from:https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python

print("******************** \n")
print("*** Reading from the url: https://www.theguardian.com/music/2018/oct/19/while-my-guitar-gently-weeps-beatles-george-harrison \n***")

url="https://www.theguardian.com/music/2018/oct/19/while-my-guitar-gently-weeps-beatles-george-harrison"
html = request.urlopen(url).read() 
soup = BeautifulSoup(html) #Extracting information after reading from the url

# Removing all script and style elements of the html syntax
for script in soup(["script", "style"]):
    script.extract()    

# Get text from the BeautifulSoup object
text = soup.get_text()

# Separating each line after removing the leading and trailing spaces
lines = (line.strip() for line in text.splitlines())
# breaking multiple headlines into a single line each
chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
# dropping all blank lines
text = '\n'.join(chunk for chunk in chunks if chunk)


# Importing packages for tokenisation and lemmatization from nltk

import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

tokens=word_tokenize(text) #tokenizing the obtained text

tokens_nopunct=[word for word in tokens if re.search("\w",word)] #removing all punctuations
tokens_updated=tokens_nopunct #Updated list of tokens

print("********************\n")
print("*** Lowercasing ***\n")
print("The number of tokens before lowercase:",len(tokens_updated))
print("The number of types before lowercase",len(set(tokens_updated)))
print("\n")

tokens_lower=[x.lower() for x in tokens_updated] #converting all tokens to lowercase
tokens_updated=tokens_lower

print("The number of tokens after lowercase:",len(tokens_updated))
print("The number of types after lowercase",len(set(tokens_updated)))
print("\n")

print("********************\n")
print("*** Lemmatization ***\n")
print("The number of tokens before lemmatization:",len(tokens_updated))
print("The number of types before lemmatization:",len(set(tokens_updated)))

lemmatizer = WordNetLemmatizer() 
tokens_lem=[lemmatizer.lemmatize(x) for x in tokens_updated] #Lemmatizing the tokens
tokens_updated=tokens_lem

print("The number of tokens after lemmatization:",len(tokens_updated))
print("The number of types after lemmatization:",len(set(tokens_updated)))
print("\n")

# # Question 2(a): Assigning POS tags to all the tokens in the text used above

print("********************\n")
print("*** ASSIGNING POS TAGS ***\n")

nltk.download('averaged_perceptron_tagger')

tokens_tagged=nltk.pos_tag(tokens_updated) #Tagging all tokens

tags_set=[word[1] for word in tokens_tagged]

print("The text after POS tagging: \n")
print(tokens_tagged)


# # Question 2(b): To understand tagging errors we compare the pos_tag() method with a combined Bigram, Unigram tagger

from nltk import DefaultTagger, UnigramTagger, BigramTagger
from nltk.corpus import brown
nltk.download('brown')

text = brown.tagged_sents(categories='news')
t0 = DefaultTagger('NN')
t1 = UnigramTagger(text, backoff=t0)
t2 = BigramTagger(text, backoff=t1)

bigram_tagged=t2.tag(tokens_updated)
ref_tag_set=[word[1] for word in bigram_tagged]

print("********************\n")
print("*** UNDERSTANDING TAGGING ERRORS ***\n")
print("POS tags from the default pos_tag() method: \n",tags_set)
print("POS tags from the Bigram tagger in nltk: \n",ref_tag_set)


from nltk.metrics import ConfusionMatrix
from collections import Counter

cm = ConfusionMatrix(ref_tag_set, tags_set) #Creating a Confusion Matrix

labels=set(ref_tag_set + tags_set) #Getting all the tags present

true_positives = Counter()
false_negatives = Counter()
false_positives = Counter()

for i in labels:
    for j in labels:
        if i == j:
            true_positives[i] += cm[i,j]
        else:
            false_negatives[i] += cm[i,j]
            false_positives[j] += cm[i,j]
print("DATA OBTAINED FROM THE CONFUSION MATRIX (i.e the tagging errors resulting from the two techniques): \n")
print("Total Number of True Positives:", sum(true_positives.values()),"\n")
print("\nFalse Negatives:", sum(false_negatives.values()),"\n")
print("\nFalse Positives:", sum(false_positives.values()),"\n")



