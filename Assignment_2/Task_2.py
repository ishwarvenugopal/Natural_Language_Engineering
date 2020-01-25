#The assignment has been done in pairs.
#Members: Ishwar Venugopal [1906084], Shreya Jadhav [1702121])

#Task 2: Read data from 'text1.txt' and find word similarities between all possible pairs (after proper pre-processing)

#Importing necessary packages
from nltk.corpus import wordnet
import nltk
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
from itertools import product

data=open("text1.txt",encoding="utf8").read() #opening the required file
tokens=nltk.word_tokenize(data) #tokenizing
tokens_nopunct=[word for word in tokens if re.search("\w",word)] #removing punctuations
tokens_lower=[x.lower() for x in tokens_nopunct] #converting to lower case
lemmatizer = WordNetLemmatizer() 
tokens_lem=[lemmatizer.lemmatize(x) for x in tokens_lower] #lemmatizing
vocab=set(tokens_lem) #retaining only the unique words

sims = [] #initialize an empty list to save similarity values

for word1 in vocab:
    for word2 in vocab:
        w1 = wordnet.synsets(word1)#finding synonyms
        w2 = wordnet.synsets(word2)
        if w1 and w2:
            val=w1[0].path_similarity(w2[0]) #finding similarity values
            sims.append((word1, word2, val))

for data in sims: #commment this line and the next if you dont want to print on screen
    print(data)
    
with open('similarity_counts.txt','w') as file: #open the file for writing
    for data in sims:
        file.write("{}\t{}\t{}\n".format(data[0],data[1],data[2])) #writing to file line by line
    
