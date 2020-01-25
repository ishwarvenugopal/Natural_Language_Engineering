#The assignment has been done in pairs.
#Members: Ishwar Venugopal [1906084], Shreya Jadhav [1702121])

#Task 1: To Calculate the word similarities in 'SimLex999-100.txt' using path_similarity function of NLTK

#Importing necessary packages
from nltk.corpus import wordnet
import nltk
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

data=[] #initialise an empty list to save all the data later
with open('SimLex999-100.txt','r') as file: #open the required file
    cnt=0
    for line in file: #read the file line by line
        cnt+=1 
        if(cnt!=1): #to ignore the line with column headings
            line=line.strip() #removes unwanted spaces and empty lines
            line=line.split() #splits each line to elements separated by blank space
            word1=line[0] #first word
            word2=line[1] #second word
            goldstd=line[2] #gold standard similarity value
            w1 = wordnet.synsets(word1) #finding synonyms
            w2 = wordnet.synsets(word2)
            if w1 and w2: 
                simval=w1[0].path_similarity(w2[0]) #finding similarity value
                data.append((word1,word2,goldstd,simval))

for item in data:
    print(item)

with open('SimLex999-100-predicted.txt','w') as file:
    for row in data:
        file.write("{}\t{}\t{}\t{}\n".format(row[0],row[1],row[2],row[3])) #writing to file line by line 
