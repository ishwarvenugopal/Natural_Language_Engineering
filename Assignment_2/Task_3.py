#The assignment has been done in pairs.
#Members: Ishwar Venugopal [1906084], Shreya Jadhav [1702121])

#Task 3: Replacing each word "SimLex999-100.txt" with its hypernyms and finding the similarity value between them

#importing necessary packages
from nltk.corpus import wordnet
import nltk
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

data=[] #Initializing empty list to save all the data
with open('SimLex999-100.txt','r') as file: #opening the required file to read from
    cnt=0
    for line in file:
        cnt+=1
        if(cnt!=1): #ignoring the line with the column names
            line=line.strip() #removing unnecessary blank spaces and empty lines
            line=line.split() #splitting the elements separated by blank spaces
            word1=line[0] #first word
            word2=line[1] #second word
            goldstd=line[2] #gold standard similarity value
            w1 = wordnet.synsets(word1) #Finding synonyms
            w2 = wordnet.synsets(word2)
            if w1 and w2:
                w1=w1[0] #the first element in the set of synonyms
                w2=w2[0]
                h1=w1.hypernyms() #finding the hypernyms
                h2=w2.hypernyms()
                if h1: #checking if the hypernym exists
                    hyp1=h1[0].name() #extracting the name of the hypernym object
                    hyp1=hyp1.split('.') 
                    hyp1=hyp1[0] #removing unecessary info and extracting just the required word
                else:
                    hyp1='None' #Assigning 'None' if there are no hypernyms found
                if h2:
                    hyp2=h2[0].name()
                    hyp2=hyp2.split('.')
                    hyp2=hyp2[0]
                else:
                    hyp2='None'
                if h1 and h2:
                    simval=h1[0].path_similarity(h2[0]) #calculating similarity between hypernyms
                else:
                    simval=0;
                data.append((word1,word2,goldstd,hyp1,hyp2,simval))

for item in data:
    print(item)
    
with open('original-pairs-hypernyms.txt','w') as file: #Opening the file for writing
    for row in data:
        file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(row[0],row[1],row[2],row[3],row[4],row[5])) #writing to file line by line
        
    
