# N-GRAM MODELS

# # Question 5: Computing a Unigram model from the Toy Dataset

import nltk
from nltk import word_tokenize
from collections import Counter
import pandas as pd 

print("---------- TOY DATASET ----------\n")
print("=== UNIGRAM MODEL === \n")

token_no=0 #to keep a track of the number of tokens (N)
with open("sampledata.txt") as train:
    for line in train:
        line=line.strip() #removing all spaces at the starting and end of each line
        line=line.split() #creating a list with words separated by spaces
        token_no+=len(line) 
        
print('The total number of tokens for unigrams is: ',token_no)

uni_vocab_list=[]
uni_vocab_list.append("UNK") #adding UNK token by default
vocab=1 #to count the number of words in the vocabulary
with open("sampledata.vocab.txt") as voc:
    for line in voc:
        vocab+=1
        line=line.strip()
        uni_vocab_list.append(line)
print(uni_vocab_list)
print("The total vocabulary for unigram is: ",vocab)

print("\n")

unigrams_f=Counter() #To count the frequency of each unigram
unigrams_f["UNK"]=0

with open("sampledata.txt") as train:
    for line in train:
        line=line.strip()
        line=line.split()
        for word in line:
            word=word.strip()
            unigrams_f[word]+=1
            
unigrams_p={} #to compute the probability of each unigram

for word in unigrams_f:
    unigrams_p[word]=unigrams_f[word]/token_no

unigrams_smoothed={} #to compute the smoothed probabilities of each unigram
for word in unigrams_f:
    unigrams_smoothed[word]=(unigrams_f[word]+1.0)/(token_no + vocab)

print("- Unsmoothed -\n")

for uni, freq in unigrams_p.items():
    print('{} {}'.format(uni, freq))

print("\n- Smoothed -\n")
for uni, freq in unigrams_smoothed.items():
    print('{} {}'.format(uni, freq))


# # Question 6: Computing a Bigram Model

print("\n=== BIGRAM MODEL === \n")

bi_vocab_list=[]
bi_vocab=3 # adding UNK,<s>,</s> by default
bi_vocab_list.append("UNK")
bi_vocab_list.append("<s>")
bi_vocab_list.append("</s>")

with open("sampledata.vocab.txt") as voc:
    for line in voc:
        bi_vocab+=1
        line=line.strip()
        bi_vocab_list.append(line)

print("The total vocabulary for bigram is: ",bi_vocab)

bigram_count=Counter() #to count the frequency of each bigram

with open("sampledata.txt") as train:
    for line in train:
        line=line.strip()
        line=line.split()
        for context in line:
            if context!=line[0]:
                bigram_count[context]=Counter() #initializing a counter for each bigram
                
with open("sampledata.txt") as train:
    for line in train:
        line=line.strip()
        line=line.split()
        i=0
        for context in line:
            if context!=line[0]:
                i+=1
                history=line[i-1]
                bigram_count[context][history]+=1


bigram_f={} #To save the frequency of each bigram from the Counter() to a dictionary 
bigram_p={} #To compute the probability of each bigram

for context in bi_vocab_list:
    if context!="<s>":
        bigram_f[context]={}
    for history in bi_vocab_list:
        if ((context!="<s>")and(history!="</s>")):
            bigram_f[context][history]=0 #initialising all possible bigram frequencies as zero
        
for context in bigram_count:
    for history in bigram_count[context]:
        bigram_f[context][history]=bigram_count[context][history] #getting frequency data from the Counter()

for context in bigram_f:
    bigram_p[context]={}
    for history in bigram_f[context]:
        if(history!="UNK"):
            bigram_p[context][history]=(bigram_f[context][history])/unigrams_f[history] #Computing probabilities
        elif(history=="UNK"):
            bigram_p[context][history]=0 #To avoid division by zero

bigram_smoothed={} #To compute smoothed probabilities of bigrams

for context in bigram_p:
    bigram_smoothed[context]={}
    for history in bigram_p[context]:
        bigram_smoothed[context][history]=(bigram_f[context][history]+1)/(unigrams_f[history]+bi_vocab)


#To display the results as a table

df1=pd.DataFrame(bigram_p).T #Creating a dataframe 
df2=pd.DataFrame(bigram_smoothed).T 
print("- Unsmoothed -\n")
print(df1)
print("\n- Smoothed -\n")
print(df2)

# # Question 7: Sentence Probabilities

uni_sent_prob=[] #Probabilities from unigram model
bi_sent_prob=[] #Probabilities from bigram model

with open("sampletest.txt") as test:
    for line in test:
        prob=1
        line=line.strip()
        line=line.split()
        for word in line:
            if word in unigrams_smoothed:
                prob=prob*unigrams_smoothed[word]
            else:
                prob=prob*unigrams_smoothed["UNK"]
        uni_sent_prob.append(prob)

with open("sampletest.txt") as test:
    for line in test:
        prob=1
        line=line.strip()
        line=line.split()
        i=0
        for context in line:
            if context!=line[0]:
                i+=1
                history=line[i-1]
                if context in bigram_smoothed:
                    if history in bigram_smoothed[context]:
                        prob=prob*bigram_smoothed[context][history]
                    else:
                        prob=prob*bigram_smoothed[context]["UNK"]
                else:
                    if history in bigram_smoothed["UNK"]:
                        prob=prob*bigram_smoothed["UNK"][history]
                    else:
                        prob=prob*bigram_smoothed["UNK"]["UNK"]
        bi_sent_prob.append(prob)
        
print("\n=== SENTENCE PROBABILITIES ===\n")
sent_data={"Unigram probability:":uni_sent_prob,"Bigram Probability:":bi_sent_prob}
df_sent=pd.DataFrame(sent_data).T
df_sent.columns=["Sent 1","Sent 2","Sent 3","Sent 4","Sent 5"]

print(df_sent)
