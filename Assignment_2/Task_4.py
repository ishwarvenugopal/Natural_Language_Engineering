#The assignment has been done in pairs.
#Members: Ishwar Venugopal [1906084], Shreya Jadhav [1702121])

#Task 4: Listing out the top 10 similar pairs

from nltk.corpus import wordnet
import nltk
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')
from itertools import product
import operator
import pandas as pd

data=open("text1.txt",encoding="utf8").read()
tokens=nltk.word_tokenize(data)
tokens_nopunct=[word for word in tokens if re.search("\w",word)]
tokens_lower=[x.lower() for x in tokens_nopunct]
lemmatizer = WordNetLemmatizer() 
tokens_lem=[lemmatizer.lemmatize(x) for x in tokens_lower]
vocab=set(tokens_lem)

sims = []

for word1 in vocab:
    for word2 in vocab:
        w1 = wordnet.synsets(word1)
        w2 = wordnet.synsets(word2)
        if w1 and w2:
            val=w1[0].path_similarity(w2[0])
            sims.append((word1, word2, val))

values={}
for i,item1 in enumerate(sims):
    if (item1[0]!=item1[1]):
        if isinstance(item1[2],float):
            values[(item1[0],item1[1])]=item1[2]

sorted_values = sorted(values.items(), key=operator.itemgetter(1))
df=pd.DataFrame(sorted_values)
df.columns=['Word Pair','Similarity']
final_df = df.sort_values(by=['Similarity'], ascending=False)
print(final_df.head(10))

with open('top.txt','w') as file:
    for index,row in final_df.iterrows():
        file.write("{}\t{}\n".format(row['Word Pair'],row['Similarity']))
