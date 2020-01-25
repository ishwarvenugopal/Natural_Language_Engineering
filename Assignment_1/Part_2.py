# PART 2: REGULAR EXPRESSIONS, FSAs and FSTs

# # Question 3: Identifying phone numbers from a given url and displaying it

# Note: The Regular Expression used here can detect RE of these main types:
# 1) +55 51 33083838
# 2) 1206 872020
# 3) 01206 872020
# 4) 05679401945
# 5) +44 5679401945
# 6) 0044 5679401945
# 7) +44 0 1206 873333


from urllib import request
from bs4 import BeautifulSoup
from urllib.parse import urlparse

url="test" #Initializing with a dummy url value

#Checking whether it is a valid url

validate=urlparse(url)

while(not(validate.scheme) and not(validate.netloc)):
    url=input("Enter a valid url: ")
    validate=urlparse(url)

#Getting data from the url as per the steps given on: https://stackoverflow.com/questions/328356/extracting-text-from-html-file-using-python    

html = request.urlopen(url).read()
soup = BeautifulSoup(html)

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

#Using NLTK functions to tokenize, lowercase and lemmatize the text obtained from the url

import nltk
nltk.download('punkt')
from nltk import word_tokenize
from nltk import re
from nltk.stem import WordNetLemmatizer 
nltk.download('wordnet')

tokens=word_tokenize(text)
tokens_nopunct=[word for word in tokens if re.search("\w",word)] # removing punctuations
tokens_updated=tokens_nopunct
tokens_lower=[x.lower() for x in tokens_updated] #lowercasing all tokens
tokens_updated=tokens_lower
lemmatizer = WordNetLemmatizer() 
tokens_lem=[lemmatizer.lemmatize(x) for x in tokens_updated] #Lemmatizing all tokens
tokens_updated=tokens_lem

text=nltk.Text(tokens_updated)
text=' '.join(text)

search=re.compile('[+]?0?0?[0-9]{2}[\s]?[0-9]{1,3}[\s]?[0-9]{4}[\s]?[0-9]{3,6}') #Searching for matches to the regular expression
iterator = search.finditer(text)
if(iterator):
    print("List of telephone numbers: ")
flag=True
for match in iterator:
    flag=False
    print(match.group())
if(flag==True):
    print("Sorry! No telephone numbers found!")






