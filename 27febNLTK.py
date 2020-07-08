#import nltk
#nltk.download()

from nltk.tokenize import word_tokenize
text="mitrc is in alwar and jak. wo  titanic wala"
word_token=word_tokenize(text)
print(word_token)

print(text.split(" "))


from nltk.tokenize import sent_tokenize
print(sent_tokenize(text))


import re
s="abcdef ggsdgtd"
p="[a-zA-Z]"
rs=re.findall(p,s)
print(rs)

from nltk.util import ngrams
from nltk.tokenize import word_tokenize
text="NLP and NLU are comle and the data. And the anuj is gone mad. Manish is Gupta."
print(list(ngrams(word_tokenize(text), 4)))

#stemming
from nltk.stem import PorterStemmer
st=PorterStemmer()
li=["sleeping", "swimming", "having", "reading"]
for i in li:
    print(st.stem(i))
    
#stopwords

from nltk.corpus import stopwords
sw=stopwords.words("english")
#print(sw)
a="i went to market yesterday"
tk=word_tokenize(a)
d=[]
d_ns=[]
for i in tk:
    if(i in sw):
        d.append(i)
    else:
        d_ns.append(i)
print(d)
print(d_ns)

from nltk.stem import WordNetLemmatizer
w_l = WordNetLemmatizer()
text="geese"
print(w_l.lemmatize(text))


text="the dog killed the bat"
t_w=word_tokenize(text)
for i in t_w:
    print(nltk.pos_tag(i))