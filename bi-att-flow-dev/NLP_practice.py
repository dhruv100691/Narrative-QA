from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

lem=WordNetLemmatizer()
stemmer=PorterStemmer()

word="multiplying"
print (lem.lemmatize(word,"v"))