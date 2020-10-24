from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tag import pos_tag
import nltk
nltk.download('averaged_perceptron_tagger')

file_content = open('datasets/dataset1.txt', encoding='utf8').read()

# Tokenisation
tokens = word_tokenize(file_content)

# Stemming
stemmed_words = []
ps = PorterStemmer() 
for token in tokens:
    stemmed = ps.stem(token)
    stemmed_words.append(stemmed)

# Sentence Segmentation

# Training the model using given text: unsupervised learning
tokenizer = PunktSentenceTokenizer()
tokenizer.train(file_content)

sentence_segmentation = tokenizer.tokenize(file_content)

# POS Tagging
sentence_domain1 = "All restaurants have children’s menus."
sentence_domain2 = "All restaurants have children’s menus."
sentence_domain3 = "All restaurants have children’s menus."

pos_tagged = pos_tag(word_tokenize(sentence))

print(pos_tagged)

print(tokens)

print(stemmed_words)

print(sentence_segmentation)