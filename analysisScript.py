from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from matplotlib import pyplot as plt
from keras.preprocessing.text import text_to_word_sequence
from gensim.models.phrases import Phrases, Phraser
import nltk
import heapdict
import sys
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def performStemming(tokens):
    stemmed_words = []
    ps = PorterStemmer()
    for token in tokens:
        stemmed = ps.stem(token)
        stemmed_words.append(stemmed)
    return stemmed_words


def removeStopWords(tokens):
    all_stopwords = stopwords.words('english')
    tokens_without_sw = [word for word in tokens if not word in all_stopwords]
    return tokens_without_sw


def performSentenceSegmentation(file_content):
    # Training the model using given text: unsupervised learning
    tokenizer = PunktSentenceTokenizer()
    tokenizer.train(file_content)
    sentence_segmentation = tokenizer.tokenize(file_content)
    return sentence_segmentation


def performPOSTagging(sentences):
    pos_tagged = {}
    for sentence in sentences:
        pos_tagged[sentence] = pos_tag(word_tokenize(sentence))
    return pos_tagged


def printPOSTagged(pos_tagged):
    for tag in pos_tagged:
        print(tag)
        print(pos_tagged[tag])
        print('\n\n')


# Get sentence length
def averageSentenceLength(segmented_sentence):
    total = 0
    size = len(segmented_sentence)
    for s in segmented_sentence:
        words = s.split()
        total += len(words)
    return total / size


# Plotting graph
def plotgraph(freqdict, graphname, xlabel, ylabel):
    x = list(freqdict.keys())
    y = list(freqdict.values())
    plt.figure()
    plt.bar(x, y, width=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    # plt.show()
    plt.savefig(graphname + '.png')
    plt.clf()


# the x-axis is the length of a token in number of characters, and the y-axis is the number of tokens of each length
def visualTokenAnalysis(tokens):
    token_analysis = {}
    for token in tokens:
        if len(token) in token_analysis.keys():
            token_analysis[len(token)] += 1
        else:
            token_analysis[len(token)] = 1
    return token_analysis


# the x-axis is the length of a sentence in number of tokens/words
# the y-axis is the number of sentences of each length
def visualSentenceAnalysis(sentence_segmentation):
    sentence_analysis = {}
    for sentence in sentence_segmentation:
        words = word_tokenize(sentence)
        words_size = len(words)
        if words_size in sentence_analysis.keys():
            sentence_analysis[words_size] += 1
        else:
            sentence_analysis[words_size] = 1
    return sentence_analysis


def top20Words(tokens):
    token_size = heapdict.heapdict()
    for token in tokens:
        if token in token_size.keys():
            token_size[token] -= 1
        else:
            token_size[token] = -1
    top20 = []
    for poptokens in range(20):
        top20.append(token_size.popitem()[0])
    return top20


def performVisualAnalysis(dataset, token_analysis, stemmed_token_analysis, sentence_analysis):
    plotgraph(token_analysis, "Token length analysis_" + dataset, "length of a token in number of characters", "number of tokens of each length")
    plotgraph(sentence_analysis, "Sentence length analysis_" + dataset, "length of a sentence in number of words/tokens", "number of sentences each length")
    plotgraph(stemmed_token_analysis, "Stemmed token length analysis_" + dataset, "length of a stemmed token in number of characters", "number of stemmed tokens of each length")


def extract_phrases(my_tree, phrase):
    my_phrases = []
    if my_tree.label() == phrase:
        my_phrases.append(my_tree.copy(True))
    for child in my_tree:
        if type(child) is nltk.Tree:
            list_of_phrases = extract_phrases(child, phrase)
            if len(list_of_phrases) > 0:
                my_phrases.extend(list_of_phrases)

    return my_phrases


def improveTokeniser(sentence_segmentation):
    phrases = []
    grammar = "NP: {<JJ>*<NN>|<NNP>*}"
    cp = nltk.RegexpParser(grammar)
    for x in sentence_segmentation:
        sentence = pos_tag(word_tokenize(x))
        tree = cp.parse(sentence)
        list_of_noun_phrases = extract_phrases(tree, 'NP')
        for phrase in list_of_noun_phrases:
            phrases.append("_".join([x[0] for x in phrase.leaves()]))
    return phrases


def writeToFile(filename, value):
    text_file = open(filename, "w")
    text_file.write(value)
    text_file.close()


def main(dataset):
    print("\n\n\n\n\n")
    print(".........STARTING ANALYSIS.........")
    print("\n\n")
    # 1: Load databases
    file_content = open('datasets/' + dataset).read()
    # 2: Split the content for each review
    each_reviews = file_content.split("---xxx---")
    # 3: Tokenisation
    tokens = text_to_word_sequence(file_content)
    writeToFile("1_tokens_" + dataset, "\n".join(tokens))
    # 4: Remove stop words from text
    tokens_without_sw = removeStopWords(tokens)
    writeToFile("2_tokens_without_sw_" + dataset, "\n".join(tokens_without_sw))
    # 5: Stemming
    stemmed_words = performStemming(tokens_without_sw)
    writeToFile("3_stemmed_words_" + dataset, "\n".join(stemmed_words))
    # 6: Top 20 words
    top20 = top20Words(tokens_without_sw)
    print("Top 20 words: ")
    print(top20)
    print("\n\n--------------\n\n")
    top20StemmedWords = top20Words(stemmed_words)
    print("Top 20 stemmed words: ")
    print(top20StemmedWords)
    print("\n\n--------------\n\n")
    # 7: Sentence segmentation
    sentence_segmentation = performSentenceSegmentation(file_content)
    writeToFile("4_sentence_segmentation_" + dataset, "\n".join(sentence_segmentation))
    # 8: Improving tokeniser by extracting noun phrases
    phrases = improveTokeniser(sentence_segmentation)
    writeToFile("5_phrases_with_improved_tokeniser_" + dataset, "\n".join(phrases))
    # 9: POS Tagging
    # 9.1 Sentences from each datasets 1, 2, 3
    if dataset=="dataset1.txt":
        sentences = ["All restaurants have children’s menus.", "Complimentary amenities include a welcome pack and daily ice-cream passes", "You won’t have to jostle with other hotel guests even if there’s a crowd."]
    elif dataset=="dataset2.txt":
        sentences = ["Now your work is saved on the branch 'my-saved-work' in case you decide you want it back (or want to look at it later or diff it against your updated branch).", 
        "Note that the first example assumes that the remote repo's name is 'origin' and that the branch named 'master' in the remote repo matches the currently checked-out branch in your local repo.", 
        "BTW, this situation that you're in looks an awful lot like a common case where a push has been done into the currently checked out branch of a non-bare repository."]
    elif dataset=="dataset3.txt":
        sentences = ["Unfortunately, no consensus has  emerged  about  the  form  or  the  existence  of  such  a  data  structure.", 
        "Such systems are often viewed as software components for constructing real-world NLP solutions.", 
        "Interactive Voice Response (IVR) applications used in call centers to respond to certain users’ requests."]
    # 9.2 POS Tagged sentences
    pos_tagged = performPOSTagging(sentences)
    print("POS Tagged sentence: ")
    printPOSTagged(pos_tagged)
    print("\n\n--------------\n\n")
    # 10: Average length of each sentence
    avg_length = averageSentenceLength(sentence_segmentation)
    print("Avg length of a sentence: ")
    print(avg_length)
    print("\n\n--------------\n\n")
    # 11: Graphical analysis
    token_analysis = visualTokenAnalysis(tokens_without_sw)
    stemmed_token_analysis = visualTokenAnalysis(stemmed_words)
    sentence_analysis = visualSentenceAnalysis(sentence_segmentation)
    performVisualAnalysis(dataset, token_analysis, stemmed_token_analysis, sentence_analysis)


if __name__ == "__main__":
    dataset = sys.argv[1]
    main(dataset)
