import os
import sys
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer

# Class Index Skeleton Code Adopted from https://nlpforhackers.io/building-a-simple-inverted-index-using-nltk/
class Index:
    """ Inverted index datastructure """
 
    def __init__(self, tokenizer, stemmer=None, stopwords=None):
        """
        tokenizer   -- NLTK compatible tokenizer function
        stemmer     -- NLTK compatible stemmer 
        stopwords   -- list of ignored words
        """
        self.tokenizer = tokenizer
        self.stemmer = stemmer
        self.index = defaultdict(list)
        self.documents = {}
        self.__unique_id = 0
        if not stopwords:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)
 
    def lookup(self, word):
        """
        Lookup a word in the index
        """
        word = word.lower()
        if self.stemmer:
            word = self.stemmer.stem(word)
 
        return [self.documents.get(id, None) for id in self.index.get(word)]
 
    def add(self, document):
        """
        Add a document string to the index
        """
        # Convert to lower cases and tokenize
        for token in [t.lower() for t in nltk.word_tokenize(document)]:
            # Remove stopwords
            if token in self.stopwords:
                continue

            # Stemming
            if self.stemmer:
                token = self.stemmer.stem(token)
 
            # Generate a unique <feature-id> for each token <feature-value>
            if self.__unique_id not in self.index[token]:
                self.index[token].append(self.__unique_id)
 
        self.documents[self.__unique_id] = document
        self.__unique_id += 1           


def class_definition():
    class_mappings = ["comp.graphics, 1",
                      "comp.os.ms-windows.misc, 1",
                      "comp.sys.ibm.pc.hardware, 1",
                      "comp.sys.mac.hardware, 1",
                      "comp.windows.x, 1",
                      "rec.autos, 2",
                      "rec.motorcycles, 2",
                      "rec.sport.baseball, 2",
                      "rec.sport.hockey, 2",
                      "sci.crypt, 3",
                      "sci.electronics, 3",
                      "sci.med, 3",
                      "sci.space, 3",
                      "misc.forsale, 4",
                      "talk.politics.misc, 5",
                      "talk.politics.guns, 5",
                      "talk.politics.mideast, 5",
                      "talk.religion.misc, 6",
                      "alt.atheism, 6",
                      "soc.religion.christian, 6"]
    
    return class_mappings

def main():

    # Get directory of newsgroups data
    directory_of_newsgroups_data = sys.argv[1]
    path = os.path.join(os.getcwd(), directory_of_newsgroups_data)

    # List of output files
    feature_definition_file = sys.argv[2]
    class_definition_file = sys.argv[3]
    training_data_file = sys.argv[4]

    # for newsgroups_directory in os.listdir(path):
    #     print(newsgroups_directory)

    index = Index(nltk.word_tokenize, EnglishStemmer(), nltk.corpus.stopwords.words('english'))

    # Write to class_definition_file (DONE)
    with open(class_definition_file, 'w') as f:
        class_mappings = class_definition()
        for mapping in class_mappings:
            f.write(mapping + '\n')

main()
