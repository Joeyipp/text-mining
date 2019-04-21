# PART 1: EXTRACTING FEATURES
# COMMANDLINE USAGE: python3 feature-extract.py mini_newsgroups feature_definition_file class_definition_file training_data_file

import os
import sys
import math
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import jsonpickle

from tqdm import tqdm
from random import shuffle
from collections import OrderedDict
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import EnglishStemmer

class Posting:
    def __init__(self, docid, classlabel):
        self.docid = docid
        self.classlabel = classlabel
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' Sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' Return the term frequency in the document'''
        # TF is defined as the number of times the term appears in a given document
        # The current TF implementation does not account any log weighting or length normalization
        return len(self.positions)

class IndexItem:
    def __init__(self, term, __unique_id):
        self.term = term
        self.feature_id = __unique_id
        self.posting = {}
        self.sorted_postings = []

    def add(self, docid, pos, classlabel):
        ''' Add a posting'''
        if docid not in self.posting:
            self.posting[docid] = Posting(docid, classlabel)
        self.posting[docid].append(pos)

    def sort(self):
        ''' Sort by document ID for more efficient merging. For each document also sort the positions'''
        # Sort by document ID and save the sorted docID order to the self.sorted_postings list
        self.sorted_postings = sorted(self.posting.keys())

        # Sort the positions within each document
        for docid in self.sorted_postings:
            self.posting[docid].sort()

class InvertedIndex:
    def __init__(self, tokenizer=RegexpTokenizer(r'\w+'), stemmer=EnglishStemmer(), stopwords=nltk.corpus.stopwords.words('english')):
        # Preprocessing Toolkits
        self.tokenizer = tokenizer          # NLTK Tokenizer
        self.stopwords = set(stopwords)     # NLTK Stopwords
        self.stemmer = stemmer              # NLTK Stemmer
        
        # Class Attributes
        self.items = {}                     # Dictionary of IndexItems
        self.nDocs = 0                      # Number of indexed documents
        self.__unique_id = 0                # Unique feature_id for each unique term
        
    # Indexing a Document object
    def indexDoc(self, doc):
        ''' Indexing a document, using the simple SPIMI algorithm '''
        document_terms = []

        # Tokenizing
        for token in [t.lower() for t in self.tokenizer.tokenize(doc.body)]:
            # Remove stopwords
            if token in self.stopwords:
                continue

            # Stemming
            if self.stemmer:
                token = self.stemmer.stem(token)
            
            document_terms.append(token)
            
        # Generate the (term, position) pair
        terms_with_positions = []
        for i in range(len(document_terms)):
            terms_with_positions.append((document_terms[i], int(i+1)))

        # Create an IndexItem object for each 
        for term in terms_with_positions:
            if term[0] not in self.items:
                self.__unique_id += 1
                self.items[term[0]] = IndexItem(term[0], self.__unique_id)
                self.items[term[0]].add(doc.docid, term[1], doc.classlabel)
            else:
                self.items[term[0]].add(doc.docid, term[1], doc.classlabel)
        
        self.nDocs += 1
    
        return document_terms

    def sort(self):
        ''' Sort all posting lists by docid'''
        for term in self.items:
            self.items[term].sort()

        ''' Sort the InvertedIndex by terms'''
        self.items = OrderedDict(sorted(self.items.items(), key=lambda t: t[0]))

    def find(self, term):
        return self.items.get(term, "None")
    
    def save(self, filename):
        ''' Save to disk'''
        print("Saving to disk...")
        with open(filename, "w") as f:
            for term in self.items:
                f.write(jsonpickle.encode(self.items[term]))
                f.write('\n')
            f.write(jsonpickle.encode(self.nDocs))
        
        print("InvertedIndex successfully saved to {}\n".format(filename))

    def load(self, filename):
        ''' Load from disk'''
        print("Loading from disk...")
        with open(filename, "r") as f:
            data = f.readlines()
            for i in range(len(data)-1):
                indexItem = jsonpickle.decode(data[i])
                self.items[indexItem.term] = indexItem
            self.nDocs = jsonpickle.decode(data[-1])
        
        print("InvertedIndex successfully loaded to memory from {}\n".format(filename))

    def idf(self, term):
        ''' Compute the inverted document frequency for a given term'''
        # Return the IDF of the term
        # log(total documents/ documents with term i)
        return math.log(self.nDocs / len(self.items[term].sorted_postings), 10)

class Document:
    def __init__(self, docid, body, classlabel):
        self.docid = int(docid)
        self.body = body
        self.classlabel = int(classlabel)

def doc_parser(doc, classlabel):
    ''' Parse and create a Document Class Object '''
    with open(doc, 'r', encoding="ascii", errors="surrogateescape") as f:
        body = " "
        docid = doc.split("/")[-1]
        textblock = []
        num_lines = -1
        
        data = f.readlines()
        for line in data:
            num_lines += 1
            if line.startswith("Subject"):
                subj = " "
                header = line.strip().split(":")[1:]
                textblock.append(subj.join(header))

            elif line == "\n":
                break

        for line in data[num_lines:]:
            textblock.append(line.strip())
 
    body = body.join(textblock)
    doc = Document(docid, body, classlabel)

    return doc

def class_definition():
    # Create the class mappings
    class_mappings = {"comp.graphics": 1,
                      "comp.os.ms-windows.misc": 1,
                      "comp.sys.ibm.pc.hardware": 1,
                      "comp.sys.mac.hardware": 1,
                      "comp.windows.x": 1,
                      "rec.autos": 2,
                      "rec.motorcycles": 2,
                      "rec.sport.baseball": 2,
                      "rec.sport.hockey": 2,
                      "sci.crypt": 3,
                      "sci.electronics": 3,
                      "sci.med": 3,
                      "sci.space": 3,
                      "misc.forsale": 4,
                      "talk.politics.misc": 5,
                      "talk.politics.guns": 5,
                      "talk.politics.mideast": 5,
                      "talk.religion.misc": 6,
                      "alt.atheism": 6,
                      "soc.religion.christian": 6}
    
    return class_mappings

def test():
    # Get directory of newsgroups data
    directory_of_newsgroups_data = sys.argv[1]
    mini_newsgroups = os.path.join(os.getcwd(), directory_of_newsgroups_data)

    # List of output files
    feature_definition_file = sys.argv[2]
    class_definition_file = sys.argv[3]
    training_data_file = sys.argv[4]

    # Instantiate an invertedIndex
    invertedIndex = InvertedIndex()
    parsed_document = []

    # Test Class Definition
    print("\n-> Testing Class Definition")
    class_mappings = class_definition()
    print(class_mappings)

    # Test Indexing
    newsgroups_directory = os.listdir(mini_newsgroups)[0]
    documents_path = os.path.join(mini_newsgroups, newsgroups_directory)
    
    # Test Class Mappings
    classlabel = class_mappings[newsgroups_directory]
    
    # Test Indexing and Document Parser
    document = os.listdir(documents_path)[0]
    doc = os.path.join(documents_path, document)
    docObj = doc_parser(doc, classlabel)
    print("\n-> Indexing {}".format(doc))
    print("---> Class Label: {}".format(docObj.classlabel))
    print("---> Document ID: {}".format(docObj.docid))
    print("---> Document Body:\n{}".format(docObj.body))
    
    # Sort the invertedIndex
    invertedIndex.sort()

    # Parsed Document
    parsed_document.append([docObj.docid, docObj.classlabel, invertedIndex.indexDoc(docObj)])
    print("\n-> Parsed Document Terms (Tokenized -> LowerCase -> Stopwords Removed -> Stemmed)")
    print("---> {}".format(parsed_document[0][2]))
    print("\nTotal documents indexed: {}".format(invertedIndex.nDocs))

    print("\n-> Term: Feature ID: TF")
    for term in parsed_document[0][2]:
        feature_id = invertedIndex.items[term].feature_id
        feature_tf = invertedIndex.items[term].posting[parsed_document[0][0]].term_freq()
        print("{}".format(term))
        print("FID: {}, TF: {}\n".format(feature_id, feature_tf))

def main():
    # Get directory of newsgroups data
    directory_of_newsgroups_data = sys.argv[1]
    mini_newsgroups = os.path.join(os.getcwd(), directory_of_newsgroups_data)

    # List of output files
    feature_definition_file = sys.argv[2]
    class_definition_file = sys.argv[3]
    training_data_file = sys.argv[4]

    # Instantiate an invertedIndex and create class mappings
    invertedIndex = InvertedIndex()
    class_mappings = class_definition()
    parsed_documents = []

    # Index all documents in the mini_newsgroups directory
    for newsgroups_directory in os.listdir(mini_newsgroups):
        classlabel = class_mappings[newsgroups_directory]
        documents_path = os.path.join(mini_newsgroups, newsgroups_directory)
        print("\n-> Indexing {}".format(documents_path))

        with tqdm(total=len(os.listdir(documents_path))) as pbar:
            for document in os.listdir(documents_path):
                doc = os.path.join(documents_path, document)
                docObj = doc_parser(doc, classlabel)
                parsed_documents.append([docObj.docid, docObj.classlabel, invertedIndex.indexDoc(docObj)])
                pbar.update(1)

    print("\nTotal documents indexed: {}".format(invertedIndex.nDocs))

    # Sort the invertedIndex
    invertedIndex.sort()

    # Shuffle the documents to train the classifiers better!
    shuffle(parsed_documents)

    # Generate features (id and value) and write to training_data_files
    print("\n-> Generating {}s".format(training_data_file))
    with open("{}.TF".format(training_data_file), 'w') as f1:
        with open("{}.IDF".format(training_data_file), 'w') as f2:
            with open("{}.TFIDF".format(training_data_file), 'w') as f3:
                with tqdm(total=len(parsed_documents)) as pbar:
                    for document in parsed_documents:
                        features_tf = {}
                        features_idf = {}
                        features_tfidf = {}

                        for term in document[2]:
                            feature_id = invertedIndex.items[term].feature_id
                            feature_tf = invertedIndex.items[term].posting[document[0]].term_freq()
                            feature_idf = invertedIndex.idf(term)

                            features_tf[feature_id] = feature_tf
                            features_idf[feature_id] = feature_idf
                            features_tfidf[feature_id] = feature_tf * feature_idf

                        list_tf = []
                        list_idf = []
                        list_tfidf = []

                        for key in sorted(features_tf.keys()):
                            list_tf.append("{}:{}".format(key, features_tf[key]))
                        
                        for key in sorted(features_idf.keys()):
                            list_idf.append("{}:{:.5f}".format(key, features_idf[key]))
                        
                        for key in sorted(features_tfidf.keys()):
                            list_tfidf.append("{}:{:.5f}".format(key, features_tfidf[key]))

                        training_data_tf = " ".join(list_tf)
                        training_data_idf = " ".join(list_idf)
                        training_data_tfidf = " ".join(list_tfidf)

                        f1.write("{} {}\n".format(document[1], training_data_tf))
                        f2.write("{} {}\n".format(document[1], training_data_idf))
                        f3.write("{} {}\n".format(document[1], training_data_tfidf))
                        pbar.update(1)

    # Generate a sorted features definition list
    features = []
    for item in invertedIndex.items.keys():
        features.append((invertedIndex.items[item].feature_id, item))
    features.sort()

    # Write to feature_definition_file
    print("\n-> Generating {}".format(feature_definition_file))
    with open("{}".format(feature_definition_file), 'w') as f:
        for feature in features:
            f.write(str(feature[0]) + ", " + feature[1] + "\n")
    print("Done!")

    # Write to class_definition_file
    print("\n-> Generating {}".format(class_definition_file))
    with open("{}".format(class_definition_file), 'w') as f:
        for mapping in class_mappings:
            f.write(mapping + ", " + str(class_mappings[mapping]) + "\n")
    print("Done!")
    
    print("\n-> All done!")

    # Save the invertedIndex
    # invertedIndex.save("index_file")

if __name__ == '__main__':
    main()
    #test()
