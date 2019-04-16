import os
import sys
import math
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
import jsonpickle

from tqdm import tqdm
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
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
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
        ''' add a posting'''
        if docid not in self.posting:
            self.posting[docid] = Posting(docid, classlabel)
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
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
        ''' indexing a document, using the simple SPIMI algorithm '''
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

    def sort(self):
        ''' sort all posting lists by docid'''
        for term in self.items:
            self.items[term].sort()

        ''' sort the InvertedIndex by terms'''
        self.items = OrderedDict(sorted(self.items.items(), key=lambda t: t[0]))

    def find(self, term):
        return self.items.get(term, "None")
    
    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index (Done)
        print("Saving to disk...")
        with open(filename, "w") as f:
            for term in self.items:
                f.write(jsonpickle.encode(self.items[term]))
                f.write('\n')
            f.write(jsonpickle.encode(self.nDocs))
        
        print("InvertedIndex successfully saved to {}\n".format(filename))

    def load(self, filename):
        ''' load from disk'''
        # ToDo (Done)
        print("Loading from disk...")
        with open(filename, "r") as f:
            data = f.readlines()
            for i in range(len(data)-1):
                indexItem = jsonpickle.decode(data[i])
                self.items[indexItem.term] = indexItem
            self.nDocs = jsonpickle.decode(data[-1])
        
        print("InvertedIndex successfully loaded to memory from {}\n".format(filename))

    def idf(self, term):
        ''' compute the inverted document frequency for a given term'''
        # ToDo: return the IDF of the term
        # log(total documents/ documents with term i)
        return math.log(self.nDocs / len(self.items[term].sorted_postings), 10)

class Document:
    def __init__(self, docid, body, classlabel):
        self.docid = int(docid)
        self.body = body
        self.classlabel = int(classlabel)

def doc_parser(doc, classlabel):
    with open(doc, 'r', encoding = 'unicode_escape') as f:
        body = ""
        docid = doc.split("/")[-1]
        list_of_text = []
        num_lines = 0
        
        line = f.readline().strip()
        while line != "":
            if line.startswith("Subject"):
                subj = ""
                header = line.split(":")[1:]
                list_of_text.append(subj.join(header))

            elif line.startswith("Lines"):
                num_lines = int(line.split(":")[1].strip())

            line = f.readline().strip()
        
        i = 0
        while(i != num_lines):
            line = f.readline().strip()
            list_of_text.append(line)
            i += 1
    
    body = body.join(list_of_text)
    doc = Document(docid, body, classlabel)

    return doc

def class_definition():
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

def main():

    # Get directory of newsgroups data
    directory_of_newsgroups_data = sys.argv[1]
    mini_newsgroups = os.path.join(os.getcwd(), directory_of_newsgroups_data)

    # # List of output files
    # feature_definition_file = sys.argv[2]
    # class_definition_file = sys.argv[3]
    # training_data_file = sys.argv[4]

    invertedIndex = InvertedIndex()
    class_mappings = class_definition()

    for newsgroups_directory in os.listdir(mini_newsgroups):
        classlabel = class_mappings[newsgroups_directory]
        documents_path = os.path.join(mini_newsgroups, newsgroups_directory)
        print("\n-> Indexing {}".format(documents_path))

        with tqdm(total=len(os.listdir(documents_path))) as pbar:
            for document in os.listdir(documents_path):
                doc = os.path.join(documents_path, document)
                print(document)
                docObj = doc_parser(doc, classlabel)
                invertedIndex.indexDoc(docObj)
                pbar.update(1)

    # Write to class_definition_file
    # with open(class_definition_file, 'w') as f:
    #     for mapping in class_mappings:
    #         f.write(mapping + ", " + str(class_mappings[mapping]))


main()
