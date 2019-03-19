import nltk
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import distance
from pdb import set_trace
from os import listdir
w2v = KeyedVectors.load_word2vec_format("vi/vi.vec")
vocab = w2v.wv.vocab

def read_file() :
    f = open("test.txt", "r+")
    return f.read()

def pre_process(content):
    content = content.lower()
    content = content.replace('\n', '. ')
    content = content.strip()
    content = content.replace('“', '')
    content = content.replace('”', '')
    # content = content.replace(".", "")
    return content

def sentences_tokenize(content):
    sentences = nltk.sent_tokenize(content)
    sentences = list(map(lambda x: x.replace(".", ""), sentences))
    sentences = [item for item in sentences if item != ""]
    return sentences

def similarity(vec1 , vec2):
    return 1 - distance.cosine(vec1, vec2)

def segmentation(content, threshold):
    preprocessed_text = pre_process(content)
    sentences = sentences_tokenize(preprocessed_text)
    paragraphs = []
    paragraph = [sentences[0]]
    sentences_vec = []
    for s in sentences:
        words_tokenize = ViTokenizer.tokenize(s)
        words = words_tokenize.split(" ")
        sentence_vec = np.zeros((100))
        for w in words:
            if w in vocab:
                 sentence_vec += w2v.wv[w]
        # set_trace()
        sentences_vec.append(sentence_vec)
    for i in range(len(sentences_vec) - 2):
        print(sentences[i])
        print(sentences[i + 1])
        print(similarity(sentences_vec[i], sentences_vec[i + 1]))
        if(similarity(sentences_vec[i], sentences_vec[i + 1]) > threshold):
            paragraph.append(sentences[i + 1])
        else:
            paragraphs.append(paragraph)
            paragraph = [sentences[i + 1]]
    paragraphs.append(paragraph)
    return paragraphs

def to_document(paragraphs):
    paragraphs = [".".join(item) for item in paragraphs]
    document = "\n ======= \n".join(paragraphs)
    return document

def list_files(directory, extension):
  return (f for f in listdir(directory) if f.endswith('.' + extension))

def saveDirectory(directory):
  files = list_files(directory, "txt")
  for f in files:
    file = open(directory + '/' + f, 'r+')
    content = file.read()
    paragraphs = segmentation(content, 0.8)
    doc_convert = to_document(paragraphs)
    new_file = open("/home/tran.minh.chien/Workspace/Python/docSegmentation/segmented_docs/" + f, mode="w", encoding="utf-8")
    new_file.write(doc_convert)
    new_file.close()

if __name__ == "__main__":
    directory = "/home/tran.minh.chien/Workspace/Python/docSegmentation/raw_docs"
    saveDirectory(directory)