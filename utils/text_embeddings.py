from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import nltk
import gensim.downloader as api
import numpy as np

# nltk.download("punkt")
GLOVE_MODEL_WIKAPEDIA = api.load("glove-wiki-gigaword-100")
GLOVE_MODEL_TWITTER = api.load("glove-twitter-100")

def get_doc2vec_value(text):
    words = simple_preprocess(text)
    # print(words)

    tagged_doc = TaggedDocument(words=words, tags=[0])
    # print(tagged_doc)

    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=20)
    model.build_vocab([tagged_doc])
    model.train([tagged_doc], total_examples=model.corpus_count, epochs=model.epochs)

    vector = model.infer_vector(words)

    return vector

def get_glove_document_vector_wikapedia(text):
    words = text.split()
    word_vectors = [GLOVE_MODEL_WIKAPEDIA.get_vector(word) for word in words if word in GLOVE_MODEL_WIKAPEDIA.key_to_index]

    document_vector = np.mean(word_vectors, axis=0)

    return document_vector

def get_glove_document_vector_twitter(text):
    words = text.split()
    word_vectors = [GLOVE_MODEL_TWITTER.get_vector(word) for word in words if word in GLOVE_MODEL_TWITTER.key_to_index]

    document_vector = np.mean(word_vectors, axis=0)

    return document_vector

















if __name__ == "__main__":
    text = "This is a sample sentence for Doc2Vec."

    # print(get_doc2vec_value(text))
    # print(get_glove_document_vector_wikapedia(text))
    # print(get_glove_document_vector_twitter(text))

   





