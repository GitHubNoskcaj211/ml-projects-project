from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import nltk
import gensim.downloader as api
import numpy as np
import pandas as pd

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import numpy as np

from transformers import BertTokenizer, BertModel
import torch

GLOVE_MODEL_WIKAPEDIA = api.load("glove-wiki-gigaword-100")
GLOVE_MODEL_TWITTER = api.load("glove-twitter-100")
BERT_MODEL_NAME = "bert-base-uncased"
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
BERT_MODEL = BertModel.from_pretrained(BERT_MODEL_NAME)


def generate_bert_embedding(description, tokenizer=BERT_TOKENIZER, model=BERT_MODEL):
    if description is None:
        return None

    tokens = tokenizer.tokenize(description)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])  # Pytorch tensor dtype

    with torch.no_grad():
        outputs = model(input_tensor)
        embedding = outputs[0][:, 0, :].numpy()  # Get the embedding for the [CLS] token

    return embedding


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
    word_vectors = [
        GLOVE_MODEL_WIKAPEDIA.get_vector(word)
        for word in words
        if word in GLOVE_MODEL_WIKAPEDIA.key_to_index
    ]

    document_vector = np.mean(word_vectors, axis=0)

    return document_vector


def get_glove_document_vector_twitter(text):
    words = text.split()
    word_vectors = [
        GLOVE_MODEL_TWITTER.get_vector(word)
        for word in words
        if word in GLOVE_MODEL_TWITTER.key_to_index
    ]

    document_vector = np.mean(word_vectors, axis=0)

    return document_vector


def process_description(description, processing_function):
    try:
        processed = processing_function(description)
    except Exception as e:
        try:
            processed = processing_function(description)
        except Exception as e:
            print(
                f"Error processing description: {description} with function {processing_function.__name__}. Error: {e}"
            )
            processed = None
    return processed


# Creates model from an array of phrases
def createDoc2VecModel(phrases):
    tagged_phrases = [
        TaggedDocument(words=simple_preprocess(phrase), tags=[i])
        for i, phrase in enumerate(phrases)
    ]

    model = Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=20)
    model.build_vocab(tagged_phrases)
    model.train(tagged_phrases, total_examples=model.corpus_count, epochs=model.epochs)

    return model


# Function predicts from a given model and phrase
def predict_from_model(model, phrase):
    if phrase is None:  # Check if the phrase is None
        return None  # Return a zero vector of shape (1, 100)
    try:
        words = simple_preprocess(phrase)
        vector = model.infer_vector(words)
        return np.array(vector.reshape(1, 100))  # Reshape the vector to (1, 100)
    except Exception as e:
        print(f"Error processing phrase: {phrase}, {e}")
        return None # Return a zero vector of shape (1, 100) in case of an error




if __name__ == "__main__":
    text = "This is a sample sentence for Doc2Vec."

    # print(get_doc2vec_value(text))
    # print(get_glove_document_vector_wikapedia(text))
    # print(get_glove_document_vector_twitter(text))
    # print("ok")
