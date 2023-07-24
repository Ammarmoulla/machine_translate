import os
from pathlib import Path
import sys
import pickle
import numpy as np
from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = os.path.join(BASE_DIR, "outputs")

path_model = os.path.join(MODELS_DIR, "model_Lstm_Embd.h5")
print(path_model)
model = load_model(path_model)

path_model = os.path.join(MODELS_DIR, "model_Lstm_Embd.h5")
print(path_model)
# model = load_model(path_model)

with open(os.path.join(MODELS_DIR, "tokenizer_en.pkl"), 'rb') as f:
    token_en= pickle.load(f)

with open(os.path.join(MODELS_DIR, "tokenizer_fr.pkl"), 'rb') as f:
    token_fr = pickle.load(f)

def padding(x, length_of_pad=None, type_pad="pre"):

  return pad_sequences(x, maxlen=length_of_pad, padding=type_pad)

def ids_to_words(ids, tokenizer):

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<P>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(ids, 1)])

def en_to_fr(en_text):

    fr_text = ""
    sequences = token_en.texts_to_sequences([en_text])
    x = padding(sequences, length_of_pad=21, type_pad="post")

    y_pred = model.predict(x)[0]
    output = ids_to_words(y_pred, token_fr)

    for word in output.split():
      if word != "<P>":
        fr_text += word + " "


    # fr_text = "new jersey est parfois calme "
    fr_text = fr_text[:-1]
    
    result = {

        "fr_text": fr_text,
    }
    
    return result 
