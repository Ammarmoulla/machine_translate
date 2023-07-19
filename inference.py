import os
import pickle
import numpy as np
from keras.models import load_model
from preprocess import padding, ids_to_words

model = load_model('outputs/model.h5')

with open('outputs/tokenizer_en.pkl', 'rb') as f:
    token_en= pickle.load(f)

with open('outputs/tokenizer_fr.pkl', 'rb') as f:
    token_fr = pickle.load(f)

text = "new jersey is sometimes quiet"

sequences = token_en.texts_to_sequences([text])
print(sequences)

x = padding(sequences, length_of_pad=21, type_pad="post")
y_pred = model.predict(x)[0]
output = ids_to_words(y_pred, token_fr)
print()
print("=========prediction=========")
for word in output.split():
  if word != "<P>":
    print(word, end=" ")