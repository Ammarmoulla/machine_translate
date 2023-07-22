import os
import pickle
import argparse
import numpy as np
from keras.models import load_model
from preprocess import padding, ids_to_words


parser = argparse.ArgumentParser(description='Process some URLs.')
parser.add_argument('--model_path', type=str, help='The URL for type model in inference')
parser.add_argument('--text', type=str, help='The English Sentence')

args = parser.parse_args()
model_path = args.model_path
#text = "new jersey is sometimes quiet"
text = args.text

model = load_model(model_path)

with open('outputs/tokenizer_en.pkl', 'rb') as f:
    token_en= pickle.load(f)

with open('outputs/tokenizer_fr.pkl', 'rb') as f:
    token_fr = pickle.load(f)


sequences = token_en.texts_to_sequences([text])
print(sequences)

x = padding(sequences, length_of_pad=21, type_pad="post")
y_pred = model.predict(x)[0]
output = ids_to_words(y_pred, token_fr)
print()
print(text)
print("=========prediction=========")
for word in output.split():
  if word != "<P>":
    print(word, end=" ")