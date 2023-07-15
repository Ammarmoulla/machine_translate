import os
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

def max_seq(seq_input, seq_output):

  max_input = max([len(x) for x in seq_input])
  max_output = max([len(x) for x in seq_output])

  print("Max English sequence length before processing:", max_input)
  print("Max French sequence length before processing:", max_output)

  return max(max_input, max_output)

def read_data(path_data):

    input_file = os.path.join(path_data)
    with open(input_file, "r") as f:
        data = f.read()

    return data.split('\n')

def show_some_sample_and_some_statics(path_en, path_fr, number_sample=2):
    
    data_en = read_data(path_en)
    data_fr = read_data(path_fr)

    print("sample from 1 to {}".format(number_sample - 1))

    for i in range(number_sample - 1):
        print("sample {}: {}".format(i+1, data_en[i]))
        print("sample {}: {}".format(i+1, data_fr[i]))
    print("===================Same Statics===================")
    print("number sentence english is :", len(data_en))
    print("number sentence english is :", len(data_fr))
    
def convert_text_to_tokenize(list_of_senetence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(list_of_senetence)
    return tokenizer.texts_to_sequences(list_of_senetence), tokenizer

def padding(x, length_of_pad=None, type_pad="pre"):

  return pad_sequences(x, maxlen=length_of_pad, padding=type_pad)


def full_process(_input, _output, length=None, type_pad="post"):

  seq_input, token_input = convert_text_to_tokenize(_input)
  seq_output, token_output = convert_text_to_tokenize(_output)

  if length == None:
    length = max_seq(seq_input, seq_output)

  pad_input = padding(seq_input, length_of_pad=length, type_pad=type_pad)
  pad_output = padding(seq_output, length_of_pad=length, type_pad=type_pad)

  (num_sample, seq_length) = pad_output.shape

  pad_input = pad_input.reshape((num_sample, seq_length, 1))
  pad_output = pad_output.reshape((num_sample, seq_length, 1))

  return pad_input, pad_output, token_input, token_output

def ids_to_words(ids, tokenizer):

    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<P>'

    return ' '.join([index_to_words[prediction] for prediction in np.argmax(ids, 1)])