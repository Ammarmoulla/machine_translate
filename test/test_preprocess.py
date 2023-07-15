import numpy as np
from train.preprocess import read_data
from train.preprocess import show_some_sample_and_some_statics
from train.preprocess import convert_text_to_tokenize
from train.preprocess import padding
from train.preprocess import full_process
from train.preprocess import ids_to_words


show_some_sample_and_some_statics(
        "/content/machine_translate/data/small_vocab_en",
        "/content/machine_translate/data/small_vocab_fr",
         4)

text = [
        "I love Nlp",
        "I am learning Nlp",
        "How are You",
    ]
    
#token_and_sequences
sequences, token = convert_text_to_tokenize(text)
dictonary = token.word_index
print("==========================")
print("index for each word")
print(dictonary)
print("==========================")

#padding
pad_sequences = padding(sequences)

for i, (sent, sequence, pad) in enumerate(zip(text, sequences, pad_sequences)):
  print("sample {}: {}".format(i+1, sent))
  print("token and sequence: {}".format(np.array(sequence)))
  print("padding sequences:  {}".format(pad))
  print("==========================")


print('Data Preprocessed')
en_input = read_data("/content/machine_translate/data/small_vocab_en")
fr_output = read_data("/content/machine_translate/data/small_vocab_fr")
length = None
type_pad = "post"

process_input, process_output, token_en, token_fr = full_process(en_input, fr_output, length=length, type_pad=type_pad)

    
maxlen_en_sequence = process_input.shape[1]
maxlen_fr_sequence = process_output.shape[1]
dict_en_size = len(token_en.word_index)
dict_fr_size = len(token_fr.word_index)

print("Max English sequence length after processing:", maxlen_en_sequence)
print("Max French sequence length: after processing:", maxlen_fr_sequence)


print("English vocabulary size:", dict_en_size)
print("French vocabulary size:", dict_fr_size)
print("==========================")


print(ids_to_words([[1, 2, 3, 4],
                    [2 , 1, 4, 0],
                    [10, 11, 12, 13],
                    [15, 16, 0, 0]],
                    token_fr))