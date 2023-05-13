import numpy as np
from preprocess import read_data
from preprocess import show_some_sample_and_some_statics
from preprocess import convert_text_to_tokenize
from preprocess import padding
from preprocess import full_process

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


en_input = read_data("/content/machine_translate/data/small_vocab_en")
fr_output = read_data("/content/machine_translate/data/small_vocab_fr")
length = None
type_pad = "post"
process_input, process_output, token_en, token_fr = full_process(en_input, fr_output, length=length, type_pad=type_pad)
    
maxlen_en_sequence = process_input.shape[1]
maxlen_fr_sequence = process_output.shape[1]
dict_en_size = len(token_en.word_index)
dict_fr_size = len(token_fr.word_index)

print('Data Preprocessed')
print("Max English sequence length:", maxlen_en_sequence)
print("Max French sequence length:", maxlen_fr_sequence)
print("English vocabulary size:", dict_en_size)
print("French vocabulary size:", dict_fr_size)
print("==========================")
