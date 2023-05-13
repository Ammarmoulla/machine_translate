import numpy as np
from preprocess import show_some_sample_and_some_statics
from preprocess import convert_text_to_tokenize
from preprocess import padding

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