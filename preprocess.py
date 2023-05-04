import os
# from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

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
    
if __name__ == '__main__':

    show_some_sample_and_some_statics(
        "E:\\repos\\AIND-Capstone\\data\\small_vocab_en",
        "E:\\repos\\AIND-Capstone\data\small_vocab_fr",
         4)