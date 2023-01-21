import pickle

from tqdm import tqdm 
import fileinput
import os


def load_vocab_dict(path_: str):
    with open(path_, 'rb') as fp:
        vocab = pickle.load(fp)
    return vocab
    
def save_vocab_dict(path, data):
    with open(path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    
def make_vocab_file(
                    path_to_main_vocab_file,  
                    path_to_words,
                    path_to_save_new_vocab_file,
                    words,
                    chars
                    ):
                     
    if os.path.isfile(path_to_save_new_vocab_file):
        main_vocab = load_vocab_dict(path_to_save_new_vocab_file)
        print("Loading a new vocab file ...")
    else:
        main_vocab = load_vocab_dict(path_to_main_vocab_file)
        print("Loading the main vocab file ...")
    
    # new_vocab_dict = main_vocab.copy()
    new_vocab_dict = {}
    
    new_vocab_dict['pad_token'] = "<<PAD>>"
    new_vocab_dict['pad_token_idx'] = 0

    new_vocab_dict['unk_token'] = "<<UNK>>"
    new_vocab_dict['unk_token_idx'] = 1


    new_vocab_dict['eos_token'] = "<<EOS>>"
    new_vocab_dict['eos_token_idx'] = 2
    
    
    new_vocab_dict['char_unk_token'] = "<<CHAR_UNK>>"
    new_vocab_dict['char_pad_token'] = "<<CHAR_PAD>>"
    new_vocab_dict['char_start_token'] = "<<CHAR_START>>"
    new_vocab_dict['char_end_token'] = "<<CHAR_END>>"
    
    new_vocab_dict['char_unk_token_idx'] = 0
    new_vocab_dict['char_pad_token_idx'] = 1
    new_vocab_dict['char_start_token_idx'] = 2
    new_vocab_dict['char_end_token_idx'] = 3
    
    
    
    new_vocab_dict['token2idx'] = {
       "<<PAD>>": 0,
       "<<UNK>>": 1,
       "<<EOS>>": 2,
    }
    
    
    
    new_vocab_dict['idx2token'] = {
      0: "<<PAD>>",
      1: "<<UNK>>",
      2: "<<EOS>>",
    }
    
    
    new_vocab_dict['chartoken2idx'] = {
      "<<CHAR_UNK>>": 0,
      "<<CHAR_PAD>>": 1,
      "<<CHAR_START>>": 2,
      "<<CHAR_END>>": 3,
    }
    
    new_vocab_dict['idx2chartoken'] = {
      0: "<<CHAR_UNK>>",
      1: "<<CHAR_PAD>>",
      2: "<<CHAR_START>>",
      3: "<<CHAR_END>>",
    }
    
    """
    print("new_vocab_dict['pad_token']", new_vocab_dict['pad_token'])
    print("new_vocab_dict['unk_token']", new_vocab_dict['unk_token'])
    print("new_vocab_dict['eos_token']", new_vocab_dict['eos_token'])
    print("new_vocab_dict['pad_token_idx']", new_vocab_dict['pad_token_idx'])
    print("new_vocab_dict['unk_token_idx']", new_vocab_dict['unk_token_idx'])
    print("new_vocab_dict['eos_token_idx']", new_vocab_dict['eos_token_idx'])
    #print("new_vocab_dict['chartoken2idx']", new_vocab_dict['chartoken2idx'])
    #print("new_vocab_dict['idx2chartoken']", new_vocab_dict['idx2chartoken'])
    print("new_vocab_dict['char_unk_token']", new_vocab_dict['char_unk_token'])
    print("new_vocab_dict['char_pad_token']", new_vocab_dict['char_pad_token'])
    print("new_vocab_dict['char_start_token']", new_vocab_dict['char_start_token'])
    print("new_vocab_dict['char_end_token']", new_vocab_dict['char_end_token'])
    print("new_vocab_dict['char_unk_token_idx']", new_vocab_dict['char_unk_token_idx'])
    print("new_vocab_dict['char_pad_token_idx']", new_vocab_dict['char_pad_token_idx'])
    print("new_vocab_dict['char_start_token_idx']", new_vocab_dict['char_start_token_idx'])
    print("new_vocab_dict['char_end_token_idx']", new_vocab_dict['char_end_token_idx'])
    """
    
    counter = 1
    print(new_vocab_dict["token2idx"].values())
    for word in tqdm(words):
        word = word.replace("\n", "").strip()
        # added_words = list(new_vocab_dict["token2idx"].keys())
        # if not(word in added_words):
        if word not in new_vocab_dict["token2idx"]:
            # print("Not skipping ...")
            index = max(list(new_vocab_dict["token2idx"].values())) + 1
            new_vocab_dict["token2idx"][word] = index
            new_vocab_dict["idx2token"][index] = word
            
        
            new_vocab_dict["token_freq"] = len(new_vocab_dict["token2idx"].keys())
            if counter % 1000 == 0:
                save_vocab_dict(path_to_save_new_vocab_file, new_vocab_dict)
    
            
        counter += 1
    for char in tqdm(chars):
        # if not(char in list(new_vocab_dict['chartoken2idx'].keys())):
        if char not in new_vocab_dict['chartoken2idx']:
            char_index = max(list(new_vocab_dict["chartoken2idx"].values())) + 1
            new_vocab_dict['chartoken2idx'][char] = char_index 
            new_vocab_dict['idx2chartoken'][char_index] = char
    
    save_vocab_dict(path_to_save_new_vocab_file, new_vocab_dict)

if __name__ == "__main__":
    # path_to_words = "/home/sekhavat/projects/Nevise_Cleaned/DATA/WORDS.txt"
    path_to_words = "/home/sekhavat/projects/Nevise_Cleaned/all54_words.pkl"
    path_to_chars = "/home/sekhavat/projects/Nevise_Cleaned/all54_chars.pkl"
    path_to_save_new_vocab_file = "/home/sekhavat/projects/Nevise_Cleaned/all54_vocab.pkl"
    
    path_to_main_vocab_file = "/home/sekhavat/projects/Nevise_Cleaned/vocab.pkl"

    with open(path_to_words, 'rb') as handle:
        words = pickle.load(handle)
    with open(path_to_chars, 'rb') as handle:
        chars = pickle.load(handle)
    
    make_vocab_file(
        path_to_main_vocab_file,  
        path_to_words,
        path_to_save_new_vocab_file,
        words,
        chars
        )