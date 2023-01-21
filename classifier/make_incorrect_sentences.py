import random
from farsi_char_maps import char_maps
from tqdm import tqdm

# path_to_correct_sentences_file = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54.txt"
# path_to_incorrect_sentences_file = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54_INCORRECT.txt"
# path_to_manipulation_GT = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54_MANIPULATION_GT.txt"
path_to_correct_sentences_file = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_normal_corrects4.txt"
path_to_incorrect_sentences_file = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_INCORRECT4.txt"
path_to_manipulation_GT = "/home/sekhavat/projects/Nevise_Cleaned/DATA/infer/nevise-news-451_MANIPULATION_GT4.txt"

def swap(word, i, j):
    word = list(word)
    word[i], word[j] = word[j], word[i]
    
    return ''.join(word)
    
with open(path_to_correct_sentences_file) as h:
    correct_sentences = h.readlines()

for cs in tqdm(correct_sentences):
    new_sentence = []
    cs = cs.replace("\n", "")
    has_word_been_changed = False

    for w in cs.split(" "):
        original_word = w
        random_state = random.randint(0, 10)

        if random_state <= 2:
            if w.__contains__(char_maps["RE"]):
                _ = random.randint(1, 10)
                if  _>= 8:    
                    w = w.replace(char_maps["RE"], char_maps["ZA"])
                elif _ >= 6:    
                    w = w.replace(char_maps["RE"], char_maps["ZAD"])
                else:    
                    w = w.replace(char_maps["RE"], char_maps["ZE"])
                
                new_sentence.append(w)
                continue
            
            if w.__contains__(char_maps["ZE"]):
                _ = random.randint(1, 10)
                if  _>= 8:    
                    w = w.replace(char_maps["ZE"], char_maps["ZA"])
                elif _ >= 6:    
                    w = w.replace(char_maps["ZE"], char_maps["ZAD"])
                else:    
                    w = w.replace(char_maps["ZE"], char_maps["RE"])
                    
                new_sentence.append(w)
                continue
            

            if w.__contains__(char_maps["SIN"]):
                _ = random.randint(1, 10)
                if _ == 10:
                    w = w.replace(char_maps["SIN"], char_maps["SE"])
                elif  _>= 7:    
                    w = w.replace(char_maps["SIN"], char_maps["SAD"])
                else:    
                    w = w.replace(char_maps["SIN"], char_maps["SHIN"])
                
                new_sentence.append(w)
                continue
            
            if w.__contains__(char_maps["SHIN"]):
                _ = random.randint(1, 10)
                if  _ >= 7:    
                    w = w.replace(char_maps["SHIN"], char_maps["ZAD"])
                else:    
                    w = w.replace(char_maps["SHIN"], char_maps["SIN"])
                
                new_sentence.append(w)
                continue
            
            if w.__contains__(char_maps["JIM"]):
                _ = random.randint(1, 10)
                if _ == 10:
                    w = w.replace(char_maps["JIM"], char_maps["CHE"])
                elif  _>= 7:    
                    w = w.replace(char_maps["JIM"], char_maps["HE"])    
                else:    
                    w = w.replace(char_maps["JIM"], char_maps["KHE"])
                    new_sentence.append(w)
                    continue
            
            if w.__contains__(char_maps["KHE"]):
                _ = random.randint(1, 10)
                if _ == 10:
                    w = w.replace(char_maps["KHE"], char_maps["CHE"])
                elif  _>= 7:    
                    w = w.replace(char_maps["KHE"], char_maps["HE"])
                else:    
                    w = w.replace(char_maps["KHE"], char_maps["JIM"])
                    new_sentence.append(w)
                    continue

            if w.__contains__(char_maps["HE"]):
                _ = random.randint(1, 10)
                if _ == 10:
                    w = w.replace(char_maps["HE"], char_maps["CHE"])
                elif  _>= 7:    
                    w = w.replace(char_maps["HE"], char_maps["KHE"])
                else:    
                    w = w.replace(char_maps["HE"], char_maps["JIM"])
                    new_sentence.append(w)
                    continue

            has_word_been_changed = False
            new_sentence.append(w)

        elif 2 <= random_state <= 3:
            try:
                random_char_index = random.randint(1, len(w)-2)
            except:
                new_sentence.append(w)
                continue
            if random.randint(1, 10) >= 5:
                w = swap(w, random_char_index, random_char_index - 1)
            else:
                w = swap(w, random_char_index, random_char_index + 1)
            
            new_sentence.append(w)
            
        elif random_state == 4:
            try:
                _ = random.randint(0, len(w) - 1)
            except:
                new_sentence.append(w)
                continue
            if _ == 0:
                w = w[0] + w
            elif _ == len(w):
                w = w + w[-1]
            else:
                new_w = w[0: _] + w[_] + w[_ + 1: len(w)-1]
                if len(new_w) != 0:
                    w = new_w
                else:
                    print("was zero1")
            
            new_sentence.append(w)
                      
        elif random_state == 5:
            try:
                new_w2 = w.replace(w[random.randint(0, len(w) - 1)], "")
                if len(new_w2) != 0:
                    w = new_w2
                else:
                    print("was zero2")
                new_sentence.append(w)
            except:
                new_sentence.append(w)
        else:
            new_sentence.append(w)    

        with open("./GTs_for_corr_incorr2.txt", "a+") as h:
            h.seek(0)
            if has_word_been_changed:
                h.writelines("1")
            else:
                h.writelines("0 ")


    if len(cs.split(" ")) != len((" ".join(new_sentence)).split(" ")):
        print("ERROR - LENGTH DOESNT MATCH")
        print(cs.split(" "))
        print((" ".join(new_sentence)).split(" "))
        print(len(cs.split(" ")))
        print(len((" ".join(new_sentence)).split(" ")))
        
    new_sentence.append("\n")

    with open(path_to_incorrect_sentences_file, "a+") as h:
        h.seek(0)
        h.writelines(" ".join(new_sentence))


    with open(path_to_manipulation_GT, "a+") as h:
        h.seek(0)

        for old_word, new_word in zip(cs.split(" "), new_sentence):
            if old_word == new_word:
                h.writelines("0 ")
            else:
                h.writelines("1 ")

        h.writelines("\n")