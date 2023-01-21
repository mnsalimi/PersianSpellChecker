from fanap_normalizer.src.normalizer import Normalizer
from tqdm import tqdm
import re

nr = Normalizer(
    clear_triple_chars_needed=True,
    clear_ye_after_halfspace=True,
    standard_numbers = True,
    # separate_nonpersian_needed = True
    )

print("Normalizer Loaded")


path_to_txt_1 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__farsnews1.txt"
path_to_txt_2 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__fararu.txt"
path_to_txt_3 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__irna.txt"
path_to_txt_4 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__kafebook.txt"
path_to_txt_5 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__psarena.txt"
path_to_txt_6 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__taghcheh.txt"
path_to_txt_7 = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/SENTENCES__vigiato.txt"

path_list = [path_to_txt_3, path_to_txt_4, path_to_txt_5, path_to_txt_6, path_to_txt_7, path_to_txt_2] #path_to_txt_1,

path_to_save = f"/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54.txt"

unique = set()

repeat = 0
line_count = 0

for p in path_list:
    print(p)
    print("line_count: ", line_count)
    with open(p) as h:
        sentences = h.readlines()
    # print(len(sentences))
    for s in tqdm(sentences):
        # set line count limit
        if line_count > 1600000:
            continue

        s = s.replace("\n", "")
        
        #remove non-persian and numbers
        s = re.sub(r'[^\u0600-\u06FF .-]'," ", s)
        s = re.sub(r'[۱۲۳۴۵۶۷۸۹۰]'," ", s).strip()
        s = re.sub(r'[١٢٣٤٥٦٧٨٩٠]'," ", s).strip()

        #normalize
        if len(s) < 1000:
            s = nr.normalize(s)
        else:
            parts = []
            for part in s.split("."):
                parts.append(nr.normalize(part))
            s = ".".join(parts)
        
        # remove half space
        s = s.replace("‌"," ")

        #remove single chars
        s = ' '.join([w for w in s.split() if len(w)>1])

        #ignore short and useless sentences
        if len(set(s)) < 5:
            continue

        if " رئی " in s:
            continue
            print(s)

        if s in unique:
            repeat +=1
            # print("repeat : " , repeat)
            # print(s)
        else:
            unique.add(s)
            line_count += 1
            with open(path_to_save, "a+") as wr:        
                wr.seek(0)
                wr.writelines(s)
                wr.writelines("\n")
print("repeat : " , repeat)