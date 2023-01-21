from tqdm import tqdm 
import fileinput
import pickle

path_to_data = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54.txt"
words_path = "/home/sekhavat/projects/Nevise_Cleaned/all54_words.pkl"
chars_path = "/home/sekhavat/projects/Nevise_Cleaned/all54_chars.pkl"
words = set()
chars = set()
line_count = 0
for line in tqdm(fileinput.input([path_to_data])):
    line = line.replace("    "," ")
    line = line.replace("   "," ")
    line = line.replace("  "," ")
    splited = line.split(" ")
    words.update(splited)
    if line_count % 500000 == 0:
        print(f"{line_count} / 20303618")
    line_count +=1

for item in tqdm(words):
    chars.update(list(item))

with open(words_path, 'wb') as handle:
    pickle.dump(words, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(chars_path, 'wb') as handle:
    pickle.dump(chars, handle, protocol=pickle.HIGHEST_PROTOCOL)