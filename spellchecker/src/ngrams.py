from os import path
from nltk import ngrams
from time import time
import pickle

class FiveGram:

    def __init__(self, data_address=None):
        self.train = open(path.join(data_address, "train.txt"), mode="r", encoding="utf-8").readlines()
        self.train = [line.strip().lstrip().rstrip() for line in self.train]
        # self.test = open(path.join(data_address, "test.txt"))
        # self.freqs = {}
        self.paddings_start = ["s1", "s2", "s3",]
        self.paddings_end = ["e1", "e2", "e3",]
        # self.vocabs = []
        # for line in self.train:
        #     line = line.split(" ")
        #     for tkn in line:
        #         if tkn not in self.vocabs:
        #             self.vocabs.append(tkn)
        # self.vocabs += self.paddings_start + self.paddings_end
        # self.freqs += self.paddings_start + self.paddings_end
        # with open("ngram_models/vocabs.pickle",'rb') as f:
        #     self.vocabs = pickle.load(f)
        # with open("ngram_models/frequencies.pickle",'rb') as f:
        #     self.freqs = pickle.load(f)
    
    def create_ngrams(self, tokens, n):
        # print(len(tokens))
        # print(tokens)
        ngrams = []
        for i in range(len(tokens)+n):
            temp_ngram = []
            pad_size = 0 if n-i-1<0 else n-i-1
            # print("pad_size:", str(pad_size))
            # print("i: ", str(i))
            if pad_size > 0:
                temp_ngram[0:0] = self.paddings_start[:pad_size]
                # print("temp_ngram: ", str(temp_ngram))
                # print("i+(n-pad_size):", str(i+(n-pad_size)))
                temp_ngram += tokens[0:n-len(temp_ngram)]
                # print("temp_ngram final:", str(temp_ngram))
            elif i < len(tokens) - pad_size+1:
                temp_ngram += tokens[i-n:i]
                # print("temp_ngram final:", str(temp_ngram))
            else:
                temp_ngram += tokens[i-n:]
                temp_ngram += self.paddings_end[:n-len(temp_ngram)]
                # print("temp_ngram final:", str(temp_ngram))
            ngrams.append(temp_ngram)
        return ngrams


    def create_ngrams_tokens(self):
        # self.fivegrams = []
        for line in self.train:
            line = line.split()
            # line[0:0] = self.paddings_start
            # line[len(line):len(line)] = self.paddings_end
            fourgrams = self.create_ngrams(line, 4)
            fivegrams = self.create_ngrams(line, 5)
            # print(fourgrams)
            for i in range(len(fivegrams)):
                # print(i)
                self.freqs[' '.join(fivegrams[i])] = self.freqs.get(' '.join(fivegrams[i]), 0) + 1

            for fourgram in fourgrams:
                self.freqs[' '.join(fourgram)] = self.freqs.get(' '.join(fourgram), 0) + 1
        with open("ngram_models/frequencies.pickle", "wb") as f:
            pickle.dump(self.freqs, f)
        with open("ngram_models/vocabs.pickle", "wb") as f:
            pickle.dump(self.vocabs, f)

    def predict(self, tokens, index):
        pad_size = 0 if index>4 else 4 - index
        # print("pad_size:", str(pad_size))
        index += pad_size
        tokens[0:0] = self.paddings_start[:pad_size]
        # print("tokens: ", str(tokens))
        # print("index: ", str(index))
        probs = {}
        makhraj_str = ' '.join(tokens[index-4:index])
        # print("makhraj_str: ", makhraj_str)
        # print("makhraj_str_val:", str(self.freqs.get(makhraj_str, 0)))
        cc = 0
        for token in self.vocabs:
            soorat_str = ' '.join(tokens[index-4:index])+" "+token
            # print("soorat_str:", str(soorat_str))
            # print("soorat_str_val:", str(self.freqs.get(soorat_str, 0)))
            val = float(
                (1+self.freqs.get(soorat_str, 0))
                /
                (len(self.vocabs)+self.freqs.get(makhraj_str, 0))
            )
            # print(val)
            # exit()
            probs[' '.join(tokens[:index]+[token])] = val
            cc += 1
            # if cc == 5:
            #     exit()
        re = [max(probs, key=probs.get).split()[-1]]
        return tokens[pad_size:index] + re + tokens[index+1:]

# t1 = time()
# text = "سال این"
# fivegram = FiveGram()
# t2 = time()
# print(t2-t1)
# # fivegram.create_ngrams_tokens()
# t3 = time()
# # print(t3-t2)
# x = fivegram.predict(text.split(), 0)
# t4 = time()
# print(x)
# print(t4-t3)
