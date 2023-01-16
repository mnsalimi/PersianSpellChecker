import time
import torch
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method("spawn", force=True)

class SpellChecker():
    
    def __init__(self, multiprocess_num=3):
        self.multiprocess_num = multiprocess_num
        self.model_name = 'HooshvareLab/bert-base-parsbert-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelWithLMHead.from_pretrained(self.model_name)
        # self.model_name = 'SajjadAyoubi/distil-bigbird-fa-zwnj'
        # self.fill = pipeline('fill-mask', model=self.model_name, tokenizer=self.model_name, device=-1)
        
    def get_suggested_token(self, sequence, target_token):
        input_ids = self.tokenizer.encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input_ids == self.tokenizer.mask_token_id)[1]

        token_logits = self.model(input_ids)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits, dim=1)
        top_k_list = []
        top_k = torch.topk(mask_token_logits, 50, dim=1)
        top_k_tokens = zip(top_k.indices[0].tolist(), top_k.values[0].tolist())
        for token, score in top_k_tokens:
            print(sequence + "\t\t\t" + self.tokenizer.decode([token]) + "\t\t\t" + str(score))
            top_k_list.append((self.tokenizer.decode([token]), score))
        top_k_dict = {x: y for x, y in top_k_list}
        top_k_token = [x for x, y in top_k_list]
        if target_token in top_k_dict and top_k_token.index(target_token) <= 20 or\
        target_token in top_k_dict and top_k_dict[target_token] >= 0.002:
            return target_token
        else:
            return top_k_list[0][0]
        
    def do_spellcheck(self, token_index):
        masked_str = [
            self.tokens[i] if i!=token_index else "[MASK]"
            for i in range(len(self.tokens))
        ]
        masked_str = ' '.join(masked_str)
        suggested_token = self.get_suggested_token(masked_str, self.tokens[token_index])
        return suggested_token

    def do_spellcheking_serially(self, query):
        self.tokens = query.split(" ")
        tokens_indexes = [i for i in range(len(self.tokens))]
        for i in range(len(self.tokens)):
            self.tokens[i] = self.do_spellcheck(i)
        return ' '.join(self.tokens)

    def do_spellcheking_parallelly(self, query):
        self.tokens = query.split(" ")
        tokens_indexes = [i for i in range(len(self.tokens))]
        multi_pool = Pool(processes=self.multiprocess_num)
        predictions = multi_pool.map(self.do_spellcheck, tokens_indexes)
        multi_pool.close()
        multi_pool.join()
        return ' '.join(predictions)


if __name__ == "__main__":

    spellchecker = SpellChecker(multiprocess_num=4)
    t1 = time.time()
    res = spellchecker.do_spellcheking_serially("پس از سال‌ها تلاش رازی موفق به کسف الکل شد. این دانشمند تیرانی باعث افتخار در تاریخ کور است")
    # res = spellchecker.do_spellcheking_parallelly("پایتخ ایران تهران است")
    print(time.time()-t1)
    print(res)