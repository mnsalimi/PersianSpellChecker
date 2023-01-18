import time
import torch

from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, BertForMaskedLM
from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead
from torch.multiprocessing import Pool, Process, set_start_method
set_start_method("spawn", force=True)

class SpellChecker():
    
    def __init__(
        self, multiprocess_num=3, edit_distance=False, parsbert=False,
        mbert=False, persian_bigbird=False, topk=50,
    ):
        self.topk = topk
        self.edit_distance = edit_distance
        self.multiprocess_num = multiprocess_num
        
        self.model_name = 'HooshvareLab/bert-base-parsbert-uncased'
        self.parsbert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.parsbert_model = AutoModelWithLMHead.from_pretrained(self.model_name)

        self.model_name = 'bert-base-multilingual-cased'
        self.mbert_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.mbert_model = AutoModelWithLMHead.from_pretrained(self.model_name)
        # self.model_name = 'SajjadAyoubi/distil-bigbird-fa-zwnj'
        # self.fill = pipeline('fill-mask', model=self.model_name, tokenizer=self.model_name, device=-1)
        self.models = {}
        if mbert:
            self.models["mbert"] = {
                "model": self.mbert_model,
                "tokenizer": self.mbert_tokenizer,
            }
        if parsbert:
            self.models["parsbert"] = {
                "model": self.parsbert_model,
                "tokenizer": self.parsbert_tokenizer,
            }
            
    def get_editdistance_suggestions(self, tokens, misspelled_token):
        similarities = []
        for token in tokens:
            similarities.append(
                fuzz.ratio(token, misspelled_token)
            )
        return tokens[
            similarities.index(max(similarities))
        ]

    def get_languagemodel_suggestions(self, model, sequence, target_token):
        input_ids = model["tokenizer"].encode(sequence, return_tensors="pt")
        mask_token_index = torch.where(input_ids == model["tokenizer"].mask_token_id)[1]

        token_logits = model["model"](input_ids)[0]
        mask_token_logits = token_logits[0, mask_token_index, :]
        mask_token_logits = torch.softmax(mask_token_logits, dim=1)
        top_k_list = []
        top_k = torch.topk(mask_token_logits, self.topk, dim=1)
        top_k_tokens = zip(top_k.indices[0].tolist(), top_k.values[0].tolist())
        for token, score in top_k_tokens:
            print(sequence, model["tokenizer"].decode([token]), f"(score: {score})")
            top_k_list.append((model["tokenizer"].decode([token]), score))

        sought_after_token_id = model["tokenizer"].encode(target_token, add_special_tokens=False,)[0]  # 928
        token_score = mask_token_logits[:, sought_after_token_id]
        target_token_score = mask_token_logits[:, sought_after_token_id][0]
        print(f"Score of {target_token}: {mask_token_logits[:, sought_after_token_id]}")
        top_k_dict = {x: y for x, y in top_k_list}
        top_k_token = [x for x, y in top_k_list]
        if target_token in top_k_dict and top_k_token.index(target_token) <= self.topk or\
        target_token in top_k_dict and top_k_dict[target_token] >= 0.002 or\
        target_token_score >= 0.0001:
            return [target_token]
        else:
            return top_k_token[:self.topk]
        
    def do_spellcheck(self, token_index):
        masked_str = [
            self.tokens[i] if i!=token_index else "[MASK]"
            for i in range(len(self.tokens))
        ]
        masked_str = ' '.join(masked_str)
        suggestions = []
        for model in self.models:
            suggestions +=\
                self.get_languagemodel_suggestions(self.models[model], masked_str, self.tokens[token_index])
        if self.edit_distance:
            res = self.get_editdistance_suggestions(suggestions, self.tokens[token_index])
            print("suggestions for editdistance", suggestions)
            print("editdistance res", res)
            return res

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

def test_cases(input_address, output_address):
    spellchecker = SpellChecker(
        multiprocess_num=4,
        topk=25,
        parsbert=True,
        # mbert=True,
        edit_distance=True,
    )
    with open(input_address, "r") as f:
        lines = [line.replace("\n", "").rstrip().lstrip().strip() for line in f.readlines()]
        lines = [line.split(",") for line in lines]
        lines = [[line[0], line[1]] for line in lines]
    lines = [
        ("کنکور سراسری از ۵ سال دیگر حرف خواهد شد!", "کنکور سراسری از ۵ سال دیگر حذف خواهد شد!"),
        ("بحران بی آتی در بسیاری ار شهرهای ایران", "بحران بی آتی در بسیاری ار شهرهای ایران")
        # ("بارش برق و باران در سراسر کشور تا آخر هفته ادامه دارد", "بارش برق و باران در سراسر کشور تا آخر هفته ادامه دارد"),
    ]
    # [print(line) for line in lines[1:]]
    for line in lines[1:]:
        print("query: {}".format(line[0]))
        t1 = time.time()
        res = spellchecker.do_spellcheking_serially(line[0])
        # res = spellchecker.do_spellcheking_parallelly("پایتخ ایران تهران است")
        print(time.time()-t1)
        print("res: {}".format(res))
        print()

if __name__ == "__main__":
    test_cases("testcases.csv", "res.csv")