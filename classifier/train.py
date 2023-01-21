import os
from tqdm import tqdm
import re
import time
import torch
import utils
from helpers import *

from hazm import Normalizer
from models import SubwordBert
from utils import get_sentences_splitters



os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_data():
    print("get data")
    # path_to_text_file_gt = "/home/rmarefat/projects/PSC/Nevise_Cleaned/DATA/TXTs_Sentences/FARSI_SENTENCES__irna__kafebook__psarena__taghcheh__vigiato.txt"
    # path_to_text_file_incorrect = "/home/rmarefat/projects/PSC/Nevise_Cleaned/DATA/TXTs_Sentences/INCORRECT_FARSI_SENTENCES__irna__kafebook__psarena__taghcheh__vigiato.txt"
    path_to_text_file_gt = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54.txt"
    path_to_text_file_incorrect = "/home/sekhavat/projects/Nevise_Cleaned/DATA/TXTs_Sentences/sekhavat3/FARSI_ALL54_INCORRECT.txt"

    with open(path_to_text_file_gt) as h:
      corrects = h.readlines()
    with open(path_to_text_file_incorrect) as h:
      wrongs = h.readlines()
    
    
    print(f"Number of training sentences: {len(corrects)}")
    
    return wrongs, corrects

def load_model(vocab_path, device, load_pretrained=False,checkpoint_path=None):
    print(f"loading vocab from {vocab_path}")
    
    vocab = load_vocab_dict(vocab_path)

    print("Subword Bert...")
    model = SubwordBert(3*len(vocab["chartoken2idx"]), vocab["token2idx"][ vocab["pad_token"] ], vocab['token_freq']) # 255534 the last number is the number of the words in unique mode
    print("Subword Bert Done")
    # get_model_nparams(model)
    
    if load_pretrained:
      if torch.cuda.is_available() and device != "cpu":
        map_location = lambda storage, loc: storage.cuda()
        print("loading from GPU")
      else:
        map_location = 'cpu'
        print("loading from CPU")
      print(f"Loading model params from checkpoint dir: {checkpoint_path}")
      # checkpoint_data = torch.load(checkpoint_path, map_location=map_location)
      # model.load_state_dict(checkpoint_data['model_state_dict'])
      model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
      print("Checkpoint loaded")
      
      """
      if optimizer is not None:
          optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
      max_dev_acc, argmax_dev_acc = checkpoint_data["max_dev_acc"], checkpoint_data["argmax_dev_acc"]
  
      if optimizer is not None:
          return model, optimizer, max_dev_acc, argmax_dev_acc
      """
    
    return model, vocab
    

def train(model, data, targets, topk, DEVICE, BATCH_SIZE, vocab, EPOCHS, optimizer):
    #if vocab_ is not None:
    #    vocab = vocab_
    
    _corr2corr, _corr2incorr, _incorr2corr, _incorr2incorr = 0, 0, 0, 0
    _mistakes = []
    

    print(f"data size: {len(data)}")
    
    input_data = []
    for t, d in zip(data, targets):
      input_data.append((d[0], t[0]))
    
    # data_iter = batch_iter(input_data, batch_size=BATCH_SIZE, shuffle=True)
    data_loader = get_dataloader(input_data, BATCH_SIZE)
    
    model.to(DEVICE)

    results = []
    line_index = 0
    
    for epoch in range(EPOCHS):
      running_loss = 0.
      print(f"Epoch: {epoch + 1}")
      
      # Training
      model.train()
      
      for batch_id, (batch_labels, batch_sentences) in tqdm(enumerate(data_loader)):
          #torch.cuda.empty_cache()
          
          batch_labels, batch_sentences, batch_bert_inp, batch_bert_splits = bert_tokenize_for_valid_examples(batch_labels, batch_sentences)
          
          try:
            batch_bert_inp = {k: v.to(DEVICE) for k, v in batch_bert_inp.items()}
          except Exception as e:
            #print(e)
            continue
          
          
          batch_labels_ids, batch_lengths = labelize(batch_labels, vocab)
          
          batch_lengths = batch_lengths.to(DEVICE)
          batch_labels_ids = batch_labels_ids.to(DEVICE)
          
          if int(batch_id) % 10000 == 0:
            PATH = f"/home/sekhavat/projects/Nevise_Cleaned/sekhavat_trains/{epoch}_{batch_id}.pth"
            torch.save(model.state_dict(), PATH)
            print(f"\nEpoch {epoch + 1} iter: {batch_id}")

          #batch_loss, batch_predictions = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk)
          loss = model(batch_bert_inp, batch_bert_splits, targets=batch_labels_ids, topk=topk)
          
          running_loss += loss.item()
          
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          
      PATH = f"/home/sekhavat/projects/Nevise_Cleaned/sekhavat_trains/{epoch}.pth"
      """      
      torch.save({
          #'epoch': epochs,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'train_loss_history': loss_history,
          }, PATH)
      """ 
      torch.save(model.state_dict(), PATH)
      print(f"\nEpoch {epoch + 1} loss: {loss / (batch_id + 1)}")

    
def run_training():
    DEVICE='cuda'
    # DEVICE='cpu'
    BATCH_SIZE=8
    # BATCH_SIZE=1
    vocab_=None
    EPOCHS=100
    # vocab_path = "/home/rmarefat/projects/PSC/Nevise_Cleaned/vocab.pkl"
    vocab_path = "/home/sekhavat/projects/Nevise_Cleaned/all54_vocab.pkl"
    #vocab_path = "/home/rmarefat/Desktop/PSC/Nevise/main_vocabd.pkl"
    # model_checkpoint_path = "/home/marefat/projects/PSC/Nevise/model.pth.tar"
    # checkpoint_path = "/home/sekhavat/projects/Nevise_Cleaned/sekhavat_trains/0_new.pth"
    checkpoint_path = None
    topk = 1

    data, targets = get_data()
    
    data_sents, splitters = get_sentences_splitters(data)
    target_sents, splitters = get_sentences_splitters(targets)
    
    print("normalizer...")
    # normalizer = Normalizer(punctuation_spacing=False, remove_extra_spaces=False)
    print("normalizer done")
    
    print("Data sents...")
    data_sents = [utils.space_special_chars(s) for s in data_sents[0]]
    data_sents = list(filter(lambda txt: (txt != '' and txt != ' '), data_sents))
    print("Data sents done")
    
    print("Target sents..")
    target_sents = [utils.space_special_chars(s) for s in target_sents[0]]
    target_sents = list(filter(lambda txt: (txt != '' and txt != ' '), target_sents))
    print("Target sents done")
    
    print("data target sents...")
    # data_sents = [(normalizer.normalize(t), normalizer.normalize(t)) for t in data_sents]
    # target_sents = [(normalizer.normalize(t), normalizer.normalize(t)) for t in target_sents]
    data_sents = [((t), (t)) for t in data_sents]
    target_sents = [((t), (t)) for t in target_sents]
    print("data target sents done")
    
    print("loading model...")
    model, vocab = load_model(vocab_path=vocab_path, device=DEVICE, load_pretrained=False,checkpoint_path=checkpoint_path)#load_pretrained is FALSE
    print("model loaded")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
    
    train(model, data_sents, target_sents, topk, DEVICE, BATCH_SIZE, vocab, EPOCHS, optimizer)
    

if __name__ == '__main__':
    import time 
        
    run_training()
    