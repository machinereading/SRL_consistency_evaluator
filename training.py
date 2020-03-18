
# coding: utf-8

# In[3]:


import json
import sys
sys.path.insert(0,'../')
from SRL_consistency_evaluator import dataio, converter
import argparse

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
import glob
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

from transformers import *
# from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
# from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from pprint import pprint
from datetime import datetime
start_time = datetime.now()


# In[3]:


parser = argparse.ArgumentParser()
parser.add_argument('--train', required=True, help='검증팀 데이터 폴더')
parser.add_argument('--model', required=False, help='모델이 저장되는 폴더', default='./model/')
parser.add_argument('--epoch', required=False, default=3)
parser.add_argument('--split', required=False, default=100)
parser.add_argument('--batch', required=False, default=3)
parser.add_argument('--n_split', required=False, default=False)
args = parser.parse_args()


# In[4]:


MAX_LEN = 256
batch_size = args.batch

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[3]:


def load_data(fdir, split=100, n_split=False):
    if n_split == False:
        ori_data = converter.load_data(fdir, split=int(split))
        tgt_data = dataio.data2tgt_data(ori_data)
    else:
        ori_data = converter.load_data(fdir, n_split=n_split)
        tgt_data = []
        for i in ori_data:
            d = dataio.data2tgt_data(i)
            tgt_data.append(d)
    
    return tgt_data


# In[5]:


class for_BERT():
    
    def __init__(self, mode='training'):
        self.mode = mode
        
        with open(dir_path+'/data/tag2idx.json','r') as f:
            self.tag2idx = json.load(f)
            
        self.idx2tag = dict(zip(self.tag2idx.values(),self.tag2idx.keys()))
        vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-frames'
        self.tokenizer = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=MAX_LEN)
        self.tokenizer.additional_special_tokens = ['<tgt>', '</tgt>']
        
        
    def idx2tag(self, predictions):
        pred_tags = [self.idx2tag[p_i] for p in predictions for p_i in p]
        
        # bert tokenizer and assign to the first token
    def bert_tokenizer(self, text):
        orig_tokens = text.split(' ')
        bert_tokens = []
        orig_to_tok_map = []
        bert_tokens.append("[CLS]")
        for orig_token in orig_tokens:
            orig_to_tok_map.append(len(bert_tokens))
            bert_tokens.extend(self.tokenizer.tokenize(orig_token))
        bert_tokens.append("[SEP]")

        return orig_tokens, bert_tokens, orig_to_tok_map
    
    def convert_to_bert_input(self, input_data):
        tokenized_texts, args = [],[]
        orig_tok_to_maps = []
        for i in range(len(input_data)):    
            data = input_data[i]
            text = ' '.join(data[0])
            orig_tokens, bert_tokens, orig_to_tok_map = self.bert_tokenizer(text)
            orig_tok_to_maps.append(orig_to_tok_map)
            tokenized_texts.append(bert_tokens)

            if self.mode == 'training':
                ori_args = data[2]
                arg_sequence = []
                for i in range(len(bert_tokens)):
                    if i in orig_to_tok_map:
                        idx = orig_to_tok_map.index(i)
                        ar = ori_args[idx]
                        arg_sequence.append(ar)
                    else:
                        arg_sequence.append('X')
                args.append(arg_sequence)

        input_ids = pad_sequences([self.tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
        orig_tok_to_maps = pad_sequences(orig_tok_to_maps, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post", value=-1)
        
        if self.mode =='training':
            arg_ids = pad_sequences([[self.tag2idx.get(ar) for ar in arg] for arg in args],
                                    maxlen=MAX_LEN, value=self.tag2idx["X"], padding="post",
                                    dtype="long", truncating="post")

        attention_masks = [[float(i>0) for i in ii] for ii in input_ids]    
        data_inputs = torch.tensor(input_ids)
        data_orig_tok_to_maps = torch.tensor(orig_tok_to_maps)
        data_masks = torch.tensor(attention_masks)
        
        if self.mode == 'training':
            data_args = torch.tensor(arg_ids)
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_args, data_masks)
        else:
            bert_inputs = TensorDataset(data_inputs, data_orig_tok_to_maps, data_masks)
        return bert_inputs


# In[6]:


def train(model_path='./model/', epochs=3, trn=False):
    print('your model would be saved at', model_path)
    
    bert_io = for_BERT(mode='training')
    
    model = BertForTokenClassification.from_pretrained("bert-base-multilingual-cased", num_labels=len(bert_io.tag2idx))
    model.to(device);
    
    trn_data = bert_io.convert_to_bert_input(trn)
    sampler = RandomSampler(trn_data)
    trn_dataloader = DataLoader(trn_data, sampler=sampler, batch_size=batch_size)
    
    # load optimizer
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters()) 
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
    optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
    
    
    # train 
    max_grad_norm = 1.0
    num_of_epoch = 0
    for _ in trange(epochs, desc="Epoch"):
        # TRAIN loop
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(trn_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_orig_tok_to_maps, b_input_args, b_input_masks = batch            
            # forward pass
            output = model(b_input_ids, token_type_ids=None,
                     attention_mask=b_input_masks, labels=b_input_args)
            loss = output[0]
            # backward pass
            loss.backward()
            # track train loss
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            model.zero_grad()
#             break
#         break

        # print train loss per epoch
        print("Train loss: {}".format(tr_loss/nb_tr_steps))
        num_of_epoch += 1
    model.save_pretrained(model_path)
    print('...training is done')
    print('your model is saved to', model_path)
    print('###################################################\n')


# In[7]:


def main(args):
    fdir = args.train
    model_path = args.model
    epochs = args.epoch    
    n_split = args.n_split
    
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if model_path[-1] != '/':
        model_path = model_path+'/'
    
    if n_split == False:
        split = args.split
        trn = load_data(fdir, split=split)
        train(model_path, epochs=int(epochs), trn=trn)
    else:
        trns = load_data(fdir, n_split=int(n_split))
        for i in range(len(trns)):
            trn = trns[i]
            model_path_split = model_path+'split_'+str(i)+'/'
            if not os.path.exists(model_path_split):
                os.makedirs(model_path_split)
            train(model_path_split, epochs=int(epochs), trn=trn)
            


# In[8]:


if __name__ == "__main__":
    main(args)

