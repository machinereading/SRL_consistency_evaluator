
# coding: utf-8

# In[1]:


import json
import sys
sys.path.insert(0,'../')
from BERT_for_Korean_SRL import dataio

import numpy as np

import torch
from torch import nn
from torch.optim import Adam
import glob
import os
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from pprint import pprint
from datetime import datetime
start_time = datetime.now()


# In[2]:


MAX_LEN = 256
batch_size = 6

try:
    dir_path = os.path.dirname(os.path.abspath( __file__ ))
except:
    dir_path = '.'


# In[3]:


def load_data(fname):
    with open(fname, 'r') as f:
        d = f.readlines()
        
    ori_data = dataio.conll2tagseq(d)
    tgt_data = dataio.data2tgt_data(ori_data)
    
    return tgt_data


# In[4]:


class for_BERT():
    
    def __init__(self, mode='training'):
        self.mode = mode
        
        with open(dir_path+'/data/tag2idx.json','r') as f:
            self.tag2idx = json.load(f)
            
        self.idx2tag = dict(zip(self.tag2idx.values(),self.tag2idx.keys()))
        
        # load pretrained BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
        
        # load BERT tokenizer with untokenizing frames
        never_split_tuple = ("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")
        added_never_split = []
        added_never_split.append('<tgt>')
        added_never_split.append('</tgt>')
        added_never_split_tuple = tuple(added_never_split)
        never_split_tuple += added_never_split_tuple
        vocab_file_path = dir_path+'/data/bert-multilingual-cased-dict-add-frames'
        self.tokenizer_with_frame = BertTokenizer(vocab_file_path, do_lower_case=False, max_len=256, never_split=never_split_tuple)
        
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
            bert_tokens.extend(self.tokenizer_with_frame.tokenize(orig_token))
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


# In[5]:


bert_io = for_BERT(mode='training')


# In[6]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test(model_fname, tst):
    model_path = '/disk/data/models/kosrl_1105/'
    models = glob.glob(model_path+'*.pt')
    
    result_path = model_path = '/disk/data/models/result_kosrl_1105/'
    results = []
    

    model = torch.load(model_fname)
    model.eval()

    tst_data = bert_io.convert_to_bert_input(tst)
    sampler = RandomSampler(tst_data)
    tst_dataloader = DataLoader(tst_data, sampler=sampler, batch_size=batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    pred_args, true_args = [],[]
    for batch in tst_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_orig_tok_to_maps, b_input_args, b_input_masks = batch

        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                          attention_mask=b_input_masks, labels=b_input_args)
            logits = model(b_input_ids, token_type_ids=None,
                           attention_mask=b_input_masks)

        logits = logits.detach().cpu().numpy()

        b_pred_args = [list(p) for p in np.argmax(logits, axis=2)]
        b_true_args = b_input_args.to('cpu').numpy().tolist()


        eval_loss += tmp_eval_loss.mean().item()

        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1

        for b_idx in range(len(b_true_args)):

            input_id = b_input_ids[b_idx]
            orig_tok_to_map = b_input_orig_tok_to_maps[b_idx]                
            pred_arg_bert = b_pred_args[b_idx]
            true_arg_bert = b_true_args[b_idx]

            pred_arg, true_arg = [],[]
            
            try:
                for tok_idx in orig_tok_to_map:
                    if tok_idx != -1:
                        tok_id = int(input_id[tok_idx])
                        if tok_id == 1:
                            pass
                        elif tok_id == 2:
                            pass
                        else:
                            pred_arg.append(pred_arg_bert[tok_idx])
                            true_arg.append(true_arg_bert[tok_idx])

                pred_args.append(pred_arg)
                true_args.append(true_arg)
            except KeyboardInterrupt:
                raise
            except:
                pass

#         break


    pred_arg_tags_old = [[bert_io.idx2tag[p_i] for p_i in p] for p in pred_args]

    pred_arg_tags = []
    for old in pred_arg_tags_old:
        new = []
        for t in old:
            if t == 'X':
                new_t = 'O'
            else:
                new_t = t
            new.append(new_t)
        pred_arg_tags.append(new)

    valid_arg_tags = [[bert_io.idx2tag[v_i] for v_i in v] for v in true_args]
    f1 = f1_score(pred_arg_tags, valid_arg_tags)

    print("Validation loss: {}".format(eval_loss/nb_eval_steps))
    print("Validation F1-Score: {}".format(f1_score(pred_arg_tags, valid_arg_tags)))
    
    return f1


# In[7]:


ns = [1,2,3,4,5,6,7,8,9,10]
ns = [6,7,8,9,10]
model_fname = '/disk/data/models/korval/ko-srl-epoch-4.pt'
result = []
for n in ns:
    tst_fname = '/disk/project/corpus/test_'+str(n)+'.conll'
    print(tst_fname)
    tst = load_data(tst_fname)
    
    f1 = test(model_fname, tst)
    
    line = str(n)+':'+str(f1)
    print(line)
    result.append(line)

with open('/disk/project/corpus/eval_result.txt','w') as f:
    for i in result:
        f.write(i+'\n')

