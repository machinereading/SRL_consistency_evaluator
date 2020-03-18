import json
import glob
import random

def get_word_id(item, word):
    word_id = False
    e = item['end']
    for i in range(len(word)):
        w = word[i]
        if w['begin'] <= e <= w['end']:
            word_id = i
            break

    return word_id

def words2tokens(words):
    tokens = []
    for i in words:
        tokens.append(i['form'])
    return tokens

def convert_for_sent(srls, words):
    result = []
    tokens = words2tokens(words)
    for srl in srls:
        if srl:
            pred_id = get_word_id(srl['predicate'], words)
            pred = srl['predicate']['lemma'] + '.' + str(srl['predicate']['sense_id'])            
            
            preds = ['_' for i in range(len(tokens))]
            senses = ['_' for i in range(len(tokens))]
            preds[pred_id] = 'PRED'
            senses[pred_id] = pred
            
            args = ['O' for i in range(len(tokens))]
            
            for arg in srl['argument']:
                arg_id = get_word_id(arg, words)
                label = arg['label'].replace('-', '_')
                args[arg_id] = label
                
            sent = []
            sent.append(tokens)
#             sent.append(preds)
            sent.append(senses)
            sent.append(args)
            
            result.append(sent)
            
    return result

def converter(files):
    conll = []
    
    doc_list = []
    for fname in files:        
        docid = fname.split('/')[-1].split('.')[0]
        doc_list.append(docid)
        
        with open(fname, 'r') as f:
            d = json.load(f)
            
        for doc in d['document']:
            for sent in doc['sentence']:
                words = sent['word']
                srl = sent['SRL']
                srl_conll = convert_for_sent(srl, words)
                
                conll += srl_conll
        
    return conll

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]


def load_data(fdir, split=100, n_split=False):
    print('converting dataset...')
    if fdir[-1] != '/':
        fdir = fdir + '/'
    files = glob.glob(fdir+'*.json')
    
    if len(files) == 0:
        print('학습데이터 경로가 잘못되었습니다')
        
    if n_split == False:
        conll = converter(files)
        k = int(len(conll) * (int(split) / 100))
        percent = int(split)
        data = random.sample(conll, k=k)
        
        print('...is done\n')
        print('###total data:', len(conll), 'annotations')
        print('###training data:',len(data), 'annotations')
        print('###percent =', str(percent)+'%')        
        
    else:
        n = int(len(files) / int(n_split))
        splited_files = list(chunks(files, n+1))
        
        data = []
        docids = []
        for i in range(len(splited_files)):
            split = splited_files[i]
            conll = converter(split)
            data.append(conll)
            print('split',i,':', len(conll),'annotations')
            
            dis = []
            for fname in split:
                di = fname.split('/')[-1].split('.')[0]
                dis.append(di)
            docid = ', '.join(dis)
            line = 'split '+str(i)+':'+'\t'+docid+'\n'
            docids.append(line)
            
        with open('./splited_documents.txt','w') as f:
            for line in docids:
                f.write(line)
            
        print('splited document ids are saved to:')
        print('\t','./splited_documents.txt')
        print('')
    
    return data
    
        
        