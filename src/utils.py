import itertools
import json
import os
import pickle
from logging import getLogger
from typing import Callable, Dict, Iterable, List
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
import sys
import os
import torch
import random



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def trim_batch(
    input_ids, pad_token_id, attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask]) 


class Seq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        prompt,
        type_path="train", 
        sample=None,
        task='def-gen-with-contras',
    ):
        super().__init__()
        sys.stderr.write('Reading corpora...')  
        self.type_path = type_path
        self.ignore_sense_id = True
        self.prompt = prompt

        self.data_dir = data_dir
     
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id
        self.task = task
        self.sample = sample
        self.data, self.data_ntoken = self.tokenize((data_dir+'{}.txt'.format(type_path.split('_')[-1])))
        
        self.data_word_egs = self.read_examples((data_dir+'{}.eg'.format(type_path.split('_')[-1])))
        self.data = self.add_examples(self.data, self.data_word_egs)
        self.tokenizer = tokenizer
        self.word_spans = []
        
        self.inputs = []
        self.targets = []
        self.target_word = []        
        self.encode(self.data, data_dir)

       
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index) -> Dict[str, torch.Tensor]:
        
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()
        src_mask    = self.inputs[index]["attention_mask"].squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"].squeeze()  # might need to squeeze
        
        word = self.target_word[index]
        
        word_span = self.word_spans[index]
        return {
            "source_ids": source_ids, "source_mask": src_mask,
            "target_ids": target_ids, "target_mask": target_mask, 
            "target_word": word, "word_span":word_span
        }
        
 
    def tokenize(self, path):
        word_desc_orig = []
        ntoken = 0
        with open(path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                elems = line.strip().split('\t')
                word = elems[0]
                description = elems[3].split()
                word_desc_orig.append((word,description))
                ntoken += (len(description) - 1) 
        return word_desc_orig, ntoken
    
    def read_examples(self, path):
        assert os.path.exists(path)
        word_egs = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                word, eg = line.strip().split('\t')
                word_egs.append((word, eg.split(' ')))
        return word_egs


    def add_examples(self, word_char_vec_desc, word_egs):
        word_char_vec_desc_eg = []
        for i, (word, desc) in enumerate(word_char_vec_desc):

            eg_id = []
            for w in word_egs[i][1]:
                eg_id.append(w)            
            word_char_vec_desc_eg.append((word, desc, eg_id))
        return word_char_vec_desc_eg  

    

    def encode_each_input(self, source, target, word):
        src = self.tokenizer.batch_encode_plus(
                  [source], max_length=self.max_source_length, pad_to_max_length=True, truncation=True, return_tensors="pt"
              )
        trg = self.tokenizer.batch_encode_plus(
                  [target], max_length=self.max_source_length, pad_to_max_length=True, truncation=True, return_tensors="pt"
              )            
       
        e = self.tokenizer.encode(word)[:-1]
        if len(e) < 1:
            e = [2]
            
        src_tok_list = src['input_ids'][0].tolist()
        flag = 0
        
        for idx, tok_id in enumerate(src_tok_list):
            if tok_id == e[0]:
                start_idx = idx
                if len(e) == 1:
                    end_idx = start_idx
                    break

                for idx_ in range(start_idx + 1, len(src_tok_list)):
                    if src_tok_list[idx_] == e[-1] and start_idx + len(e) == idx_ + 1:
                        end_idx = idx_
                        flag = 1
                        break
            if flag:
                break
        
        try:
            self.word_spans.append((start_idx, end_idx + 1))
        except UnboundLocalError:
            print('this sample has some error, we pass it')
            return 
        self.inputs.append(src)
        self.targets.append(trg)    
        self.target_word.append(e)  


    def encode(self, data, data_dir):
        for index, (word, definition, example) in enumerate(data):
            if self.sample:
                if index + 1 > self.sample:
                    break  
    
            sememe = "　"
            word_ = "word: " + word
            context = " context: " + " ".join(example).replace('<TRG>', word)
            source = word_ + sememe + context  
            target = " ".join(definition) 
            self.encode_each_input(source, target, word)


            
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        
        pad_token_id = self.pad_token_id
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)        
        y = trim_batch(target_ids, pad_token_id)
        
        words = [x["target_word"] for x in batch]
        word_spans = [x["word_span"] for x in batch]
        
        batch = {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "decoder_input_ids": y,
            "target_word": words,
            "word_spans":word_spans
        }
        return batch
    
logger = getLogger(__name__)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)




def save_json(content, path):
    with open(path, "w+") as f:
        json.dump(content, f, indent=4)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))

