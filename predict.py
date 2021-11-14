# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:22:06 2021

@author: yilin3
"""
import os
import  sys
import torch
import config as config
from preprocess import load_obj
from ner import DICNN_CRF


if __name__ == '__main__':
    text = list(sys.argv[1])
    data_dir = f"data/processed"
    word_to_ix = load_obj(os.path.join(data_dir, "word_to_ix.pkl"))
    tag_to_ix = load_obj(os.path.join(data_dir, "tag_to_ix.pkl"))
    idx2tag = load_obj(os.path.join(data_dir, "idx2tag.pkl"))
    
    model = DICNN_CRF(
        word_embedding_dim=config.word_embedding_dim,
        word2id=word_to_ix,
        num_tag=len(tag_to_ix),
        save_dir=config.save_dir,
        processed_emb = data_dir
        )

    device = torch.device('cuda')
    
    model.load_state_dict(torch.load(os.path.join(config.save_dir,config.save_name)))
    model.eval()
    model.cuda()

    word_ids_ = [[word_to_ix.get(w, word_to_ix["<UNK>"]) for w in text]]

    word_ids = []
    for word_id in word_ids_:
        word_ids.append(word_id[:config.max_len] if len(word_id)>config.max_len else word_id + [0]*(config.max_len-len(word_id))) 

    word_ids = torch.tensor(word_ids, dtype=torch.long)
    word_ids = word_ids.cuda()
    logits = model(word_ids)
    slot_pred = model.crf.decode(logits)
    slot_pred = torch.squeeze(slot_pred)
    slot_pred = [idx2tag.get(c.item(),'O') for c in slot_pred]
    result='  '.join([t+r for t, r in zip(text,slot_pred)])
    print(result)
