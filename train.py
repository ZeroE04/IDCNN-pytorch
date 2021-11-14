# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:22:06 2021
@author: yilin3
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import config as config
from ner import DICNN_CRF
from preprocess import load_obj
from dataset import BatchPadding, NERDataset

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)

def fit(model, training_iter, eval_iter, num_epoch, initial_lr, verbose=1):
    model.cuda()
    model.apply(weights_init)
    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr,weight_decay=0.01)
    best_valid_score = 0.0
    non_progress_count = 0
    train_loss = 0
    for e in range(num_epoch):
        print("begin train epoch "+str(e)+"...")
       
        model.train()
        for index, (inputs, label, input_mask) in enumerate(training_iter):
            
            inputs,label,input_mask = inputs.cuda(),label.cuda(),input_mask.cuda()
            output = model(inputs)
            # input_mask = inputs.ne(-1).int()
            train_loss = -1 * model.crf(output, label, mask=input_mask.byte(),
                                               reduction="mean")
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            if index%50==0:
                print("进度：%.2f%%-----> train_loss: %f "%(index/len(training_iter)*100, train_loss.detach()))
        
        torch.cuda.empty_cache()
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            predict_set, label_set = [], []
            pre_set ,lab_set = [],[]
            count = 0
            for eval_inputs, eval_label, eval_mask in eval_iter:
                eval_inputs, eval_label, eval_mask = eval_inputs.cuda(), eval_label.cuda(), eval_mask.cuda()
                
                output = model(eval_inputs)
                eval_loss += -1 * model.crf(output, eval_label, mask=eval_mask.byte(),reduction="mean")
                eval_predicts = model.crf.decode(output)
                eval_predicts = torch.squeeze(eval_predicts) 
                for i in range(len(eval_label)):
                    for j in range(len(eval_label[i])):
                        if eval_label[i][j] == -1:
                            eval_predicts[i][j] = -1
                pre_set.append(eval_predicts)
                eval_predicts = eval_predicts.contiguous().view(1, -1).squeeze()
                eval_predicts = eval_predicts[eval_predicts != -1]
                predict_set.append(eval_predicts)

                lab_set.append(eval_label)
                eval_label = eval_label.contiguous().view(1, -1).squeeze()
                eval_label = eval_label[eval_label != -1]
                label_set.append(eval_label)
                count += 1
            eval_acc_sen= model.evaluate_sentence(pre_set, lab_set)

            print("epoch:" + str(e) +  ", val acc is: "+str(eval_acc_sen))
            if eval_acc_sen > best_valid_score:
                best_valid_score = eval_acc_sen
                non_progress_count = 0
                model.save()
            else:
                non_progress_count += 1
       
            if config.early_stop > 0:
                if non_progress_count >= config.early_stop:
                    print(f"Early stop at epoch {e+1}, best_valid_score is: "+str(best_valid_score))
                    return ""

        
if __name__ == '__main__':
    data_dir = f"data/processed"
    train_data = NERDataset(os.path.join(data_dir, "train.pkl"))
    valid_data = NERDataset(os.path.join(data_dir, "valid.pkl"))
    word_to_ix = load_obj(os.path.join(data_dir, "word_to_ix.pkl"))
    tag_to_ix = load_obj(os.path.join(data_dir, "tag_to_ix.pkl"))
    train_loader = DataLoader(train_data, batch_size=config.batch_size, collate_fn=BatchPadding(), shuffle=True, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=config.batch_size, collate_fn=BatchPadding(), shuffle=True, num_workers=0, pin_memory=True)
    
    model = DICNN_CRF(
        word_embedding_dim=config.word_embedding_dim,
        word2id=word_to_ix,
        num_tag=len(tag_to_ix),
        save_dir=config.save_dir,
        processed_emb = data_dir
        )
    
    fit(model, train_loader, valid_loader,
        config.num_epoch,
        config.initial_lr, verbose=1)
