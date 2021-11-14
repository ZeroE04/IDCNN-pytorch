# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 10:22:06 2021
"""
import os
import pickle
import config as config

word_to_ix = {"<PAD>": 0, "<UNK>": 1}
tag_to_ix = {}

def read_file(filename):
    """Parse plain text data into arrays
    """
    data = []
    with open(filename, "r", encoding="utf8") as f:
        sentences = f.read().split("\n")
        for s in sentences:
            try:
                q, l = s.split("\t")
                q = list(q)
                l = l.split(" ")
                data.append((q,l))
            except:
                continue
    return data



def get_vocabs_from_data(train_data):
    """Extract all words in training data
    """
    global word_to_ix
    global tag_to_ix
    for words, tags in train_data:
        for w in words:
            if w not in word_to_ix:
                word_to_ix[w] = len(word_to_ix)
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)


def dump_obj(obj, filename):
    """Wrapper of pickle.dump
    """
    with open(filename, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_obj(filename):
    """Wrapper of pickle.load
    """
    with open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj

def transform(data):
    """Transform words and tags to idx
    """
    new_data = []
    for words, tags in data:
        word_ids = [word_to_ix.get(w, word_to_ix["<UNK>"]) for w in words]
        tag_ids = [tag_to_ix.get(t, tag_to_ix['O']) for t in tags]
        if len(word_ids)>config.max_len:
            word_ids = word_ids[:config.max_len]
            tag_ids = tag_ids[:config.max_len]
        new_data.append((word_ids, tag_ids))
    return new_data


if __name__ == "__main__":
    train_data = read_file(config.TRAIN_DATA)
    test_data = read_file(config.VALID_DATA)
    get_vocabs_from_data(train_data)
    train_data = transform(train_data)
    test_data = transform(test_data)
    data_dir = f"data/processed"
 
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    dump_obj(train_data, os.path.join(data_dir, "train.pkl"))
    dump_obj(test_data, os.path.join(data_dir, "valid.pkl"))
    dump_obj(word_to_ix, os.path.join(data_dir, "word_to_ix.pkl"))
    dump_obj(tag_to_ix, os.path.join(data_dir, "tag_to_ix.pkl"))
    idx2tag = {value:key for key,value in tag_to_ix.items()}
    dump_obj(idx2tag, os.path.join(data_dir, "idx2tag.pkl"))
    print("Done!")
