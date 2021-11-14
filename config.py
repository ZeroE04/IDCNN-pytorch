# -----------INPUT----------------
TRAIN_DATA = 'data/raw/train.txt'
VALID_DATA = 'data/raw/valid.txt'
# EMBEDDING_FILE = 'sgns.wiki.word'

# -----------out_story----------------
save_dir = 'model'
save_name = 'fidcnn_crf.pb'

# -----------PARAMETERS----------------
max_len = 30
word_embedding_dim = 300
batch_size = 512
num_epoch = 50
early_stop = 5
initial_lr = 0.0005
