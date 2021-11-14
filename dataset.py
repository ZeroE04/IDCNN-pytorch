import torch
from torch.utils.data import Dataset

from preprocess import load_obj
import config as config

class NERDataset(Dataset):
    """NER Dataset
    """

    def __init__(self, dataset_pkl):
        super(NERDataset, self).__init__()
        self.dataset = load_obj(dataset_pkl)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.dataset[idx][0], dtype=torch.long),
            torch.tensor(self.dataset[idx][1], dtype=torch.long),
        )


class BatchPadding(object):
    """Padding in batch and sequences is sorted by length in order
    """
    def __init__(self, descending=True):
        self.reverse = True if descending else False

    def __pad__(self,sequences,max_len,p=1):
        out_dims = (max_len,  len(sequences))
        out_tensor = sequences[0].data.new(*out_dims).fill_(p)
        for i, tensor in enumerate(sequences):
            
            length = tensor.size(0)
            out_tensor[:length, i, ...] = tensor
        return out_tensor
    
    def __call__(self, batch):
        """batch should be a list of tensors
        """
        sorted_batch = sorted(batch, key=lambda x: len(x[0]), reverse=self.reverse)
        seqs, tags = tuple(zip(*sorted_batch))
        seqs = self.__pad__(seqs,config.max_len,0) # 0 padding         
        tags = self.__pad__(tags,config.max_len,-1)
        seqs = seqs.transpose(0,1)
        tags = tags.transpose(0,1)
        masks = tags.ne(-1)
        return seqs, tags, masks.int()
