import torch
from torch.utils.data import Dataset, DataLoader

#Dataset class which loads the data and converts it to torch tensors on dalaloader request
class PriceMovesDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, max_len=512):
        self.sequence = sequence
        self.max_len = max_len
        self.max_word_len = max([len(x) for x in sequence])
        


    def __len__(self):
        last_seq_len = 0
        idx = -2
        while last_seq_len <= self.max_len:
            last_seq_len += len(self.sequence[idx])
            idx -= 1

        return len(self.sequence) - last_seq_len
    

    def __getitem__(self, index):
        input = torch.zeros(self.max_len, dtype=torch.long)
        target = torch.zeros(self.max_word_len, dtype=torch.long)
        idx = 0
        while idx+ len(self.sequence[index]) + len(self.sequence[index+1]) <= self.max_len:
            input[idx:idx+len(self.sequence[index])] = torch.tensor(self.sequence[index])
            idx = idx + len(self.sequence[index])
            index += 1
        target[:len(self.sequence[index])] = torch.tensor(self.sequence[index])
        return input, target