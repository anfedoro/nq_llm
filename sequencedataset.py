
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class CandleSeriesDataset(Dataset):
    def __init__(self, sequence, window_size=256):
        """
        Args:
            sequence (Tensor): Tensor of candlesticks tokens sequence and labels.
            window_size (int): window size for the sliding window approach.
        """
        self.sequence = sequence
        self.window_size = window_size

    def __len__(self):
        
        # Subtruact window_size to leave room for the label (the next token)  
        return len(self.sequence) - self.window_size

    def __getitem__(self, idx):
        
        # Rerurn a window of sequence and the next token as a label

        return (self.sequence[idx:idx+self.window_size], self.sequence[idx+self.window_size])
    

def candle_restore(candles:(np.array,torch.Tensor), start_price:float, index_to_value:list, number_of_candles = None, tick_size = 0.25 ) -> pd.DataFrame:
    #function receives a dataframe with columns ['first_wick', 'last_wick'] - respectivey columns 0 and 2 in the array) for wicks data and column 'body' (column 1 in the array) for body data 
    # and restores the candles starting from initial price. New open gap should be considered from column 3 in the array
    #use index_to_bottom_wick, index_to_body, index_to_top_wick to convert index to value
    if isinstance(candles, torch.Tensor):
        candles = candles.numpy()
    if number_of_candles is not None:
        candles = candles[:number_of_candles]

    index_to_bottom_wick = index_to_value[0]
    index_to_body = index_to_value[1]
    index_to_top_wick = index_to_value[2]
    index_to_open_gap = index_to_value[3]

    restored_candles = []
    for idx, candle in enumerate(candles):
        restored_candle = {}
        open = start_price if idx == 0 else restored_candles[idx-1]['close'] + index_to_open_gap[candle[3]] * tick_size
        if candle[0] in index_to_bottom_wick:
            low_wick = index_to_bottom_wick[candle[0]]
            high_wick = index_to_top_wick[candle[2]]
            body = index_to_body[candle[1]]
            close = open + body * tick_size
            low = open - low_wick * tick_size
            high = close + high_wick * tick_size
            restored_candle['open'] = open
            restored_candle['high'] = high
            restored_candle['low'] = low
            restored_candle['close'] = close

        elif candle[0] in index_to_top_wick:
            high_wick = index_to_top_wick[candle[0]]
            low_wick = index_to_bottom_wick[candle[2]]
            body = index_to_body[candle[1]]
            close = open - body * tick_size
            low = close - low_wick * tick_size
            high = open + high_wick * tick_size
            restored_candle['open'] = open
            restored_candle['high'] = high
            restored_candle['low'] = low
            restored_candle['close'] = close

            
        restored_candles.append(restored_candle)
    return pd.DataFrame(restored_candles)


