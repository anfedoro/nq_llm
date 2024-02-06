
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
import math

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, os, time
from PIL import Image



def data_prepare ():
        #load data from hdf5 file and create a dataset
    with h5py.File("data/nq17-23_1min_candle_seq_1024.hdf5", "r") as f:
        # total_records = f["data"].shape[0]
        # start_index = int(total_records*0.62)
        # stop_index = int(total_records*0.65)
        # dataset = f["data"][start_index:stop_index]
        dataset = f["data"][:]
        dataset = torch.from_numpy(dataset).long()
    #load index to candlestick mapping from hdf5 file

        index_to_candle = {}
        for key in f.keys():
            if key != "data":
                group = f[key]
                sizes = group['sizes'][:]
                direction = group['direction'][()]
                index_to_candle[int(key)] = {'sizes':tuple(sizes), 'direction':direction}
        

    candle_seq_len = 16
    vocab_size = dataset.max() + 1

    dataset = dataset.unfold(0, candle_seq_len, 1)
    input_dataset = dataset[:, :-1]
    target_dataset = dataset[: , 1:]
    seq_len = input_dataset.shape[1]

    print(f'Input dataset shape: {input_dataset.shape}, Target dataset shape: {target_dataset.shape}')

    torch.manual_seed(42)
    split_idx = int(len(dataset)*0.8)
    train_data = input_dataset[:split_idx]
    train_targets = target_dataset[:split_idx]
    test_data = input_dataset[split_idx:]
    test_targets = target_dataset[split_idx:]

    val_split_idx = int(len(train_data)*0.8)
    val_data = train_data[val_split_idx:]
    val_targets = train_targets[val_split_idx:]
    train_data = train_data[:val_split_idx]
    train_targets = train_targets[:val_split_idx]

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    batch_size = 64
    #load data into dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=6)

    return train_loader, val_loader, test_loader, index_to_candle, vocab_size, seq_len


# function to restore candles from their codes and plot candlestick chart to writer

def full_candle_restore(candle_index:(np.array,torch.Tensor), index_map, start_price:float = 1000, number_of_candles = None) -> pd.DataFrame:
        
        '''
        Restore full candle from index of candle and start price
        '''
        tick_size = 0.25

        if isinstance(candle_index, torch.Tensor):
            candle_index = candle_index.numpy()
        if number_of_candles is not None:
            candle_index = candle_index[:number_of_candles]
        len = candle_index.shape[0]
        candle_index = candle_index[np.where(candle_index != 1024)]
        
        candles = []
        for idx, cdl_idx in enumerate(candle_index):
            
            top_wick, body, bottom_wick = index_map[cdl_idx]['sizes']
            direction = index_map[cdl_idx]['direction']
            candle = {}
            if idx == 0:
                candle['open'] = start_price
            
            else:
                candle['open'] = candles[-1]['close']
            
            close = candle['open'] + body * tick_size * direction
            high = close + top_wick * tick_size if close > candle['open'] else candle['open'] + top_wick * tick_size
            low = candle['open'] - bottom_wick * tick_size if close > candle['open'] else close - bottom_wick * tick_size
            candle['high'] = high
            candle['low'] = low
            candle['close'] = close
            candles.append(candle)
        
        #if datafame is empty fill with defaul canldes open = start_price, high = start_price, low = start_price, close = start_price. Number of candles = candle_index.shape[0]
        df = pd.DataFrame(candles)
        if df.empty:
            df = pd.DataFrame({'open': [start_price]*len, 'high': [start_price+2]*len, 'low': [start_price-1]*len, 'close': [start_price+1]*len})
        
        return df

def write_charts_to_TB(name,writer, targets, outputs, epoch, index_map):
    
    
    #convert outputs logit to candlestick codes
    outputs = outputs.softmax(dim=1).argmax(dim=1)

    

    original_candles = full_candle_restore(targets, index_map)
    predicted_candles = full_candle_restore(outputs, index_map)
    
    fig1 = go.Figure(data=[go.Candlestick(x=original_candles.index,
                open=original_candles['open'],
                high=original_candles['high'],
                low=original_candles['low'],
                close=original_candles['close'])])
    #increase chart size
    fig1.update_layout(height=600, width=1200)
    fig1.update_layout(xaxis_rangeslider_visible=False)
    

    fig2 = go.Figure(data=[go.Candlestick(x=predicted_candles.index,
                    open=predicted_candles['open'],
                    high=predicted_candles['high'],
                    low=predicted_candles['low'],
                    close=predicted_candles['close'])])
    
    fig2.update_layout(height=600, width=1200)
    fig2.update_layout(xaxis_rangeslider_visible=False)
    
    #convert figures to image and write to tensorboard
    fig1_bytes = fig1.to_image(format="png")
    fig2_bytes = fig2.to_image(format="png")

    # Преобразуем байтовые данные в массив NumPy
    fig1_image = np.array(Image.open(io.BytesIO(fig1_bytes)))
    fig2_image = np.array(Image.open(io.BytesIO(fig2_bytes)))

    # Преобразуем массивы NumPy в тензоры PyTorch
  

    # Добавляем изображения в TensorBoard
    writer.add_image(f'{name}_original', fig1_image, epoch, dataformats='HWC')
    writer.add_image(f'{name}_predicted', fig2_image, epoch, dataformats='HWC')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:x.size(1)].transpose(0, 1)  # Изменение размеров для соответствия формату batch_first
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        self.embed_dim = embed_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)  

    def forward(self, src):
        src = self.embedding(src) * math.sqrt(self.embed_dim)
      
        src = self.pos_encoder(src)
       
        output = self.transformer_encoder(src)
       
    
        output = self.output_layer(output)
        
        return output


#check if we have CUDA or MPS and setup respecive device, if not CUDA nor MPS is available, then use CPU
def init_model(vocab_size, embed_dim, num_heads, num_layers, dropout):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    #device = torch.device('cpu')

    print(f"Device to be used: {device}")
    #Initialize model
    torch.manual_seed(42)
    model = TransformerModel(vocab_size, embed_dim, num_heads, num_layers, dropout)

    model = model.to(device)
    next(model.parameters()).device
    return model, device
    

#define writer for tensorboard
def init_writer(continue_training = False, purge_step = None):

    
    if not continue_training:
        #use python os package to delete logs including files in subfolders and subfolders itself
        for root, dirs, files in os.walk('./runs/nq_llm_2'):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                for fils in os.listdir(os.path.join(root, dir)):
                    os.remove(os.path.join(root, dir, fils))
                os.rmdir(os.path.join(root, dir))   
    print(f"Initializing writer...purge step: {purge_step}")
    writer = SummaryWriter('runs/nq_llm_2', purge_step = purge_step)
    return writer


#caclulate number of trainable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params} trainable parameters')

def restore_model(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    best_vloss = checkpoint['vloss']
    epoch = checkpoint['epoch']
    return epoch, model, optimizer, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss

def adjust_learning_rate(optimizer,lr, weight_decay = None):
    for g in optimizer.param_groups:
        g['lr'] = lr
        if weight_decay is not None:
            g['weight_decay'] = weight_decay
    return optimizer


def train(model, optimizer, loss_fn, train_loader, val_loader, writer, device, loss_weights, index_to_candle, hyperparams):


    start_time = time.time()

    #unpack hyperparameters
    start_epoch = hyperparams['start_epoch']
    num_epochs = hyperparams['num_epochs']
    vocab_size = hyperparams['vocab_size']
    embed_dim = hyperparams['embed_dim']
    num_heads = hyperparams['num_heads']
    num_layers = hyperparams['num_layers']
    dropout = hyperparams['dropout']
    seq_len = hyperparams['seq_len']
    best_vloss = hyperparams['best_vloss']

    #set model to train mode

    try:
        for epoch in range(start_epoch,num_epochs):
            # Train the model
            model.train()
            epoch_loss = 0
            for batch_idx, (data, labels) in enumerate(train_loader):
                data = data.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(data)

                #loss caclulation with weighted last candle

                loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))
                loss = loss.view(-1, seq_len)
                weighted_loss = loss * loss_weights
                loss = weighted_loss.mean()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                time_since_start = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
                print(f'Training time: {time_since_start}, Epoch: [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {epoch_loss / (batch_idx + 1):.6f}, Instant loss:{loss.item():.6f}', end='\r', flush=True)
                if batch_idx % 10 == 0:
                    writer.add_scalars('Batch Loss', {'Average': epoch_loss / (batch_idx + 1), 'Instant': loss.item()}, epoch * len(train_loader) + batch_idx)
                
                   

            #add weights and biases to tensorboard
            for name, param in model.named_parameters():
                if 'weight' in name:
                    writer.add_histogram(f'weights/{name}', param, epoch)
                elif 'bias' in name:
                    writer.add_histogram(f'biases/{name}', param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f'grads/{name}', param.grad, epoch)



            # Test the model
            
            model.eval()
            vepoch_loss = 0
            with torch.inference_mode():
                correct = 0
                total = 0
                for vbatch_idx, (vdata, vlabels) in enumerate(val_loader):
                    vdata = vdata.to(device)
                    vlabels = vlabels.to(device)
                    voutputs = model(vdata)
                    #loss caclulation with weighted last candle
                    vloss = loss_fn(voutputs.view(-1, vocab_size), vlabels.view(-1))
                    vloss = vloss.view(-1, seq_len)
                    vweighted_loss = vloss * loss_weights
                    vloss = vweighted_loss.mean()
                    vepoch_loss += vloss.item()

                    #calculate accuracy
                    last_redicted_candle = voutputs[:,-1,:].softmax(dim=1).argmax(dim=1)
                    last_actual_candle = vlabels[:,-1]
                    correct += (last_redicted_candle == last_actual_candle).sum().item()
                    total += last_actual_candle.size(0)
                    accuracy = correct / total * 100
                    
                    print(f'Training time: {time_since_start}, Epoch [{epoch + 1}/{num_epochs}], Step [{vbatch_idx + 1}/{len(val_loader)}], Validation Loss: {vepoch_loss/(vbatch_idx +1):.6f}, Validation accuracy: {accuracy:.2f}%', end='\r', flush=True)
                    

            # Save the model checkpoint if validation loss is less than best validation loss
            if vepoch_loss/len(val_loader) < best_vloss:
                best_vloss = vepoch_loss/len(val_loader)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'vloss': best_vloss,
                
                    }, './models/nq-llm_0_2.pth')
                lr = next(iter(optimizer.param_groups))['lr']
                weight_decay = next(iter(optimizer.param_groups))['weight_decay']
                print(f"Training time: {time_since_start}, Model saved at epoch {epoch+1} with validation loss {vepoch_loss/len(val_loader):.6f} Learning rate: {lr:.2e} Weight decay: {weight_decay:.2e} ")
            

            print(f'Training time: {time_since_start}, Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss / len(train_loader):.6f}, Validation Loss: {vepoch_loss / len(val_loader):,.6f}, Validation accuracy: {accuracy:.2f}% ')
            writer.add_scalars('Loss', {'Train': epoch_loss / len(train_loader), 'Test': vepoch_loss / len(val_loader)}, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
            
            
            #write charts to tensorboard
            write_charts_to_TB('Test data sample',writer, vlabels[0].cpu(), voutputs[0].cpu(), epoch, index_to_candle)
            write_charts_to_TB('Test predicted candles sequence',writer, vlabels[:60,-1].cpu(), voutputs[:60,-1,:].cpu(), epoch, index_to_candle)
            

    except KeyboardInterrupt:
        print(f"Interrupted at epoch {epoch}")
    finally:
        writer.close()
        print('Training completed.')

def main():
    
    #load data
    train_loader, val_loader, test_loader, index_to_candle, vocab_size, seq_len = data_prepare()
    
    loss_weights = torch.ones(seq_len)
    loss_weights[-1] = 5.0

    #initialize hyperparameters
    hp = {'start_epoch':0, 
            'num_epochs':5, 
            'vocab_size':vocab_size, 
            'embed_dim':256, 
            'num_heads':32, 
            'num_layers':32, 
            'dropout':0.1, 
            'seq_len': seq_len, 
            'best_vloss':float('inf')}

    #initialize model, loss function and optimizer
    model, device = init_model(hp['vocab_size'], hp['embed_dim'], hp['num_heads'], hp['num_layers'], hp['dropout'])
    count_parameters(model)
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    loss_weights = loss_weights.to(device)

    # #load model from checkpoint
    # print('Loading model from checkpoint...')
    # epoch, model, optimizer, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss = restore_model(model, optimizer, './models/nq-llm_0_2.pth')
    # print(f'Model loaded from checkpoint. Epoch: {epoch+1}, Validation loss: {best_vloss:.6f}')

    # #update hyperparameters
    # hp['start_epoch'] = epoch + 1
    # hp['num_epochs']  = 15
    # hp['best_vloss'] = best_vloss

    #set learning rate and weight decay
    optimizer = adjust_learning_rate(optimizer,1e-5, 1e-5)

    #initialize writer
    #purge_step = (epoch+1)*len(train_loader)
    writer = init_writer(continue_training=False)


    print(f'Model training with learning rate: {optimizer.param_groups[0]["lr"]:.2e}, weight decay: {optimizer.param_groups[0]["weight_decay"]:.2e}')
    #train model
    train(model, optimizer, loss_fn, train_loader, val_loader, writer, device, loss_weights, index_to_candle, hp)
    
if __name__ == '__main__':
    main()



        

