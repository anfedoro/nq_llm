
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
#from torch.utils.data import DataLoader, TensorDataset
import math
import threading

import h5py
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import io, os, time
from PIL import Image

best_model_path = ''
last_model_path = ''
log_dir = ''


def data_prepare (candle_seq_len = 11):
        #load data from hdf5 file and create a dataset
    with h5py.File("data/nq17-23_1min_candle_seq_bpe100.hdf5", "r") as f:
        # total_records = f["data"].shape[0]
        # start_index = int(total_records*0.60)
        # stop_index = int(total_records*0.65)
        # dataset = f["data"][start_index:stop_index]
        dataset = f["data"][:]
        dataset = torch.from_numpy(dataset).long()
    #load index to candlestick mapping from hdf5 file

        index_to_candle = {}
        # for key in f.keys():
        #     if key != "data":
        #         group = f[key]
        #         sizes = group['sizes'][:]
        #         direction = group['direction'][()]
        #         index_to_candle[int(key)] = {'sizes':tuple(sizes), 'direction':direction}
        

    vocab_size = dataset.max() + 1 

    dataset = dataset.unfold(0, candle_seq_len, 1)
    input_dataset = dataset[:, :-1]
    target_dataset = dataset[: , -1]

    print(f'Input dataset shape: {input_dataset.shape}, Target dataset shape: {target_dataset.shape}')

    torch.manual_seed(42)
    split_idx = int(len(dataset)*0.8)
    train_data = input_dataset[:split_idx]
    train_targets = target_dataset[:split_idx]
    val_data = input_dataset[split_idx:]
    val_targets = target_dataset[split_idx:]
   
    return vocab_size, train_data, val_data, train_targets, val_targets, index_to_candle


# function to restore candles from their codes and plot candlestick chart to writer

class BatchIterator:
    def __init__(self, batch_size, num_batches, inputs, targets, shuffle=True):
        self.batch_size = batch_size
        self.num_batches = num_batches - 1
        self.inputs = inputs
        self.targets = targets
        self.shuffle = shuffle
        self.dataset_len = batch_size * num_batches

    def __iter__(self):
        self.current_batch = 0
        if self.shuffle:
            self.perm = torch.randperm(self.inputs.shape[0])[:self.dataset_len]
        else:
            self.perm = torch.arange(self.inputs.shape[0])[:self.dataset_len]
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        s = self.current_batch * self.batch_size
        ids = self.perm[s : s + self.batch_size]
        batch_inputs = self.inputs[ids]
        batch_targets = self.targets[ids]
        self.current_batch += 1

        return self.current_batch, batch_inputs, batch_targets





def test_seq_generator(model, start_seq, test_seq_len):
    with torch.inference_mode():
        seq = start_seq.clone()
        predict_seq = torch.zeros(test_seq_len, dtype=torch.long)
        for i in range(test_seq_len):
            output = model(seq)
            output_token = output.softmax(dim=1).argmax(dim=1)
            predict_seq[i] = output_token
            #shift sequence to the left and add predicted token to the end
            seq = seq.roll(-1, dims=1)
            seq[:,-1] = output_token
    return predict_seq

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

def write_charts_to_TB(name,writer, sequence, epoch, index_map):
    
    
    #convert outputs logit to candlestick codes
    #outputs = outputs.softmax(dim=1).argmax(dim=1)

    

  
    predicted_candles = full_candle_restore(sequence, index_map)
    

    

    fig2 = go.Figure(data=[go.Candlestick(x=predicted_candles.index,
                    open=predicted_candles['open'],
                    high=predicted_candles['high'],
                    low=predicted_candles['low'],
                    close=predicted_candles['close'])])
    
    fig2.update_layout(height=600, width=1200)
    fig2.update_layout(xaxis_rangeslider_visible=False)
    #add candle index label higher candle highs for few points
    fig2.add_trace(go.Scatter(x=predicted_candles.index, y=predicted_candles['high']+2, text=sequence, mode="text", textposition="bottom center"))
    
    
    #convert figures to image and write to tensorboard
    fig2_bytes = fig2.to_image(format="png")

    # Преобразуем байтовые данные в массив NumPy

    fig2_image = np.array(Image.open(io.BytesIO(fig2_bytes)))

    # Преобразуем массивы NumPy в тензоры PyTorch
  

    # Добавляем изображения в TensorBoard

    writer.add_image(f'{name}', fig2_image, epoch, dataformats='HWC')
    writer.flush()

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

class AutoregressionDecoderModel(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim, num_heads, num_layers, dropout=0.1, ff_mult=4):
        super(AutoregressionDecoderModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0,)
        #self.pos_encoder = nn.Embedding(seq_len, embed_dim) 
        self.pos_encoder = PositionalEncoding(embed_dim, dropout, seq_len)
        self.decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, embed_dim*ff_mult, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)
        self.output_layer = nn.Linear(embed_dim, vocab_size, bias=False)
        self.bnorm = nn.BatchNorm1d(vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.output_combine = nn.Linear(vocab_size*seq_len, vocab_size)
        
        self.embed_dim = embed_dim
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.05)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)


    def forward(self, src):
        #positions = torch.arange(0, src.size(1)).unsqueeze(0).to(src.device)
        #x = self.embedding(src) + self.pos_encoder(positions)
        x = self.embedding(src)
        x = self.pos_encoder(x)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1),)
        
        tgt_mask = tgt_mask.to(src.device)
        output = self.decoder(x, x, tgt_mask=tgt_mask)
        output = self.output_layer(output)
        output = self.bnorm(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = nn.ReLU6()(output)
        output = self.dropout(output)
        output = self.output_combine(output.reshape(output.shape[0], -1))

        return output


#check if we have CUDA or MPS and setup respecive device, if not CUDA nor MPS is available, then use CPU
def init_model(vocab_size, seq_len, embed_dim, num_heads, num_layers, dropout, ff_mult=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    #if CUDA enable TF32
    if device == torch.device('cuda'):
        torch.backends.cuda.matmul.allow_tf32 = True 
        torch.backends.cudnn.allow_tf32 = True   

    #device = torch.device('cpu')

    print(f"Device to be used: {device}")
    #Initialize model
    torch.manual_seed(42)
    model = AutoregressionDecoderModel(vocab_size, seq_len, embed_dim, num_heads, num_layers, dropout, ff_mult)

    model = model.to(device)
    next(model.parameters()).device
    return model, device
    

#define writer for tensorboard
def init_writer(continue_training = False, purge_step = None, writers = None):
 
    if not continue_training:
        #use python os package to delete logs including files in subfolders and subfolders itself
        for root, dirs, files in os.walk(log_dir):
            print(f"Deleting writer logs in {root}")
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                for fils in os.listdir(os.path.join(root, dir)):
                    os.remove(os.path.join(root, dir, fils))
                os.rmdir(os.path.join(root, dir))  
    if purge_step is not None and writers is not None:
        print(f"Initializing writer...purge step: {purge_step}")
        for wr in reversed(writers):
            writer = SummaryWriter(wr+'/', purge_step = purge_step)    
    else:
        print(f"Initializing writer...")
        writer = SummaryWriter(log_dir)
    return writer


#caclulate number of trainable parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {params} trainable parameters')

def restore_model(model, optimizer, scaler, path, device):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model = model.to(torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler'])
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    num_heads = checkpoint['num_heads']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    best_vloss = checkpoint['vloss']
    best_loss = checkpoint['loss']
    best_accuracy = checkpoint['accuracy']
    epoch = checkpoint['epoch']
    writers = checkpoint['writers']
    return epoch, model, optimizer, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss,best_loss, best_accuracy, writers

def get_leaning_rates(embed_dim, warmup_steps, num_epochs):
    l_rates = []
    for epoch in range(num_epochs):
        lr = embed_dim ** (-2.0) * min((epoch+1) ** (-0.5), (epoch+1)*warmup_steps**(-1.5))
        if epoch > int(warmup_steps) and epoch % 10 in range(0,5):
            lr = lr * 1
        l_rates.append(lr)
        
    return l_rates, max(l_rates)

def set_learning_rate(optimizer, lr, weight_decay = 0):
    
    for g in optimizer.param_groups:
        g['lr'] = lr 
        g['weight_decay'] = weight_decay
    return optimizer

def modified_learning_rate_schedule(optimizer, step, total_steps, warmup_steps, max_lr, max_wd = 0):
    step += 1
    if step < warmup_steps:
        # Гиперболическое увеличение скорости обучения во время разогрева
        lr =  max_lr * (1 - (warmup_steps - step) / warmup_steps) ** 2
        weight_decay = 0
    else:
        # Параболическое уменьшение скорости обучения после разогрева
        decay_steps = total_steps - warmup_steps
        decay_rate = (step - warmup_steps) / decay_steps
        lr =  max_lr * (1 - decay_rate) ** 2 # * 10 if (step % 10) in [0,1,2] else max_lr * (1 - decay_rate) ** 2
        weight_decay = max_wd*(1-decay_rate)**8

    for g in optimizer.param_groups:
        g['lr'] = lr
        g['weight_decay'] = weight_decay
    return optimizer


def store_model(model, optimizer, scaler, path, epoch, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss, best_loss, best_accuracy, writer):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler': scaler.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_dim': embed_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'vloss': best_vloss,
                    'loss': best_loss,
                    'accuracy': best_accuracy,
                    'writers': list(writer.all_writers.keys())
                
                    }, path)

max_train_batches = 0
max_val_batches = 0


def train(model, optimizer, scaler, loss_fn, train_loader, val_loader, writer, device, index_to_candle, hyperparams):

   
    RED = "\033[31m"
    GREEN = "\033[32m"
    RESET = "\033[0m"

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
    best_loss = hyperparams['best_loss']
    best_accuracy = hyperparams['best_accuracy']
    
    warmup_steps = hyperparams['warmup_steps']
    l_rates, max_lr = get_leaning_rates(embed_dim, warmup_steps, num_epochs)

    
    
    start_time = time.perf_counter()
    #set model to train mode
    try:
        for epoch in range(start_epoch,num_epochs):
            # Train the model
            optimizer = set_learning_rate(optimizer, l_rates[epoch], weight_decay=1e-4)
            #optimizer = modified_learning_rate_schedule(optimizer, epoch, num_epochs, warmup_steps=warmup_steps, max_lr=max_lr, max_wd=0)
            if epoch == 0:
                print(f'Model training with initial learning rate: {next(iter(optimizer.param_groups))["lr"]:.2e} and '
                      f'weight decay: {next(iter(optimizer.param_groups))["weight_decay"]:.2e} for {num_epochs} epochs.\n'
                      f'Max.learning rate: {max_lr:.2e} at epoch {warmup_steps}')
            model.train()
            epoch_loss = 0
            toc = time.perf_counter()
            data_len = 0
            samples_num = None
            for batch_idx, data, labels in train_loader:
               
                data = data.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    # Forward pass
                outputs = model(data)

                
                    #loss calculation
                loss = loss_fn(outputs.view(-1, vocab_size), labels.view(-1))

                epoch_loss += loss.item()

                # Backward and optimize
                
                loss.backward()
                optimizer.step()
                

                tic = time.perf_counter() - toc
                data_len += data.shape[0]
                step_time = data_len / tic
                if samples_num is None:
                    samples_num = max_train_batches * data.shape[0] 
                    
                estimated_time = samples_num /step_time    
                time_since_start = int(time.perf_counter() - start_time)
                formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
                str_ = f'Training time: {formatted_time}, Epoch: [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{max_train_batches}], Loss: {epoch_loss / (batch_idx + 1):.6f}, Instant loss:{loss.item():.6f}, Iter time: {step_time:.0f} smpls/sec Est.time: {estimated_time:.2f} sec'
        
                print(str_, end='\r', flush=True)
                
                # if batch_idx >= max_train_batches - 1:
                #     break  

            #add weights and biases to tensorboard
            if epoch % 5 == 0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        writer.add_histogram(f'weights/{name}', param, epoch)
                    elif 'bias' in name:
                        writer.add_histogram(f'biases/{name}', param, epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad, epoch)
                    

            print(' '*len(str_), end='\r', flush=True)

            # Test the model
            
            model.eval()
            vepoch_loss = 0
            with torch.inference_mode():
                correct = 0
                total = 0
                for vbatch_idx, vdata, vlabels in val_loader:
                    
                    vdata = vdata.to(device)
                    vlabels = vlabels.to(device)
                    #with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                        # Forward pass
                    voutputs = model(vdata)
                    #loss calculation
                    vloss = loss_fn(voutputs.view(-1, vocab_size), vlabels.view(-1))
                    last_redicted_candle = voutputs.softmax(dim=1).argmax(dim=1)
                    last_actual_candle = vlabels
                    correct += (last_redicted_candle == last_actual_candle).sum().item()
                    total += last_actual_candle.size(0)
                    accuracy = correct / total * 100
                    
                    vepoch_loss += vloss.item()

                    #calculate accuracy
                    
                    time_since_start = int(time.perf_counter() - start_time)
                    formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
                    print(f'Training time: {formatted_time}, Epoch [{epoch + 1}/{num_epochs}], '
                          f'Step [{vbatch_idx + 1}/{max_val_batches}], Validation Loss: {vepoch_loss/(vbatch_idx +1):.6f}, '
                          f'Validation accuracy: {accuracy:.2f}%', end='\r', flush=True)
                    
                    # if vbatch_idx >= max_val_batches - 1:
                    #     break


            
            time_since_start = int(time.perf_counter() - start_time)
            formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
            if epoch_loss / max_train_batches < best_loss:
                best_loss = epoch_loss / max_train_batches
                best_loss_mark = RED
            else:
                best_loss_mark = RESET
            if vepoch_loss / max_val_batches < best_vloss:
                best_vloss_mark = RED
            else:
                best_vloss_mark = RESET
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_mark = RED
            else:
                best_accuracy_mark = RESET

            lr = next(iter(optimizer.param_groups))['lr']
            print(f'Training time: {formatted_time}, Epoch [{epoch + 1}/{num_epochs}], '
                  f'Train Loss: {best_loss_mark}{epoch_loss / max_train_batches:.6f}{RESET}, Validation Loss: {best_vloss_mark}{vepoch_loss / max_val_batches:,.6f}{RESET}, '
                  f'Validation accuracy: {best_accuracy_mark}{accuracy:.2f}%{RESET}, Learning rate: {lr:.2e} ', end='\n', flush=True)
            writer.add_scalars('Loss', {'Train': epoch_loss / max_train_batches, 'Test': vepoch_loss / max_val_batches}, epoch)
            writer.add_scalar('Accuracy', accuracy, epoch)
            
            # def write_predictions(writer, model, vdata, voutputs, epoch, index_to_candle):
            #     predicted_seq = test_seq_generator(model, vdata[:1], 60)
            #     write_charts_to_TB('Generated sequence',writer, predicted_seq.cpu(), epoch, index_to_candle)
            #     vout_tockens = voutputs.softmax(dim=1).argmax(dim=1)
            #     write_charts_to_TB('Predicted sequence',writer, vout_tockens[:60].cpu(), epoch, index_to_candle)


            # #write charts to tensorboard
            # if epoch % 5 == 0:
            #     write_predictions(writer, model, vdata, voutputs, epoch, index_to_candle)
    
                
            # if epoch == 0:
            #     write_charts_to_TB('Target sequence',writer, vlabels[:60].cpu(), epoch, index_to_candle)
                
                
            
            
            # Save the model checkpoint if validation loss is less than best validation loss
            if vepoch_loss/max_val_batches < best_vloss:
                best_vloss = vepoch_loss/max_val_batches
                store_model(model, optimizer, scaler, best_model_path, epoch, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss, best_loss, best_accuracy, writer)

                lr = next(iter(optimizer.param_groups))['lr']
                wd = next(iter(optimizer.param_groups))['weight_decay']
                time_since_start = int(time.perf_counter() - start_time)
                formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'
                print(f"Training time: {formatted_time}, The best model saved at epoch {epoch+1} with validation loss {best_vloss_mark}{vepoch_loss/max_val_batches:.6f}{RESET} Learning rate: {lr:.2e} Weight decay: {wd:.2e} ")
            
            writer.flush()

    except KeyboardInterrupt:
        print(f"Interrupted at epoch {epoch}")
    finally:
        time_since_start = int(time.perf_counter() - start_time)
        formatted_time = f'{time_since_start//3600:02d}:{(time_since_start//60)%60:02d}:{time_since_start%60:02d}'

        store_model(model, optimizer, scaler, last_model_path, epoch, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss, best_loss, best_accuracy, writer)
        writer.close()
        print(f'Training time: {formatted_time}, The last model saved at epoch {epoch+1} with validation loss {best_vloss:.6f}')
        print('Training completed.')

def main():
    
    #load data
    batch_size = 1024
    seq_len = 64
    vocab_size, train_inputs, val_inputs, train_targets, val_targets, index_to_candle = data_prepare(candle_seq_len=seq_len+1)

    
    
    continue_training = False
    is_best = False

    

    global max_train_batches 
    global max_val_batches
    max_train_batches = (train_inputs.shape[0] + batch_size - 1) // batch_size //400
    max_val_batches = (val_inputs.shape[0] + batch_size - 1) // batch_size  //100

    train_loader = BatchIterator(batch_size, max_train_batches, train_inputs, train_targets, shuffle=True)
    val_loader = BatchIterator(batch_size, max_val_batches, val_inputs, val_targets, shuffle=False)

    global best_model_path
    global last_model_path
    global log_dir
    best_model_path = './models/nq-llm_decoder_bpe100_best.pth'
    last_model_path = './models/nq-llm_decoder_bpe100_last.pth'
    log_dir = 'runs/nq_llm_decoder_bpe100'

    #initialize hyperparameters
    hp = {'start_epoch':0, 
            'num_epochs':10000,
            'warmup_steps':200, 
            'vocab_size': vocab_size, 
            'embed_dim':64, 
            'num_heads':8, 
            'num_layers':4, 
            'dropout':0.1, 
            'ff_mult':4,
            'seq_len': seq_len, 
            'best_vloss':float('inf'),
            'best_loss':float('inf'),
            'best_accuracy':0
           }

    #initialize model, loss function and optimizer
    model, device = init_model(hp['vocab_size'], hp['seq_len'], hp['embed_dim'], hp['num_heads'], hp['num_layers'], hp['dropout'], hp['ff_mult'])
    scaler = None #torch.cuda.amp.GradScaler(enabled=True)
    


    
    optimizer = optim.AdamW(model.parameters(), lr=1e-6) 
   
    if continue_training:
        #load model from checkpoint
        load_path = best_model_path if is_best else last_model_path

        print(f'Loading model from checkpoint {load_path}')
      
        epoch, model, optimizer, vocab_size, embed_dim, num_heads, num_layers, dropout, best_vloss, best_loss, best_accuracy, writers = restore_model(model, optimizer,scaler, load_path,device)
        print(f'The best model loaded from checkpoint. Epoch: {epoch+1}, Best train loss: {best_loss:.6f}, Best validation loss: {best_vloss:.6f}, Best accuracy: {best_accuracy:.2f}%')

        #update hyperparameters
        hp['start_epoch'] = epoch + 1 if is_best else epoch
        hp['best_vloss'] = best_vloss
        hp['best_loss'] = best_loss
        hp['best_accuracy'] = best_accuracy
        purge_step = epoch if is_best else None
    else:
        purge_step = None
        writers = None

    print(f'Model hyperparameters are:\nVocab_size: {hp["vocab_size"]},\tSeq_len: {hp["seq_len"]},\tEmbed_dim: {hp["embed_dim"]}\nNum_heads: {hp["num_heads"]},\tNum_layers: {hp["num_layers"]},\tDropout: {hp["dropout"]}, FF_mult: {hp["ff_mult"]}')
    
    count_parameters(model)
    token_weights = (torch.arange(1, vocab_size+1) ** 0.2)
    token_weights = token_weights / token_weights.sum() * vocab_size
    token_weights = token_weights.to(device)
    loss_fn = nn.CrossEntropyLoss(weight=token_weights, label_smoothing=0.05)

    # #set learning rate and weight decay
    # optimizer = set_learning_rate(optimizer, hp['start_epoch'], hp['embed_dim'], weight_decay=0)

    #initialize writer
    
    writer = init_writer(continue_training=continue_training, purge_step=purge_step, writers=writers)


    
    #train model
    train(model, optimizer, scaler, loss_fn, train_loader, val_loader, writer, device, index_to_candle, hp)
    
if __name__ == '__main__':
    main()



        

