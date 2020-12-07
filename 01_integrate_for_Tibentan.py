#!/usr/bin/env python
# coding: utf-8

# In[6]:


import sentencepiece as spm
import pandas as pd
from tokenizers import SentencePieceBPETokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SparseAdam
from transformers import (
    T5Model, 
    T5ForConditionalGeneration, 
    AdamW,
)
import pytorch_lightning as pl
import time
from datetime import datetime
import textwrap

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(f'device = {device}')


# In[3]:


boDataForTokenizerPath = './data/boTokenData.txt'
enDataForTokenizerPath = './data/enTokenData.txt'

boDataPath = './data/train.bo'
enDataPath = './data/train.en'

boTokenizerPath = './preProcessing/bo.model'
enTokenizerPath = './preProcessing/en.model'


# ## Load data 
# 

# In[7]:


boFile = open(boDataPath, 'r', encoding = 'utf-8')
enFile = open(enDataPath, 'r', encoding = 'utf-8')

dataMatrix = []

while True: 
    boLine = boFile.readline().strip()
    enLine = enFile.readline().strip()
    if not boLine or not enLine: 
        break 
    dataMatrix.append([boLine, enLine])
  
# Create pandas dataframe 
df = pd.DataFrame(dataMatrix, columns = ['bo', 'en'])
df


# In[8]:


boTextsAll = df['bo'].tolist()
enTextsAll = df['en'].tolist()


# ## Tokenizers for Tibetan and English
# 
# The code cell below uses Google SentencePiece tokenizer, but we cannot yet figure out how to truncate and pad the tokenizations to the same length, nor the special characters. Save the code but we will not use it for now. 

# In[9]:


'''
## Ignore this cell
'''

# Load tokenizers that are already trained
boTokenizer = spm.SentencePieceProcessor(model_file=boTokenizerPath)
enTokenizer = spm.SentencePieceProcessor(model_file=enTokenizerPath)

# Verify for Tibetan
print(boTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
print(boTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'བཀ྄ྲ་ཤིས་བདེ་ལེགས།'], out_type=int))
print(boTokenizer.decode([4149, 306, 6, 245, 4660, 748]))
print(boTokenizer.decode(['▁ངའི་', 'མིང་', 'ལ་', 'བསྟན་', 'སྒྲོལ་མ་', 'ཟེར་']))
print('Vocab size of Tibetan Tokenizer:', boTokenizer.get_piece_size())

# Verify for English
print(enTokenizer.encode(["My name isn't Tenzin Dolma Gyalpo"], out_type=str))
print(enTokenizer.encode(['My name is Tenzin Dolma Gyalpo', 'Hello'], out_type=int))
print(enTokenizer.decode([[8803, 180, 12, 5519, 15171, 17894], [887, 21491]]))
print('Vocab size of English Tokenizer:', enTokenizer.get_piece_size())


# Instead, we use huggingface tokenizer now. The bad news is we can no longer find the right API. 

# In[10]:


boTokenizer = SentencePieceBPETokenizer()
boTokenizer.train([boDataForTokenizerPath], vocab_size = 32000, special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

enTokenizer = SentencePieceBPETokenizer()
enTokenizer.train([enDataForTokenizerPath], vocab_size = 25000, special_tokens = ['<s>', '<pad>', '</s>', '<unk>', '<mask>'])

print('Tibetan tokenizer vocab size:', boTokenizer.get_vocab_size())
print('English tokenizer vocab size:', enTokenizer.get_vocab_size())


# In[7]:


# Verify for Tibetan
outputs = boTokenizer.encode_batch(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'བཀ྄ྲ་ཤིས་བདེ་ལེགས།'])

for output in outputs: 
    print('ids:', output.ids)
    print('token:', output.tokens)
    print('mask:', output.attention_mask)


# In[11]:


# Verify for English
outputs = enTokenizer.encode_batch(["My name isn't Tenzin Dolma Gyalpo", 'My name is Tenzin Dolma Gyalpo', 'Hello'])

for output in outputs: 
    print('ids:', output.ids)
    print('token:', output.tokens)
    print('mask:', output.attention_mask)


# ## Pytorch `Dataset`

# In[9]:


class MyDataset(Dataset): 
    def __init__(self, boTexts, enTexts, boTokenizer, enTokenizer, boMaxLen, enMaxLen): 
        super().__init__()
        self.boTexts = boTexts
        self.enTexts = enTexts
        self.boTokenizer = boTokenizer
        self.enTokenizer = enTokenizer
        
        # Enable padding and truncation 
        self.boTokenizer.enable_padding(length = boMaxLen)
        self.boTokenizer.enable_truncation(max_length = boMaxLen)
        self.enTokenizer.enable_padding(length = enMaxLen)
        self.enTokenizer.enable_truncation(max_length = enMaxLen)
        
    ''' Return the size of dataset '''
    def __len__(self): 
        return len(self.boTexts)
    
    '''
    -- The routine for querying one data entry 
    -- The index of must be specified as an argument
    -- Return a dictionary 
    '''
    def __getitem__(self, idx): 
        # Apply tokenizer
        boOutputs = self.boTokenizer.encode(self.boTexts[idx])
        enOutputs = self.enTokenizer.encode(self.enTexts[idx])
        
        # Get numerical tokens 
        boEncoding = boOutputs.ids
        enEncoding = enOutputs.ids
        
        # Get attention mask 
        boMask = boOutputs.attention_mask
        enMask = enOutputs.attention_mask
        
        return {
            'source_ids': torch.tensor(boEncoding), 
            'source_mask': torch.tensor(boMask), 
            'target_ids': torch.tensor(enEncoding), 
            'target_mask': torch.tensor(enMask)
        }


# ## Define model class

# In[10]:


class T5FineTuner(pl.LightningModule): 
    ''' Part 1: Define the architecture of model in init '''
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams['pretrainedModelName'], 
            return_dict = True    # I set return_dict true so that outputs  are presented as dictionaries
        )
        self.boTokenizer = hparams['boTokenizer']
        self.enTokenizer = hparams['enTokenizer']
        self.hparams = hparams
        
        
    ''' Part 2: Define the forward propagation '''
    def forward(self, input_ids, attention_mask = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):  
        return self.model(
            input_ids, 
            attention_mask = attention_mask, 
            decoder_input_ids = decoder_input_ids, 
            decoder_attention_mask = decoder_attention_mask, 
            labels = labels
        )
    
    
    ''' Part 3: Configure optimizer and scheduler '''
    def configure_optimizers(self): 
        # I have no idea why to configure parameter this way 
        optimizer_grouped_parameters = [
            {
                # parameter with weight decay 
                'params': [param for name, param in model.named_parameters() if ('bias' not in name and 'LayerNorm.weight' not in name)], 
                'weight_decay': self.hparams['weight_decay'], 
            }, 
            {
                'params': [param for name, param in model.named_parameters() if ('bias' in name or 'LayerNorm.weight' in name)], 
                'weight_decay': 0.0, 
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams['learning_rate'])
        return optimizer
    
    
    ''' Part 4.1: Training logic '''
    def training_step(self, batch, batch_idx):         
        loss = self._step(batch)
        self.log('train_loss', loss)
        return loss 
    
    
    def _step(self, batch): 
        labels = batch['target_ids'] 
        labels[labels[:, ] == 0] = -100    # Change the pad id from 0 to -100, but I do not know why the example chooses to do so. I will comment it out for now
        
        outputs = self(
            input_ids = batch['source_ids'], 
            attention_mask = batch['source_mask'], 
            labels = labels, 
            decoder_attention_mask = batch['target_mask']
        )
        
        return outputs.loss

    
    ''' Part 4.2: Validation logic '''
    def validation_step(self, batch, batch_idx):        
        loss = self._step(batch)
        self.log('val_loss', loss)
        
        
    ''' Part 4.3: Test logic '''
    def test_step(self, batch, batch_idx): 
        loss = self._step(batch)
        self.log('test_loss', loss)
    
    
    ''' Part 5: Data loaders '''
    def _get_dataloader(self, start_idx, end_idx): 
        dataset = MyDataset(
            boTexts = boTextsAll[start_idx:end_idx], 
            enTexts = enTextsAll[start_idx:end_idx], 
            boTokenizer = self.hparams['boTokenizer'], 
            enTokenizer = self.hparams['enTokenizer'], 
            boMaxLen = self.hparams['max_input_len'], 
            enMaxLen = self.hparams['max_output_len']
        )
        
        return DataLoader(dataset, batch_size = hparams['batch_size'])
    
    
    def train_dataloader(self): 
        start_idx = 0
        end_idx = int(self.hparams['train_percentage'] * len(boTextsAll))
        return self._get_dataloader(start_idx, end_idx)
        
    
    def val_dataloader(self): 
        start_idx = int(self.hparams['train_percentage'] * len(boTextsAll))
        end_idx = int((self.hparams['train_percentage'] + self.hparams['val_percentage']) * len(boTextsAll))
        return self._get_dataloader(start_idx, end_idx)
    
    
    def test_dataloader(self): 
        start_idx = int((self.hparams['train_percentage'] + self.hparams['val_percentage']) * len(boTextsAll))
        end_idx = len(boTextsAll)
        return self._get_dataloader(start_idx, end_idx)


# In[11]:


hparams = {
    'boTokenizer': boTokenizer,
    'enTokenizer': enTokenizer,
    'pretrainedModelName': 't5-small', 
    'train_percentage': 0.85, 
    'val_percentage': 0.13, 
    'learning_rate': 3e-4, 
    'max_input_len': 100, 
    'max_output_len': 100, 
    'batch_size': 8, 
    'num_train_epochs': 2, 
    'num_gpu': 1, 
    'weight_decay': 0, 
}


# ## Training

# In[ ]:


torch.cuda.empty_cache()

train_params = dict(
    gpus = hparams['num_gpu'], 
    max_epochs = hparams['num_train_epochs'], 
    progress_bar_refresh_rate = 20, 
)

model = T5FineTuner(hparams)

trainer = pl.Trainer(**train_params)

trainer.fit(model)

# Save model for later use
now = datetime.now()
trainer.save_checkpoint('01_t5simple_bo_en' + now.strftime("%Y-%d-%m-%Y--%H=%M=%S") + '.ckpt')

trainer.test()


# In[ ]:




