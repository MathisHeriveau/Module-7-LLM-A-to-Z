# Import

import os,sys
import ipdb # Debug
from tqdm import tqdm
from datetime import datetime
import platform, shutil # Detect platform type
import requests,zipfile, io

import torch
import torch.nn as nn
from torch.nn import functional as F

import sentencepiece as spm # Tokenizer 

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.cuda.empty_cache()

# files_url = "https://ideami.com/llm_train"
# print('Downloading files Python')
# response = requests.get(files_url)
# zipfile.ZipFile(io.BytesIO(response.content)).extractall(".\Module 7 - LLM - A to Z\LLM-Mastery-hand")



# ****************************************************** #
#                       PARAMETER                        #
# ****************************************************** #

path = "Module 7 - LLM - A to Z/LLM-Mastery-hand"

# *********************** #
#      ARCHITECTURE       #
# *********************** #
batch_size = 64 # 8 -> 128 Nombre de process en parallele
context = 512 # L'ia va prendre attention a 512 token, voir les connections entre : 1 = Incohérant, 100_000 excellent mais trop long à entrainer
embed_size = 384 # Each token is transformed into a vector of 384 nb (Embedding)

# Une archiecture LLM est un trasnformatteur, qui est composer de layers/de bloc et a 
# l'interieur de chaque on retrouve la :
#   - Communication : un mécanisme qui apprend les differents relations entre token
#   - Calcul : une couche qui fournit un traitement complexe pour le réseau
# Et a l'interieur du mécanisme d'attention de chaque bloc, on a un nombre de head (multi head attention mechanisme)
n_layers = 7
n_heads = 7

BIAS = True # Activation des BIAS (parametre additionnel)


# *********************** #
#    HYPER PARAMETER      #
# *********************** #
lr = 0.0001
dropout = 0.05 # Régularisation de l'overfit pour apprendre plus de robust features
weight_decay = 0.01 # Régulation des weight pour qu'elle soit petite et pas faire de over-reliance
# Grad vanish : perte des information / impossible d'apprendre
# Grad explosion : Trop large, et devient instable
grad_clip = 1. # Stabilise les gradients a 1. 


# *********************** #
#    TRAINING PARAM       #
# *********************** #
train_iters = 100_000
eval_interval = 50 # Savoir s'il fait un over-fit
eval_iters = 10 # Quand on fait l'évaluation du loss du batch (average of 10 batch)
compile = True # A verifier si on le mets pas en Flse
checkpoint_dir ='models'
checkpoint_fn='lastest.pt'
checkpoint_load_fn='lastest.pt'
dtype = torch.bfloat16


# *********************** #
#           MODE          #
# *********************** #
inference = False # Inference : Donne un new input au NN pour produire un new tokens qui suit l'input 



# *********************** #
#         DEVICE          #
# *********************** #
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'You will using {device}')



# ****************************************************** #
#                       LOGGING                          #
# ****************************************************** #
wandb_log = True
wandb_project = "llm_test"
wandb_run_name = "llm1_"+datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name= wandb_run_name)
    
    
# ****************************************************** #
#                       TOKENIZER                        #
# ****************************************************** #

with open(f"{path}/wiki.txt", "r", encoding='utf-8') as f:
    text = f.read()
    
sp = spm.SentencePieceProcessor(model_file=f'{path}/wiki_tokenizer.model')

vocab_size = sp.get_piece_size()
print(f"Tokenizer vocab_size : {vocab_size}")



    
# ****************************************************** #
#                       ENCODE                           #
# ****************************************************** #
# TEXT - > TOKEN
encode = lambda s: sp.Encode(s)
    
# ****************************************************** #
#                       DECODE                           #
# ****************************************************** #
# TOKEN - > TEXT
decode = lambda l: sp.Decode(l)

if os.path.exists(f"{path}/encoded_data.pt"):
    print(f"Load encode data from {path}/encoded_data.pt")
    data = torch.load(f"{path}/encoded_data.pt")
else: 
    data = torch.tensor(encode(text), dtype=torch.long)
    torch.save(data,f"{path}/encoded_data.pt")

data_size = len(data)
spl = int(0.9*data_size)
train_data= data[:spl] # Train sur 90% des données
val_data=data[spl:] # Test sur 10% des données

print(f"Total data: {data_size/1e6:.2f} Million | Training: {len(train_data)/1e6:.2f} Million | Validation: {len(val_data)/1e6:.2f} Milion")


# ****************************************************** #
#                       BATCH                            #
# ****************************************************** #
def get_batch(split): # En gros, ca prend une phrase au hasard : x = "Virginia. She was named Maria, but people" y = ". She was named Maria, but people called"
    # BS = Batch size 8 / SL = Sequence or Context Lenght : 512
    data = train_data if split=="train" else val_data
    inds = torch.randint(len(data)-context, (batch_size,))
    x = torch.stack([data[i: i+context] for i in inds])  # (8,215)
    y = torch.stack([data[i+1: i+context+1] for i in inds]) # (8,215)
    x,y = x.to(device), y.to(device)
    return x,y


# ****************************************************** #
#                      LLM MODEL                         #
# ****************************************************** #
# 19 Milion parameters with the default conf, in 1 single GPU
# With 8 batch size, should require 4 GB of GPU memory
# With 128 batch size, should require 24 GB of GPU memory
# Small dataset and model, result will ne limited but enough to
# demonstrate good improvement during the training and unersytan all the
#main technology involved in building LLMS

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size) # 4096 x 384
        self.positions = nn.Embedding(context, embed_size) # 512 x 384 
        self.blocks = nn.Sequential(*[torch.Block(n_heads) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size)
        self.final_linear = nn.Linear(embed_size, vocab_size, bias=BIAS) # 384 x 4096
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, input, targets=None):
        loss = None
        BS,SL = input.shape
        emb = self.embeddings(input)
        pos = self.positions(torch.arange(SL,device=device))
        x = emb + pos 
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.final_linear(x)
        
        if targets is not None:
            BS, SL, VS = logits.shape
            logits = logits.view(BS*SL,VS)
            targets = targets.view(BS*SL)
            loss = F.cross_entropy(logits, targets)
            
            counts = logits.exp()
            prob = counts / counts.sum(-1, keepdim=True)
            loss2 = - prob[torch.arange(BS*SL), targets].log().mean()
            
            if (not torch.allclose(loss,loss2)):
                print(f"[Loss Diff] Pytorch:{loss.item()} Manuel: {loss2.item()}")
            
        return logits, loss, loss2
    
    def generate (self, input ,max=500): # Max est la valeur max de l"output
        for _ in range(max):
            input = input[:,-context:] # (1, input leght until max of SL)
            logits, _ = self(input)
            logits = logits[:,-1,:] # Last prediction
            probs = F.softmax(logits, dim=1)
            next = torch.multinomial(probs,num_samples=1)
            input = torch.cat((input,next),dim=1)
        return input
            



# ****************************************************** #
#                      MAIN                              #
# ****************************************************** #

x, y = get_batch("train")
model = GPT()    
model = model.to(dtype)    
model = model.to(device)    
            
logits, loss, loss2 = model(x,y)
print(logits.shape)


@torch.no_grad()
def generate_sample(input):
    t1 = torch.tensor(encode(input), dtype=torch.long, device=device)
    t1 = t1[None, :]
    newgen = model.generate(t1, max=64)[0].tolist()
    result = decode(newgen)
    print(f"result : {result}")

text10 = "je veux une phrase bien longue plus grande que 500 caractere, mais pas non plus trop ongue parce que ca va etre long decrire 500 careactere comme ca, tu penses pas ? moi je pense en tout cas, les dissertation c'est pas mon fort. Je vais parler de mr bean, c'est mieux ! Quel acteur Jonhy English ! En plus il a une voiture de malade a des milion, qu'il a lui meme casser, ptdr la honte"
generate_sample(text10)
