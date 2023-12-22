'''
This code summarize an article.
'''

from transformers import pipeline
import os

# %% Using pipelines

# device = -1 => usnig cpu
# device >= 0 => cuda
pipe = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

file = open('./article.txt',mode='r')
article = file.read()

res = pipe(article)
print(res)

# %% Save
pipe.save_pretrained('./model/bart-large-cnn/')

# %% Load pretrained model from file
model_path = './model/bart-large-cnn'
pipe2 = pipeline("summarization", model=model_path, tokenizer=model_path, device=0) # cuda:0

file = open('./article.txt',mode='r')
article = file.read()

res = pipe2(article)
print(res)

# %% Using accelerate
import torch
from transformers import pipeline

pipe = pipeline(model="facebook/opt-1.3b", torch_dtype=torch.bfloat16, device=0)
output = pipe("This is a cool example!", do_sample=True, top_p=0.95)

# %%
