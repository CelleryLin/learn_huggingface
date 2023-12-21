'''
This code summarize an article.
'''

from transformers import pipeline
import os

# %% Using pipelines

pipe = pipeline("summarization", model="facebook/bart-large-cnn")

file = open('./article.txt',mode='r')
article = file.read()

res = pipe(article)
print(res)

# %% Save
pipe.save_pretrained('./model/bart-large-cnn/')

# %% Load pretrained model from file
model_path = './model/bart-large-cnn'
pipe2 = pipeline("summarization", model=model_path, tokenizer=model_path)

file = open('./article.txt',mode='r')
article = file.read()

res = pipe2(article)
print(res)

# %%
