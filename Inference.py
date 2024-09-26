"""
This py file is for generating the inference for the images based on the query.
Model used: MiniCPM-V2 from openbmb Huggingface
Model parameters: 3.43 Billion
Data: Amazon Hackathon test dataset
Server: https://jarvislabs.ai/templates/pytorch
Cloud Storage: 100GB
CPU specification: 7 CPUs, 32GB RAM
GPU specification: RTX A5000, 24GB VRAM
Inference time total: 18 hours 27 minutes 13 seconds

Libraries:
timm peft
Pillow==10.1.0
torch==2.1.2
torchvision==0.16.2
transformers==4.40.0 
sentencepiece==0.1.99 
accelerate==0.30.1 
bitsandbytes==0.43.1
"""

# Load the libraries

import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import time
import os

# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2', trust_remote_code=True)
model = model.to(device='cuda')
model.eval()

# Load the test data csv file provided
# keep the batch size to 64

df = pd.read_csv('test.csv')
df['entity_value'] = ''
image_folder = 'test_images/Test Images'
batch_size = 64
pbar = tqdm(range(0, len(df), batch_size), desc='Processing images') # generates the batch start indices 

def create_batch(df, image_folder, batch_start, batch_size):
    """
    description: generates the batched dataframe and batch images
    args: df -> original dataframe, image_folder -> location of the images, batch_start -> starting index of the batch, batch_size -> size of each batch (64 in our case)
    returns: batch_df, batch_images, batch_entity_names (name of the entites in the query in batch)
    """
    batch_df = df.iloc[batch_start:batch_start + batch_size]
    batch_images = []
    batch_entity_names = []
    for index, row in batch_df.iterrows():
        image_id = row['image_id']
        entity_name = row['entity_name']
        image_path = os.path.join(image_folder, f'{image_id}.jpg')
        image = Image.open(image_path).convert('RGB')
        batch_images.append(image)
        batch_entity_names.append(entity_name)
    return batch_df, batch_images, batch_entity_names

def create_messages(batch_entity_names):
    """
    description: generates the prompt question based on the entity name (which is to be answered)
    args: batch_entity_names -> entity names in the batches 
    returns: prompt questions in batches
    """
    batch_msgs = []
    for entity_name in batch_entity_names:
        question = f'what is the {entity_name} of product in the image'
        msgs = [{'role': 'user', 'content': question}]
        batch_msgs.append(msgs)
    return batch_msgs

def get_responses(model, tokenizer, batch_images, batch_msgs):
    """
    description: generates the result using the model
    args: model -> MiniCPM-V2 LM, tokenizer -> tokenizer for generating the tokens in prompt, batch_images -> images in batches, batch_msgs -> prompt questions in batches
    returns: result of the prompt question
    """
    batch_res = []
    with torch.no_grad():
        for image, msgs in zip(batch_images, batch_msgs):
            res, context, _ = model.chat(
                image=image,
                msgs=msgs,
                context=None,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.7
            )
            batch_res.append(res)
    return batch_res

def store_results(df, batch_df, batch_res):
    """
    description: stores the result in the dataframe at the right index of the entity_name
    args: df -> original dataframe, batch_df -> batch dataframe, batch_res -> result in batches
    returns: the modified df
    """
    for index, res in zip(batch_df.index, batch_res):
        df.at[index, 'entity_value'] = res
    return df

def process_batch(model, tokenizer, df, image_folder, batch_start, batch_size, pbar):
    """
    description: processes the batches 
    args: model ->  MiniCPM-V2 LM, tokenizer -> tokenizer for generating the tokens in prompt, df -> original dataframe, image_folder -> location of the images, batch_start -> batch index, pbar -> instance of pbar created earlier
    returns: updated df
    """
    batch_df, batch_images, batch_entity_names = create_batch(df, image_folder, batch_start, batch_size)
    batch_msgs = create_messages(batch_entity_names)
    batch_res = get_responses(model, tokenizer, batch_images, batch_msgs)
    df = store_results(df, batch_df, batch_res)
    torch.cuda.empty_cache()
    pbar.update(1)
    return df

# Initiates the whole inference process

for batch_start in pbar:
    df = process_batch(model, tokenizer, df, image_folder, batch_start, batch_size, pbar)

# Save the final df
df.to_csv("test_result.csv",index=False)