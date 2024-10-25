import torch
from torch.utils.data import Dataset

import json
import random
from collections import namedtuple

data_element = namedtuple(
    "data_element",
    ["input_token_tensor", "input_masking", "output_token_tensor", "output_masking"]
)

class Ruozhi_dataset(Dataset):
    
    def __init__(self, data_path, tokenizer, sequence_length, select_type='all'):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.complete_content = self.read_json(data_path)
        self.data_list = []
        if select_type == 'all':
            for element in self.complete_content:
                input_data = element['original_data']
                input_token = self.tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.sequence_length)
                for key, value in element.items():
                    if key != 'data_id' and key != 'original_data':
                        output_data = value
                        output_token = self.tokenizer(output_data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.sequence_length)
                        instance_data = data_element(
                            input_token_tensor = input_token.input_ids,
                            input_masking = input_token.attention_mask,
                            output_token_tensor = output_token.input_ids,
                            output_masking = output_token.attention_mask
                        )
                        self.data_list.append(instance_data)
        elif select_type == 'single':
            for element in self.complete_content:
                output_list = []
                for key, value in element.items():
                    if key != 'data_id' and key != 'original_data':
                        output_list.append(value)
                input_data = element['original_data']
                input_token = self.tokenizer(input_data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.sequence_length)
                output_data = random.choice(output_list)
                output_token = self.tokenizer(output_data, return_tensors="pt", padding="max_length", truncation=True, max_length=self.sequence_length)
                instance_data = data_element(
                    input_token_tensor = input_token.input_ids,
                    input_masking = input_token.attention_mask,
                    output_token_tensor = output_token.input_ids,
                    output_masking = output_token.attention_mask
                )
                self.data_list.append(instance_data)
            
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            complete_content = json.load(f)
        return complete_content

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        element = self.data_list[idx]
        output_token = element.output_token_tensor
        output_token[element.output_masking == 0] = -100
        return (element.input_token_tensor, element.input_masking, output_token)


