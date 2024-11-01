from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn
import torch.optim as optim

from ruozhi_dataset import Ruozhi_dataset
from torch.utils.data import DataLoader

import glob

dataset_dir = 'D:/info/program/NLP/asm/dataset/'

train_list = glob.glob(dataset_dir + 'train*.json')
val_list = glob.glob(dataset_dir + 'val*.json')

model_name = "D:/models/Qwen2.5-7B-Instruct"

# 创建模型实例
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# 取出对应的模型词嵌入
tokenizer = AutoTokenizer.from_pretrained(model_name)

for name, param in model.named_parameters():
    # split out the layer number
    name_l = name.split('.')
    layer_num = name_l[2] if name_l[1] == 'layers' else None
    
    if layer_num is not None:
        # frozen the lower layers to preserve the features
        # train the higher layers to fit into the specific downstream task
        layer_num = int(layer_num)
        if layer_num < 23:
            param.requires_grad = False
        elif layer_num >= 23:
            param.requires_grad = True


batch_size = 16
epochs = 10
lr = 5e-5
sequence_length = 512
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train1 = Ruozhi_dataset(train_list[0], tokenizer = tokenizer, sequence_length = sequence_length)
val1 = Ruozhi_dataset(val_list[0], tokenizer = tokenizer, sequence_length = sequence_length)

train_loader1 = DataLoader(train1, batch_size=batch_size, shuffle=True)
val_loader1 = DataLoader(val1, batch_size=len(val1), shuffle=True)


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)



model.to(device)
for epoch in range(1):
    for input_token, output_token in train_loader1:
        input_token = input_token.view(batch_size, sequence_length)
        output_token = output_token.view(batch_size, sequence_length)

        input_token = input_token.to(device)
        output_token = output_token.to(device)

        output = model(input_ids = input_token, labels = output_token)
        loss = output.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
    for input_token, output_token in val_loader1:
        with torch.no_grad():
            input_token = input_token.view(batch_size, sequence_length)
            output_token = output_token.view(batch_size, sequence_length)

            input_token = input_token.to(device)
            output_token = output_token.to(device)


            output = model(input_ids = input_token, labels = output_token)
            loss = output.loss

            print(f"Epoch {epoch + 1}, Validation Loss: {loss.item()}")



