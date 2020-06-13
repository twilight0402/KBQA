import torch

from model.LSTM.tool import build_and_cache_dataset

data_dir = r"E:\Workspaces\Python\KG\QA_healty39\data"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

fields, train_dataset = build_and_cache_dataset()
print(type(train_dataset))
print(train_dataset)
train_dataset = train_dataset[:100]
print(type(train_dataset))
print(train_dataset)