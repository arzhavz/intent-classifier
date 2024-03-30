import torch
from torch.utils.data import Dataset


class Intents(Dataset):
	def __init__(self, X_train, y_train):
		self.n_samples = len(X_train)
		self.x_data = torch.tensor(X_train, dtype=torch.float32)
		self.y_data = torch.tensor(y_train, dtype=torch.long)
		
	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.n_samples
	