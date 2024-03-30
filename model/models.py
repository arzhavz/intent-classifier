import torch
import torch.nn as nn
import torch.nn.functional as F
	

class Expert(nn.Module):
	""" 
	Name: Expert Layer
	Version: 0.0.1
	Architecture: Multi-Layer Perceptron (MLP)

	Parameters:
	- input_dim (int): Dimension of the input features
	- hidden_dim (int): Number of hidden units in the MLP
	- output_dim (int): Dimension of the output
	"""
	def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
		super(Expert, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		return x

class FireflyMoE(nn.Module):
	""" 
	Name: Firefly MoE
	Version: 0.0.1
	Architecture: Mixture of Experts (MoE) with Multi-Layer Perceptron (MLP) Experts

	Parameters:
	- n_vocab (int): Number of vocabulary
	- n_hidden (int): Number of hidden units in each expert MLP
	- n_classes (int): Number of output classes
	- n_experts (int): Number of experts in the MoE
	"""
	def __init__(self, n_vocab: int, n_hidden: int, n_classes: int, n_experts: int):
		super(FireflyMoE, self).__init__()
		self.n_experts = n_experts
		self.experts = nn.ModuleList([Expert(n_vocab, n_hidden, n_classes) for _ in range(n_experts)])
		self.gate_fc = nn.Linear(n_vocab, n_experts)

	def forward(self, x):
		gates = F.softmax(self.gate_fc(x), dim=1)
		expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
		weighted_outputs = torch.sum(gates.unsqueeze(2) * expert_outputs, dim=1)
		return weighted_outputs
