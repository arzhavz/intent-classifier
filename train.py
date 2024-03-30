import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.models import FireflyMoE
from model.datasets import Intents
from lib.preprocess import (
	prepare_data,
	bag_of_words,
	tokenize,
	stem
)
from lib.trainer import Trainer
from lib.utils import (
	number_formatter,
	calculate_params
)


def SaveData(file_path, *args, **kwargs):
	data = {
		"FireflyMoE": {
			"state_dict": kwargs.get("model").state_dict(),
			"n_vocab": kwargs.get("n_vocab"),
			"n_hidden": kwargs.get("n_hidden"),
			"n_classes": kwargs.get("n_classes"),
			"n_expert": kwargs.get("n_expert"),
			"all_words": kwargs.get("all_words"),
			"all_tags": kwargs.get("all_tags"),
			"intents": kwargs.get("intents"),
		}
	}

	torch.save(data, file_path)
	print(
		"\n{}Training complete! Model saved to{} {}{}{}".format(
			colorama.Fore.GREEN,
			colorama.Style.RESET_ALL,
			colorama.Fore.MAGENTA,
			file_path,
			colorama.Style.RESET_ALL,
		)
	)


def main():
	parser = argparse.ArgumentParser(description='Konfigurasi data training untuk Model AI')

	parser.add_argument('--lr', type=float, default=0.001,
						help='Learning rate yang digunakan saat pelatihan model')
	parser.add_argument('--dataset', type=str, default='dataset',
						help='Nama file dataset pada folder data/')
	parser.add_argument('--batch_size', type=int, default=8,
						help='Ukuran batch saat pelatihan model')
	parser.add_argument('--n_epochs', type=int, default=1000,
						help='Jumlah epoch pada Intents Classifier')
	parser.add_argument('--n_hidden', type=int, default=32,
						help='Jumlah hidden layer pada Intents Classifier')
	parser.add_argument('--n_expert', type=int, default=4,
						help='Jumlah expert untuk model Intents Classifier')
	
	args = parser.parse_args()

	DATASET = args.dataset
	BATCH_SIZE = args.batch_size
	LR = args.lr
	N_EPOCHS = args.n_epochs
	N_HIDDEN = args.n_hidden
	N_EXPERT = args.n_expert

	x, y, all_words, tags, intents = prepare_data(f"data/{DATASET}.json")
	dataset = Intents(x, y)
	dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
	model = FireflyMoE(len(x[0]), N_HIDDEN, len(tags), N_EXPERT)
	optimizer = torch.optim.Adam(model.parameters(), lr=LR)

	trainer = Trainer(model, dataloader, optimizer, N_EPOCHS)
	trainer.fit()
	
	_, params = calculate_params(model)

	file_path = f"checkpoint-{number_formatter(params)}.moe"
	SaveData(
		file_path,
		model=model,
		n_vocab=len(x[0]),
		n_hidden=N_HIDDEN,
		n_classes=len(tags),
		n_expert=N_EXPERT,
		all_words=all_words,
		all_tags=tags,
		intents=intents
	)


if __name__ == "__main__":
	import colorama
	colorama.init()
	main()
