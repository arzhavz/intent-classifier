import torch
import torch.nn as nn

import time
import datetime
import colorama
from datetime import timedelta


class Trainer:
	def __init__(self, model, loader, optimizer, epochs):
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model = model
		self.loader = loader
		self.optimizer = optimizer
		self.epochs = epochs
		
	def fit(self):
		prev_loss = float("inf")
		start_time = time.monotonic()

		print(
			"{}[{}]{} {}Training Intents Classifier model using {} {}".format(
				colorama.Fore.MAGENTA,
				datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				colorama.Style.RESET_ALL,
				colorama.Fore.GREEN,
				self.device,
				colorama.Style.RESET_ALL,
			)
		)

		self.model.train()
		self.model.to(self.device)

		criterion = nn.CrossEntropyLoss()

		for epoch in range(self.epochs):
			for words, labels in self.loader:
				words, labels = words.to(self.device), labels.to(self.device)
				output = self.model(words)
				loss = criterion(output, labels)

				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			elapsed_time = time.monotonic() - start_time
			elapsed_formatted = str(timedelta(seconds=int(elapsed_time)))

			loss_str = "{:.12f}".format(loss.item())
			if loss.item() <= prev_loss:
				loss_color = colorama.Fore.YELLOW + loss_str + colorama.Style.RESET_ALL
			else:
				loss_color = colorama.Fore.RED + loss_str + colorama.Style.RESET_ALL

			if (epoch + 1) % (self.epochs // 10) == 0:
				print(
					"{}[{}]{} {}Epoch:{} {}{:04d}/{:04d}{} {}Loss:{} {} {}{}{}".format(
						colorama.Fore.MAGENTA,
						datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						colorama.Style.RESET_ALL,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						colorama.Fore.BLUE,
						epoch + 1,
						self.epochs,
						colorama.Style.RESET_ALL,
						colorama.Fore.CYAN,
						colorama.Style.RESET_ALL,
						loss_color,
						colorama.Fore.GREEN,
						elapsed_formatted,
						colorama.Style.RESET_ALL,
					)
				)
				prev_loss = loss.item()