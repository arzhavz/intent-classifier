import torch

import os
import random
import colorama
from rich.pretty import pprint

from model.models import FireflyMoE
from lib.preprocess import bag_of_words, tokenize
from lib.utils import display, list_files_by_extension

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoints = list_files_by_extension(".moe")

FILE = input(f"Insert model's filename {checkpoints}: ")
#FILE = checkpoints[0]
MODEL = torch.load(FILE, map_location=device)
base_model = MODEL["FireflyMoE"]


class Chatbot:
	def __init__(self):
		self.base_model = FireflyMoE(
			base_model["n_vocab"], base_model["n_hidden"], base_model["n_classes"], base_model["n_expert"]
		).to(device)
		self.base_model.load_state_dict(base_model["state_dict"])
		self.base_model.eval()
		
	def predict(self, message):
		intents = base_model["intents"]
		sentence = tokenize(message)
		X = bag_of_words(sentence, base_model["all_words"])
		X = X.reshape(1, X.shape[0])
		X = torch.from_numpy(X).to(device)

		output = self.base_model(X)
		_, predicted = torch.max(output, dim=1)

		tag = base_model["all_tags"][predicted.item()]

		probs = torch.softmax(output, dim=1)
		prob = probs[0][predicted.item()]

		return tag, prob

	def reply(self, message):
		tag, prob = self.predict(message)

		intents = base_model["intents"]
		if prob.item() >= 0.99:
			for intent in intents["intents"]:
				if intent["tag"] == tag:
					response = random.choice(intent["responses"])
					return response, {"score": prob.item(), "label": tag}
		else:
			return "I don't know what you're saying.", {"score": prob.item(), "label": tag}


def main():
	chatbot = Chatbot()
	os.system("clear")
	print(colorama.Fore.MAGENTA + f"{'#' * 19} type 'quit' to exit {'#' * 19}" + colorama.Style.RESET_ALL + "\n\n")
	
	while True:
		message = input("{}You:{} ".format(colorama.Fore.CYAN, colorama.Style.RESET_ALL))
		
		if message == "quit":
			os.system("clear")
			break
		
		response, metadata = chatbot.reply(message)
		display(response + "\n")
		
		pprint(metadata)
		
		print()


if __name__ == "__main__":
	main()
