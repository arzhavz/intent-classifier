import sys
import time
import os


def display(string):
	for char in string:
		sys.stdout.write(char)
		sys.stdout.flush()
		time.sleep(0.05)


def calculate_params(model):
	params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	buffer = sum(p.numel() for p in model.buffers())
	total_params = params + buffer

	output = f"Jumlah parameter dari model {model.__class__.__name__}: \nParameters: {params:,}"
	return output, params


def number_formatter(angka):
	angka = float(angka)
	satuan = ["", "K", "M", "B", "T", "Q"]
	faktor = [1, 1e3, 1e6, 1e9, 1e12, 1e15]

	for i in range(len(satuan)):
		if angka < faktor[i]:
			return f"{round(angka/faktor[i-1],1)}{satuan[i-1]}"
	return f"{round(angka/faktor[-1],1)}{satuan[-1]}"

		
def list_files_by_extension(extension):
	current_dir = os.getcwd()  
	file_list = []

	for file in os.listdir(current_dir):
		if file.endswith(extension):
			file_list.append(file)

	return file_list