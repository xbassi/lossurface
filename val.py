import torch
import numpy as np
from sklearn.metrics import f1_score,accuracy_score


def validate(val_loader,model, criterion,device):

	model.eval()
	pred = None
	gt = None

	for i, (inputs, label) in enumerate(val_loader):

		label = label.detach().cpu().numpy()

		output = model(inputs.to(device))

		output = torch.argmax(output,dim=1)
		output = output.detach().cpu().numpy()

		if pred is None:
			pred = output
		else:
			pred = np.concatenate((pred,output))

		if gt is None:
			gt = label
		else:
			gt = np.concatenate((gt,label))

	f1 = f1_score(gt,pred,average='weighted')
	acc = accuracy_score(gt,pred)

	print("F1:",f1,"ACC:",acc)