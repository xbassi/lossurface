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

	return f1 



def evalav(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model = model.cuda()
    model.eval()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    #print(len(loader.dataset))

    return {
        'loss': loss_sum / len(loader.dataset),
        #'loss': loss_sum,
        'accuracy': correct / len(loader.dataset) * 100.0,
    }
