import torch
from torchvision import datasets
import torchvision.transforms as transforms


def get_loaders(args):


	train_transforms = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])

	test_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			])


	train_set = datasets.CIFAR10(args.data_dir,train=True, transform=train_transforms, download=True)
	test_set = datasets.CIFAR10(args.data_dir,train=False, transform=test_transforms, download=True)


	train_loader = torch.utils.data.DataLoader(
	        train_set,
	        batch_size=args.batch_size,
	        shuffle=True,
	        num_workers=args.num_workers,
	        pin_memory=False
	    )

	test_loader = torch.utils.data.DataLoader(
	        test_set,
	        batch_size=args.batch_size,
	        shuffle=True,
	        num_workers=args.num_workers,
	        pin_memory=False
	    )


	return train_loader,test_loader