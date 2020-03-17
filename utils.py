import numpy as np
import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)




class Surface(object):

    def __init__(self):

        self.num_weights = 5
        self.series = []
        for i in range(self.num_weights):
            self.series.append([])

    def add(self,model,loss):

        weight = list(model.parameters())[0].view(-1).data.detach().cpu().numpy()
        
        for i in range(self.num_weights):
            self.series[i].append([weight[i],loss])

    def plot(self):

        for i in range(self.num_weights):

            self.series[i] = np.array(self.series[i])
            self.series[i] = self.series[i][self.series[i][:,1].argsort()] 

            plt.figure(figsize=(18,10))

            plt.ylabel("Weight Value")
            plt.xlabel("Loss")

            plt.plot(self.series[i][:,1],self.series[i][:,0])
            # plt.show()
            plt.savefig(f"graphs/surface{str(i)}.png", dpi=600)
            plt.close()
