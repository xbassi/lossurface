import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib as mpl 
mpl.rcParams['agg.path.chunksize'] = 10000000

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

        self.num_weights = 7
        self.series = []
        for i in range(self.num_weights):
            self.series.append([])

        self.counter = 0

    def add(self,model,loss,score):

        self.counter += 1
        
        
        for i in range(self.num_weights):
            weight = list(model.parameters())[i].view(-1).data.detach().cpu().numpy()
            self.series[i].append([weight[0],loss,score,self.counter])

    def plot(self):

        for i in range(self.num_weights):
            print(len(self.series[i]))
            self.series[i] = np.array(self.series[i])
            plt.figure(figsize=(18,10))

            ysmoothed = gaussian_filter1d(self.series[i][:,1], sigma=4)

            plt.ylabel("Loss")
            plt.xlabel("Weight Value")

            # plt.plot(self.series[i][:,2],self.series[i][:,0],color="blue")
            plt.plot(self.series[i][:,0],self.series[i][:,1],color="red",linewidth=1,marker=".")
            plt.plot(self.series[i][:,0],ysmoothed,color="#777777",linewidth=1,marker=".")

            plt.plot([self.series[i][-1,0]],[self.series[i][-1,1]],color="blue",marker=".")
            # plt.show()
            plt.savefig(f"graphs/adam_r4_{str(i)}_1.png", dpi=300)
            # plt.clf()

            self.series[i] = self.series[i][self.series[i][:,0].argsort()]

            # plt.ylabel("Score")
            # plt.xlabel("Weight Value")

            # plt.plot(self.series[i][:,0],self.series[i][:,2])
            # # plt.show()
            # plt.savefig(f"graphs/surface{str(i)}_2.png", dpi=300)
            # plt.clf()

            # self.series[i] = self.series[i][self.series[i][:,1].argsort()]

            # plt.ylabel("Loss")
            # plt.xlabel("Weight Value")

            # plt.plot(self.series[i][:,0],self.series[i][:,1])
            # # plt.show()
            # plt.savefig(f"graphs/surface{str(i)}_3.png", dpi=300)

            plt.close()



