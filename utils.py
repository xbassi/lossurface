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



class AsymValley(object):

    def __init__(self,name,num_directions):

        self.name=name
        self.num_directions = num_directions


    def draw(self,args,model1,model2,loss_func):

        vec_sgd = self.parameters_to_vector(model1.parameters())
        vec_swa = self.parameters_to_vector(model2.parameters())
        # print(vec_swa.size())
        vec_rand = torch.rand(vec_swa.shape)
        vec_rand = vec_rand / torch.norm(vec_rand)

        # print(vec_sgd)
        # print(vec_swa)


        distances = 20

        loss_record = np.zeros( (distances*2) + 1)
        # vec_rand = vec_swa - vec_sgd
        # distances_scale = torch.norm(vec_rand)/5


        for distance in range(-distances, distances + 1):
            # print(distance)
            vec_temp = vec_swa + distance * vec_rand * args.distances_scale
            self.vector_to_parameters(vec_temp, model1.parameters())

            model1.eval()
            out = model1(x)  # input x and predict based on x
            loss_record[distance+distances] = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOT one-hotted

        np.savetxt(os.path.join(args.dir, 'loss_record.txt'), loss_record)


        #draw figure with format?
        sgd_train_loss_results = np.loadtxt(os.path.join(args.dir, 'loss_record.txt'))
        
        distances_scale = args.distances_scale
        plt.rcParams['figure.figsize'] = (7.0, 4.0)
        plt.subplots_adjust(bottom=.12, top=.99, left=.1, right=.99)
        plt.plot(np.arange(-distances*distances_scale, distances*distances_scale + distances_scale, distances_scale), sgd_train_loss_results, label='Training loss', color='dodgerblue')
        plt.scatter(0, sgd_train_loss_results[distances], marker='o',s=70,c='orange',label='SGD solution')
        plt.legend(fontsize=14)
        plt.ylabel('Loss',fontsize=14)
        plt.xlabel('A random direction generated from (0,1)-uniform distribution',fontsize=13)
        plt.savefig(os.path.join(args.dir, self.name+str(seed0)+'.png'))
        # plt.savefig(os.path.join(args.dir, 'logistic_regression_asym'+str(seed0)+'.pdf'))
        plt.close()

    def parameters_to_vector(self,parameters):
        r"""Convert parameters to one vector
        Arguments:
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.
        Returns:
            The parameters represented by a single vector
        """
        # Flag for the device where the parameter is located
        param_device = None

        vec = []
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = self._check_param_device(param, param_device)

            vec.append(param.view(-1))
        return torch.cat(vec)


    def vector_to_parameters(self,vec, parameters):
        r"""Convert one vector to the parameters
        Arguments:
            vec (Tensor): a single vector represents the parameters of a model.
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.
        """
        # Ensure vec of type Tensor
        if not isinstance(vec, torch.Tensor):
            raise TypeError('expected torch.Tensor, but got: {}'
                            .format(torch.typename(vec)))
        # Flag for the device where the parameter is located
        param_device = None

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in parameters:
            # Ensure the parameters are located in the same device
            param_device = self._check_param_device(param, param_device)

            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param.data = vec[pointer:pointer + num_param].view_as(param).data

            # Increment the pointer
            pointer += num_param


    def _check_param_device(self,param, old_param_device):
        r"""This helper function is to check if the parameters are located
        in the same device. Currently, the conversion between model parameters
        and single vector form is not supported for multiple allocations,
        e.g. parameters in different GPUs, or mixture of CPU/GPU.
        Arguments:
            param ([Tensor]): a Tensor of a parameter of a model
            old_param_device (int): the device where the first parameter of a
                                    model is allocated.
        Returns:
            old_param_device (int): report device for the first time
        """

        # Meet the first parameter
        if old_param_device is None:
            old_param_device = param.get_device() if param.is_cuda else -1
        else:
            warn = False
            if param.is_cuda:  # Check if in same GPU
                warn = (param.get_device() != old_param_device)
            else:  # Check if in CPU
                warn = (old_param_device != -1)
            if warn:
                raise TypeError('Found two parameters on different devices, '
                                'this is currently not supported.')
        return old_param_device


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



