import torch
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

import torchvision
import torchvision.transforms as transforms
import ipywidgets

import os
import argparse
import csv
import numpy
import random
import torch.nn.functional as F
import torch.optim as optim

from onecycle import CustomOneCycleLR

import models



model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

print('Available Models: \n')
print(', '.join(model_names)) 

print('Torch version: ' + str(torch.__version__)) #There have been version dependent results


#######################################
# Parser

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='rezero_preactresnet18', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: fixup_resnet110)')
parser.add_argument('--sess', default='lightning', type=str, help='session id')
parser.add_argument('--seed', default=random.randint(0,10000), type=int, help='rng seed')
parser.add_argument('--decay', default=2e-4, type=float, help='weight decay (default=2e-4)')
parser.add_argument('--batchsize', default=512, type=int, help='batch size per GPU (default=512)')
parser.add_argument('--n_epoch', default=45, type=int, help='total number of epochs')
parser.add_argument('--init_lr', default=0.032, type=float)
parser.add_argument('--point_1_step', default = 0.1, type=float)
parser.add_argument('--point_1_lr', default = 1.2, type=float)
parser.add_argument('--point_2_step', default = 0.9, type=float)
parser.add_argument('--point_2_lr', default = 0.032, type=float)
parser.add_argument('--end_lr', default=0.001, type=float)
parser.add_argument('--resweight_lr', default=0.05, type=float)
parser.add_argument('--momentum_range', default=(0.85, 0.95), type=tuple)
parser.add_argument('--exp_decay', default=False, type=bool)
parser.add_argument('--progress_bar', default=True, type=bool, help='display progress bar')
parser.add_argument('--precision', default = 16, type=int)

global arg
args = parser.parse_args(); args

args.progress_bar = (args.progress_bar=='True')
if args.progress_bar:
    from utils import progress_bar


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
numpy.random.seed(args.seed)
random.seed(args.seed)
#torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = args.batchsize
lr = args.init_lr



normalize = transforms.Normalize(mean=[0.4914 , 0.48216, 0.44653], std=[0.24703, 0.24349, 0.26159])


transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(.25,.25,.25),
            transforms.RandomRotation(2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=8)


from torch.optim import Optimizer

class CustomOneCycleLR:
    """ Custom Scheduler
    Example:
    
    scheduler = CustomOneCycleLR(optimizer, 
                             num_steps = total_iterations,
                             init_lr = 0.048,
                             point_1 = (0.39,1.2),
                             point_2 = (0.78,0.048),
                             end_lr = 0.0012,
                             momentum_range = (0.85, 0.95),
                             exp_decay = False)
    """

    def __init__(self,
                 optimizer: Optimizer,
                 num_steps: int,
                 init_lr: float = 0.048,
                 point_1: tuple = (0.39,1.2),
                 point_2: tuple = (0.78,0.048),
                 end_lr: float = 0.0012,
                 momentum_range: tuple = (0.85, 0.95),
                 last_step: int = -1,
                 exp_decay: bool = False,
                 param_group: int = 0
                 ):
        
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        
        self.param_group = param_group
        
        self.optimizer = optimizer

        self.num_steps = num_steps
        
        
        
        self.step_0 = 0
        self.lr_0 = init_lr
        self.step_1 = round((point_1[0] * self.num_steps))
        self.lr_1 = point_1[1]
        self.step_2 = round((point_2[0] * self.num_steps))
        self.lr_2 = point_2[1]
        self.step_3 = num_steps
        self.lr_3 = end_lr
        
        self.lrs = ([self.lr_0,self.lr_1,self.lr_2])
        self.max_lr = max(self.lrs)
        self.min_lr = min(self.lrs)
        
        self.last_step = last_step
        
        self.exp_decay = exp_decay
        if self.exp_decay:
            self.decay_factor = ((self.lr_3)/self.lr_2)**(1/(self.step_3-self.step_2))
        

        self.min_momentum, self.max_momentum = momentum_range[0], momentum_range[1]
        assert self.min_momentum < self.max_momentum, \
            "Argument momentum_range must be (min_momentum, max_momentum), where min_momentum < max_momentum"

        self.lr = self.lr_0
        self.momentum = self.max_momentum
        
        self.last_step = last_step
    
        if self.last_step == -1:
            self.step()
        

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state. (Borrowed from _LRScheduler class in torch.optim.lr_scheduler.py)
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        return self.optimizer.param_groups[self.param_group]['lr']

    def get_momentum(self):
        return self.optimizer.param_groups[self.param_group]['momentum']

    def compute_lr(self):
        current_step = self.last_step + 1
        self.last_step = current_step
        if current_step <= self.step_1:
            # point 0->1
            scale = (current_step - self.step_0) / (self.step_1 - self.step_0)
            lr = self.lr_0 + scale * (self.lr_1 - self.lr_0)
            
            momentum = self.max_momentum + scale * (self.min_momentum - self.max_momentum)
        elif current_step <= self.step_2:
            # point 1->2
            scale = (current_step - self.step_1) / (self.step_2 - self.step_1)
            lr = self.lr_1 + scale * (self.lr_2 - self.lr_1)
            
            momentum = self.min_momentum - scale * (self.min_momentum - self.max_momentum)
        elif current_step <= self.step_3:
            # point 2->3
            if self.exp_decay:
                lr = self.lr_2 * self.decay_factor**((current_step-self.step_2))
            else:
                scale = (current_step - self.step_2) / (self.step_3 - self.step_2)
                lr = self.lr_2 + scale * (self.lr_3 - self.lr_2)
            
            momentum = self.max_momentum
        else:
            print('Exceeded given num_steps, returns to start')
            self.last_step = 0
            return
        self.momentum = momentum
        self.lr = lr
    
    def step(self):
        """Conducts one step of learning rate and momentum update
        """

        self.compute_lr()

        self.optimizer.param_groups[self.param_group]['lr'] = self.lr
        self.optimizer.param_groups[self.param_group]['momentum'] = self.momentum
    
    def test(self):
        self.last_step_memory = self.last_step
        lrs = []
        momentums = []
        for i in range(0,self.step_3):
            self.compute_lr()
            lrs.append(self.lr)
            momentums.append(self.momentum)
        self.last_step = self.last_step_memory
        return lrs, momentums
    
    def test_plt(self):
        import matplotlib.pyplot as plt
        
        self.last_step_memory = self.last_step
        lrs = []
        momentums = []
        for i in range(0,self.step_3):
            self.compute_lr()
            lrs.append(self.lr)
            momentums.append(self.momentum)
        self.last_step = self.last_step_memory
        
        plt.plot(lrs)
        plt.ylabel('Learning Rate')
        plt.xlabel('Iterations')
        plt.show()

        plt.plot(momentums)
        plt.ylabel('Momentum')
        plt.xlabel('Iterations')
        plt.show()
        







class LitModel(LightningModule):

    def __init__(self,args):
        super().__init__()
        self.args = args
        self.model = models.__dict__[self.args.arch]()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        images, target = batch
        output = self(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        tensorboard_logs = {'train_loss': loss, 'acc1': acc1, 'acc5': acc5}
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        images, target = batch
        output = self.model(images)
        loss = F.cross_entropy(output, target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        tensorboard_logs = {'val_loss': loss, 'val_acc1': acc1, 'val_acc5': acc5}
        return {'val_loss': loss, 'log': tensorboard_logs}
    
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['log']['val_loss'] for x in outputs]).mean()
        avg_acc1 = torch.stack([x['log']['val_acc1'] for x in outputs]).mean()
        avg_acc5 = torch.stack([x['log']['val_acc5'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc1': avg_acc1, 'val_acc5': avg_acc5}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def train_dataloader(self):

        return trainloader
    
    def configure_optimizers(self):
        parameters_others = [p[1] for p in self.model.named_parameters() if not ('resweight' in p[0])]
        parameters_resweight = [p[1] for p in self.model.named_parameters() if 'resweight' in p[0]]
        optimizer = optim.SGD(
            parameters_others,
            lr = .01,#scheduler will change this
            momentum = 0.9,#scheduler will change this,
            weight_decay=self.args.decay
        )
        args = self.args
        customonecycle = CustomOneCycleLR(optimizer, 
                             num_steps = args.n_epoch * len(trainloader),
                             init_lr = args.init_lr,
                             point_1 = (args.point_1_step,args.point_1_lr),
                             point_2 = (args.point_2_step,args.point_2_lr),
                             end_lr = args.end_lr,
                             momentum_range = args.momentum_range,
                             exp_decay = args.exp_decay,
                             param_group = 0)
        
        scheduler = {'scheduler': customonecycle,'interval': 'step'}
        optimizer_resweight = optim.Adagrad([
        {'params': parameters_resweight, 'lr': self.args.resweight_lr}])
        return [optimizer, optimizer_resweight], [scheduler]
    
    
    
    def val_dataloader(self):
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.args.batchsize, shuffle=False, num_workers=8)
        return testloader
    
    def __accuracy(cls, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
            
litmodel = LitModel(args)
wandb_logger = WandbLogger(project = 'cifar_lightning',  entity='tbachlechner', name = args.sess)
#os.environ["WANDB_MODE"] = "dryrun"



trainer = Trainer( max_epochs = args.n_epoch, gpus=1, num_nodes=1,logger=wandb_logger, 
                  precision=args.precision,log_save_interval=50 )
trainer.fit(litmodel)
