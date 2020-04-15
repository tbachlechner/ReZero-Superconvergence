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
        

