'''A wrapper class for scheduled optimizer '''
import numpy as np

class ScheduledOptim():
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))

    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer. The optimizer is stored as '_optimizer'.
        """
        return {
            'lr_mul': self.lr_mul,
            'd_model': self.d_model,
            'n_warmup_steps': self.n_warmup_steps,
            'n_steps': self.n_steps,
            '_optimizer': self._optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.lr_mul = state_dict['lr_mul']
        self.d_model = state_dict['d_model']
        self.n_warmup_steps = state_dict['n_warmup_steps']
        self.n_steps = state_dict['n_steps']
        self._optimizer.load_state_dict(state_dict['_optimizer'])