from abc import ABC, abstractmethod
import torch
import numpy as np

__DIFFUSION_SCHEDULER__ = {}

def get_diffusion_scheduler(name: str, **kwargs):
    if __DIFFUSION_SCHEDULER__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __DIFFUSION_SCHEDULER__[name](**kwargs)


class Scheduler(ABC):
    """
    Abstract base class for diffusion scheduler. Design is similar to the author's github

    Schedulers manage time steps, noise scales (sigma), scaling factors, and coefficients 
    used in diffusion stochastic/ordinary differential equations (SDEs/ODEs).
    """

    def __init__(self, num_steps):
        self.num_steps = num_steps + 1 # include the initial step

    def discretize(self, time_steps):
        sigma_steps = self.get_sigma(time_steps[:-1])
        sigma_steps = torch.cat([sigma_steps, torch.zeros_like(sigma_steps[:1])])
        self.sigma_steps = sigma_steps

    def tensorize(self, data):
        if isinstance(data, (int, float)):
            return torch.tensor(data).float()
        if isinstance(data, list):
            return torch.tensor(data).float()
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        if isinstance(data, torch.Tensor):
            return data.float()
        raise ValueError(f"Data type {type(data)} is not supported.") 

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Noise Scheduling & Scaling Function 
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @abstractmethod
    def get_scaling(self, t):
        pass
    
    def get_sigma(self, t):
        pass
    
    def get_scaling_derivative(self, t):
        pass

    def get_sigma_derivative(self, t):
        pass

    def get_sigma_inv(self, sigma):
        pass
    
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Time & Sigma Range Function
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_t_min(self):
        pass

    def get_t_max(self):
        pass

    def get_discrete_time_steps(self, num_steps):
        pass

    def get_sigma_max(self):
        return self.get_sigma(self.get_t_max())

    def get_sigma_min(self):
        return self.get_sigma(self.get_t_min())
    
    def get_prior_sigma(self):
        # simga(t_max) * scaling(t_max)
        return self.get_sigma_max() * self.get_scaling(self.get_t_max())

    def __iter__(self):
        self.pbar = tqdm.trange(self.num_steps) if self.verbose else range(self.num_steps)
        self.pbar_iter = iter(self.pbar)
        return self

    def __next__(self):
        try:
            step = next(self.pbar_iter)
            time, scaling, sigma, scaling_factor, factor = self.time_steps[step], self.scaling_steps[step], \
                self.sigma_steps[step], self.scaling_factor_steps[step], self.factor_steps[step]
            return self.pbar, time, scaling, sigma, factor, scaling_factor
        except StopIteration:
            raise StopIteration


class EDMScheduler(Scheduler):
    """
        EDM (Elucidating the Design Space of Diffusion-Based Generative Models) Scheduler.
    """

    def __init__(self, num_steps, sigma_max=100, sigma_min=1e-2, timestep='poly-7'):
        super().__init__(num_steps)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        p = int(timestep.split('-')[1])
        self.time_steps_fn = lambda r: (sigma_max ** (1 / p) + r * (sigma_min ** (1 / p) - sigma_max ** (1 / p))) ** p

        # get time_steps
        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_sigma(self, t):
        # sigma(t) = t
        return self.tensorize(t)

    def get_scaling(self, t):
        # s(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_sigma_derivative(self, t):
        # sigma'(t) = 1
        return torch.ones_like(self.tensorize(t))

    def get_scaling_derivative(self, t):
        # s'(t) = 0
        return torch.zeros_like(self.tensorize(t))
    
    def get_sigma_inv(self, sigma):
        return self.tensorize(sigma)

    def get_t_min(self):
        return self.tensorize(self.sigma_min)
    
    def get_t_max(self):
        return self.tensorize(self.sigma_max)

    def get_discrete_time_steps(self, num_steps):
        steps = np.linspace(0, 1, num_steps)
        time_steps = np.array([self.time_steps_fn(s) for s in steps])
        return torch.from_numpy(time_steps)


#@register_diffusion_scheduler('vp')
class VPScheduler(Scheduler):
    """Variance Preserving Scheduler."""

    def __init__(self, num_steps, beta_max=20, beta_min=0.1, epsilon=1e-5, beta_type='linear'):
        super().__init__(num_steps)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_type = beta_type
        self.epsilon = epsilon

        if beta_type == 'linear':
            self.n = 1
        elif beta_type == 'scaled_linear':
            self.n = 2
        else:
            raise NotImplementedError
        
        self.a = beta_max ** (1 / self.n) - beta_min ** (1 / self.n)
        self.b = beta_min ** (1 / self.n)

        time_steps = self.get_discrete_time_steps(self.num_steps)
        self.discretize(time_steps)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # For VP Scheduler Only
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_beta(self, t):
        # beta(t) = (a * t + b) ^ n
        t = self.tensorize(t)
        return (self.a * t + self.b) ** self.n

    def get_beta_integrated(self, t):
        # beta_integrated(t) = [(a * t + b) ^ (n + 1) - b ^ (n + 1)] / a / (n + 1)
        t = self.tensorize(t)
        return ((self.a * t + self.b) ** (self.n + 1) - self.b ** (self.n + 1)) / self.a / (self.n + 1)

    def get_alpha(self, t):
        # alpha(t) = exp(-beta_integrated(t))
        t = self.tensorize(t)
        return torch.exp(-self.get_beta_integrated(t))

    def get_alpha_derivative(self, t):
        # alpha'(t) = -beta(t) * alpha(t)
        t = self.tensorize(t)
        return - self.get_beta(t) * self.get_alpha(t)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # General Interface
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def get_scaling(self, t):
        # s(t) = sqrt(alpha(t))
        t = self.tensorize(t)
        return torch.sqrt(self.get_alpha(t))

    def get_sigma(self, t):
        # sigma(t) = sqrt(1 / alpha(t) - 1)
        t = self.tensorize(t)
        return torch.sqrt(1 / self.get_alpha(t) - 1)

    def get_scaling_derivative(self, t):
        # s'(t) = -s(t) * beta(t) / 2
        t = self.tensorize(t)
        return - self.get_scaling(t) * self.get_beta(t) / 2

    def get_sigma_derivative(self, t):
        # sigma'(t) = beta(t) / 2 / sigma(t) / alpha(t)
        t = self.tensorize(t)
        return self.get_beta(t) / 2 / self.get_sigma(t) / self.get_alpha(t)

    def get_sigma_inv(self, sigma):
        # t = {[a(n+1)log(sigma^2 + 1) + b^(n+1)]^(1/(n + 1)) - b}/a
        sigma = self.tensorize(sigma)
        return ((self.a * (self.n + 1) * torch.log(sigma ** 2 + 1) + self.b ** (self.n + 1)) ** (1 / (self.n + 1)) - self.b) / self.a

    def get_t_min(self):
        return self.tensorize(self.epsilon)
    
    def get_t_max(self):
        return self.tensorize(1)

    def get_discrete_time_steps(self, num_steps):
        return torch.linspace(1, self.epsilon, num_steps)