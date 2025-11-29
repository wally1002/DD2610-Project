import torch
from torchdiffeq import odeint



class PFODE():
    def __init__(self, scheduler, model, input_shape, method = 'euler'): 
        self.scheduler = scheduler
        self.model = model
        self.method = method
        self.x_T_shape = input_shape
        self.device = next(self.model.parameters()).device

    def diff_eq(self, t, x_t_flatten):
        #The PFODE differential equation in the EDM paper(as DAPS mentioned in page no-21)
        #Takes flatten input, convert to original shape, compute the RHS of the ODE and returns the flattened derivative
        x_t = x_t_flatten.view(*self.x_T_shape)
        s_t = self.scheduler.get_scaling(t)
        ds_t = self.scheduler.get_scaling_derivative(t)
        sigma_t = self.scheduler.get_sigma(t)
        dsigma_t = self.scheduler.get_sigma_derivative(t)
        rhs =  ds_t / s_t * x_t - (s_t**2) * dsigma_t * sigma_t * self.model.score(x_t/s_t, sigma=sigma_t)
        return rhs.flatten(1)

    
    def solve(self, x_t):
        max_t = self.scheduler.get_t_max()
        min_t = self.scheduler.get_t_min()
        #Because of the reverse direction
        #t_step = torch.tensor([max_t, min_t],dtype=torch.float32).to(x_t.device)
        t_step = self.scheduler.get_discrete_time_steps(self.scheduler.num_steps).to(x_t.device)
        #Solve to find an estimate of x_0
        x_0_flatten = odeint(self.diff_eq, x_t.flatten(1), t_step, method=self.method)#(2,())
        #x_0 = x_0_flatten.view(2, *self.x_T_shape)
        x_0 = x_0_flatten.view(self.scheduler.num_steps, *self.x_T_shape)
        return x_0[-1]

    def gaussian_prior_x_T(self, batch_size):
        in_shape = self.model.get_in_shape()
        x_T = torch.randn(batch_size, *in_shape, device=self.device) * self.scheduler.get_prior_sigma()
        return x_T