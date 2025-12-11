import torch
import numpy as np
import torch.nn as nn
from main.scheduler import get_diffusion_scheduler, EDMScheduler
from main.pfode import PFODE
import tqdm
from forward_operator import LatentWrapper


class MCMC_Langevin():
    def __init__(self, eta = 5*10**(-5), beta_y=0.01, iter = 100, delta = 0.01):
        self.eta = eta
        self.beta_y = beta_y
        self.iter = iter
        self.delta = delta

    def mcmc_chain(self, x0_hat, measurement, operator, r_t, t_fraction):
        x0_list = []
        # pbar = tqdm.trange(self.iter)
        pbar = range(self.iter) 
        x0 = x0_hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x0], self.eta)

        for p in pbar:
            optimizer.zero_grad()
            loss = operator.error(x0, measurement).sum() / (2 * self.beta_y ** 2)
            loss += ((x0 - x0_hat.detach()) ** 2).sum() / (2 * r_t ** 2)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                epsilon = torch.randn_like(x0)
                #If you dont use the data attribute the optimizer can't track any grad_fn
                x0.data = x0.data + np.sqrt(2 * self.eta*(1+t_fraction*(self.delta - 1))) * epsilon
                x0_list.append(x0)

        return x0, x0_list


class DAPS():
    def __init__(self, annealing_scheduler_config, diffusion_scheduler_config, lv_config):
        #self.annealing_scheduler = get_diffusion_scheduler(**annealing_scheduler_config)
        self.annealing_scheduler = EDMScheduler(num_steps=annealing_scheduler_config['num_steps'], sigma_max= annealing_scheduler_config['sigma_max'], sigma_min=annealing_scheduler_config['sigma_min'], timestep=annealing_scheduler_config['timestep'])
        self.diffusion_scheduler = diffusion_scheduler_config #because this will change with steps
        self.lv_config = MCMC_Langevin(eta = lv_config['lr'], beta_y = lv_config['tau'], iter = lv_config['num_steps'], delta = lv_config['lr_min_ratio'])


    def daps_sample(self, model, x_init, operator, measurement, record=False):

        if record:
            self.record_data = {'x0_hats':[], 'x0_conds':[]}
        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1)
        x_t = x_init
        for i in pbar:
            sigma_t = self.annealing_scheduler.sigma_steps[i]
            with torch.no_grad():
                #diffusion_scheduler = get_diffusion_scheduler(**self.diffusion_scheduler_config, sigma_max=sigma_t)
                diffusion_scheduler = EDMScheduler(num_steps=self.diffusion_scheduler['num_steps'], sigma_max= sigma_t, sigma_min=self.diffusion_scheduler['sigma_min'], timestep=self.diffusion_scheduler['timestep'])
                pfode = PFODE(diffusion_scheduler, model, input_shape = tuple(x_init.shape) , method='euler')
                x0_hat = pfode.solve(x_t)

            x0_cond, x0_cond_list = self.lv_config.mcmc_chain(x0_hat, measurement, operator, r_t = sigma_t, t_fraction=i/self.annealing_scheduler.num_steps)

            if i != self.annealing_scheduler.num_steps - 1:
                x_t = x0_cond + torch.randn(x0_cond.shape, device=x0_cond.device, dtype=x0_cond.dtype) * self.annealing_scheduler.sigma_steps[i + 1]
            else:
                x_t = x0_cond

            if record:
                self.record_data['x0_hats'].append(x0_hat.detach().cpu())
                self.record_data['x0_conds'].append(x0_cond.detach().cpu())

        return x_t
    
    def gaussian_prior_x_T(self, batch_size, in_shape, device):
        x_T = torch.randn(batch_size, *in_shape, device=device) * self.annealing_scheduler.get_prior_sigma()
        return x_T

    
class LatentDAPS(DAPS):
    
    def latent_daps_sample(self, model, z_init, operator, measurement):

        pbar = tqdm.trange(self.annealing_scheduler.num_steps - 1)
        z_t = z_init
        for i in pbar:
            sigma_t = self.annealing_scheduler.sigma_steps[i]
            with torch.no_grad():
                diffusion_scheduler = EDMScheduler(num_steps=self.diffusion_scheduler['num_steps'], sigma_max= sigma_t, sigma_min=self.diffusion_scheduler['sigma_min'], timestep=self.diffusion_scheduler['timestep'])
                pfode = PFODE(diffusion_scheduler, model, input_shape = tuple(z_init.shape) , method='euler')
                z0_hat = pfode.solve(z_t)
            
            z0_cond, z0_cond_list = self.lv_config.mcmc_chain(z0_hat, measurement, LatentWrapper(operator, model), r_t = sigma_t, t_fraction=i/self.annealing_scheduler.num_steps)

            if i != self.annealing_scheduler.num_steps - 1:
                z_t = z0_cond + torch.randn(z0_cond.shape, device=z0_cond.device, dtype=z0_cond.dtype) * self.annealing_scheduler.sigma_steps[i + 1]
            else:
                z_t = z0_cond
            with torch.no_grad():
                x_t = model.decode(z_t)

        return x_t


    

