import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
from pyro.infer import Predictive
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.distributions import Normal, Categorical
from torch.nn.functional import softmax
from tqdm.auto import trange, tqdm
from ae import AE
import torch


class BNN(AE):
    
    def __init__(self, Xin, input_dim, is_dropout, name):
        
        super(input_dim, is_dropout, name).__init__()
        self.Xin = Xin
        
    
    def model(self):
        outw_prior = Normal(loc=torch.zeros_like(self.out.weight), scale=torch.ones_like(self.out.weight)).to_event(2)
        outb_prior = Normal(loc=torch.zeros_like(self.out.bias), scale=torch.ones_like(self.out.bias)).to_event(1)
        priors = {'out.weight': outw_prior, 'out.bias': outb_prior}
        # lift module parameters to random variables sampled from the priors
        lifted_module = pyro.random_module("module", self, priors)
        # sample a regressor (which also samples w and b)
        lifted_ae_model = lifted_module()
        with pyro.plate("data", self.Xin.shape[0]):
            Xout = lifted_ae_model(self.Xin)
            obs = pyro.sample("obs", dist.Categorical(Xout), obs=self.Xin)
            dist.Normal