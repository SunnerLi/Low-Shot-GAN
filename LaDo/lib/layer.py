from torch.nn import Parameter
import torch.nn as nn
import torch

"""
    This script defines the layer which will be used in the module of LaDo

    @Revised: Cheng-Che Lee
"""

class SpectralNorm2d(nn.Module):
    """
        The definition of spectral normalization

        Ref: https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
    """
    def __init__(self, module, name = 'weight', power_iterations = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._get_params():
            self._make_params()

    def l2normalize(self, t, eps = 1e-12):
        return t / (t.norm() + eps)

    def _get_params(self):
        """
            Get the parameters from the given module
        """
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        """
            Set the parameters from scratch
        """
        # Create the parameters
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width  = w.view(height, -1).data.shape[1]
        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad = False)
        v = Parameter(w.data.new(width ).normal_(0, 1), requires_grad = False)
        u.data = self.l2normalize(u.data)
        v.data = self.l2normalize(v.data)
        w_bar  = Parameter(w.data)

        # Regist the parameter into the module
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def _update_params(self):
        """
            Update the u and v parameter
            This function will be called during forwarding
        """
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self.l2normalize(torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def forward(self, *args):
        """
            Forward process of the spectral normalization
        """
        self._update_params()
        return self.module.forward(*args)