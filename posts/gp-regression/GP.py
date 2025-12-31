import numpy as np
import torch
import torch.nn as nn

class Kernel(nn.Module):
    def forward(self, x1, x2):
        raise NotImplementedError

class RBFKernel(Kernel):
    def __init__(self, log_lengthscale, log_variance):
        super().__init__()
        self.log_lengthscale = nn.Parameter(log_lengthscale)
        self.log_variance = nn.Parameter(log_variance)

    def forward(self, x1, x2):
        l = self.log_lengthscale.exp()
        var2 = self.log_variance.exp()
        d2 = (x1[:, None] - x2[None, :])**2
        return var2 * torch.exp(-0.5 * d2 / l**2)
    
class DeltaKernel(Kernel):
    def __init__(self, log_variance):
        super().__init__()
        self.log_variance = nn.Parameter(log_variance)

    def forward(self, x1, x2):
        var2 = self.log_variance.exp()
        n = x1.shape[0]
        return var2 * torch.eye(n)
    
class PeriodicKernel(Kernel):
    def __init__(self, log_lengthscale, log_variance, log_p):
        super().__init__()
        self.log_lengthscale = nn.Parameter(log_lengthscale)
        self.log_variance = nn.Parameter(log_variance)
        self.log_p = nn.Parameter(log_p)

    def forward(self, x1, x2):
        l = self.log_lengthscale.exp()
        var2 = self.log_variance.exp()
        p = self.log_p.exp()
        d = (x1[:, None] - x2[None, :])
        return var2 * torch.exp(-2.0 * torch.square(torch.sin(torch.pi * d / p)) / l**2)
    
class PolynomialKernel(Kernel):
    def __init__(self, log_c, log_variance, m):
        super().__init__()
        self.m = m
        self.log_c = nn.Parameter(log_c)
        self.log_variance = nn.Parameter(log_variance)

    def forward(self, x1, x2):
        c = self.log_c.exp()
        var2 = self.log_variance.exp()
        return var2*torch.pow(x1[:, None]@x2[None, :] + c, self.m)
    
class SumKernel(Kernel):
    def __init__(self, *kernels):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x1, x2):
        return sum(k(x1, x2) for k in self.kernels)
    
class ProductKernel(Kernel):
    def __init__(self, *kernels):
        super().__init__()
        self.kernels = nn.ModuleList(kernels)

    def forward(self, x1, x2):
        K = self.kernels[0](x1, x2)
        for k in self.kernels[1:]:
            K = K * k(x1, x2)
        return K
    
class GaussianProcess(nn.Module):
    def __init__(self, kernel, log_noise):
        super().__init__()
        self.kernel = kernel
        self.log_noise = nn.Parameter(log_noise)

    def forward(self, x, y):
        n = x.shape[0]
        K = self.kernel(x, x)
        K = K + torch.eye(n, device=x.device) * self.log_noise.exp()

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y[:, None], L)

        logdet = 2 * torch.sum(torch.log(torch.diag(L)))
        nll = 0.5 * y @ alpha.squeeze() + 0.5 * logdet
        return nll
    
    def fit(self, x, y, optimizer_cls=torch.optim.Adam, lr=0.05, n_steps=500, callback=None):
        optimizer = optimizer_cls(self.parameters(), lr=lr)
        losses = []

        for step in range(n_steps):
            optimizer.zero_grad()
            loss = self(x, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if callback is not None:
                callback(step, loss, self)

        return losses
    
    @torch.no_grad()
    def sample(self, X):
        cov = self.kernel(X, X)
        y = np.random.multivariate_normal(mean=np.zeros_like(X), cov=cov)
        return y
    
    @torch.no_grad()
    def condition(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

        K = self.kernel(x_train, x_train)
        K = K + torch.eye(len(x_train), device=x_train.device)*self.log_noise.exp()

        L = torch.linalg.cholesky(K)
        alpha = torch.cholesky_solve(y_train[:, None], L)

        self.L = L
        self.alpha = alpha

    @torch.no_grad()
    def predict(self, x_test, return_cov=False):
        K_star = self.kernel(self.x_train, x_test)
        mean = K_star.T @ self.alpha

        if not return_cov:
            return mean.squeeze()
        
        v = torch.linalg.solve_triangular(self.L, K_star, upper=False)
        K_test = self.kernel(x_test, x_test)
        cov = K_test - v.T @ v

        return mean.squeeze(), torch.diag(cov)
    
    @torch.no_grad()
    def print_hyperparameters(self):
        for name, param in self.named_parameters():
            value = param.detach().cpu().item() if param.numel() == 1 else param.detach().cpu()
            if name.split('.')[-1].split('_')[0] == 'log':
                name = "exp("+name+")"
                print(f"{name:40s} = {np.exp(value)}")
            else:
                print(f"{name:40s} = {value}")