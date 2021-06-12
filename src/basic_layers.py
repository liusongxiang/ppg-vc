import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x

class Linear(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)

class Conv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', param=None):
        super(Conv1d, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1)/2)
        
        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain, param=param))

    def forward(self, x):
        # x: BxDxT
        return self.conv(x)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Reshape layer    
class Reshape(nn.Module):
    def __init__(self, outer_shape):
        super(Reshape, self).__init__()
        self.outer_shape = outer_shape
    def forward(self, x):
        return x.view(x.size(0), *self.outer_shape)

# Sample from the Gumbel-Softmax distribution and optionally discretize.
class GumbelSoftmax(nn.Module):

    def __init__(self, f_dim, c_dim):
        super(GumbelSoftmax, self).__init__()
        self.logits = nn.Linear(f_dim, c_dim)
        self.f_dim = f_dim
        self.c_dim = c_dim
     
    def sample_gumbel(self, shape, is_cuda=False, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax_sample(self, logits, temperature):
        y = logits + self.sample_gumbel(logits.size(), logits.is_cuda)
        return F.softmax(y / temperature, dim=-1)

    def gumbel_softmax(self, logits, temperature, hard=False):
        """
        ST-gumple-softmax
        input: [*, n_class]
        return: flatten --> [*, n_class] an one-hot vector
        """
        #categorical_dim = 10
        y = self.gumbel_softmax_sample(logits, temperature)

        if not hard:
            return y

        shape = y.size()
        _, ind = y.max(dim=-1)
        y_hard = torch.zeros_like(y).view(-1, shape[-1])
        y_hard.scatter_(1, ind.view(-1, 1), 1)
        y_hard = y_hard.view(*shape)
        # Set gradients w.r.t. y_hard gradients w.r.t. y
        y_hard = (y_hard - y).detach() + y
        return y_hard 
  
    def forward(self, x, temperature=1.0, hard=False):
        logits = self.logits(x).view(-1, self.c_dim)
        prob = F.softmax(logits, dim=-1)
        y = self.gumbel_softmax(logits, temperature, hard)
        return logits, prob, y

class Softmax(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.logits = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim
        self.out_dim = out_dim
     
    def forward(self, x):
        logits = self.logits(x).view(-1, self.out_dim)
        prob = F.softmax(logits, dim=-1)
        return logits, prob

# Sample from a Gaussian distribution
class Gaussian(nn.Module):
    def __init__(self, in_dim, z_dim, use_bias=True):
        super(Gaussian, self).__init__()
        self.mu = nn.Linear(in_dim, z_dim, bias=use_bias)
        self.log_std = nn.Linear(in_dim, z_dim, bias=use_bias)
        # self.mu.weight.data.fill_(0.0)
        # self.log_std.weight.data.fill_(0.0)
        # if use_bias:
            # self.mu.bias.data.fill_(0.0)
            # self.log_std.bias.data.fill_(0.0)

    def reparameterize(self, mu, log_std):
        std = torch.exp(log_std)
        noise = torch.randn_like(std)
        z = mu + noise * std
        return z      

    def forward(self, x):
        mu = self.mu(x)
        log_std = self.log_std(x)
        z = self.reparameterize(mu, log_std)
        return mu, log_std, z 

# https://github.com/fungtion/DANN/blob/master/models/functions.py
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
# class Gaussian(nn.Module):
    # def __init__(self, in_dim, z_dim):
        # super(Gaussian, self).__init__()
        # self.mu = nn.Linear(in_dim, z_dim)
        # self.var = nn.Linear(in_dim, z_dim)

    # def reparameterize(self, mu, var):
        # std = torch.sqrt(var + 1e-10)
        # noise = torch.randn_like(std)
        # z = mu + noise * std
        # return z      

    # def forward(self, x):
        # mu = self.mu(x)
        # var = F.softplus(self.var(x))
        # z = self.reparameterize(mu, var)
        # return mu, var, z 

def tile(x, count, dim=0):
    """
    Tiles x on dimension dim count times.
    """
    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
         .transpose(0, 1) \
         .repeat(count, 1) \
         .transpose(0, 1) \
         .contiguous() \
         .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def sort_batch(data, lengths):
    '''
    sort data by length
    sorted_data[initial_index] == data
    '''
    sorted_lengths, sorted_index = lengths.sort(0, descending=True)
    sorted_data = data[sorted_index]
    _, initial_index = sorted_index.sort(0, descending=False)

    return sorted_data, sorted_lengths, initial_index
