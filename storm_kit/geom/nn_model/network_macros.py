#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#

import torch
from torch import nn
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, ReLU6, ELU, Dropout, BatchNorm1d as BN, LayerNorm as LN


def xavier(param):
    """ initialize weights with xavier.

    Args:
        param (network params): params to initialize.
    """    
    nn.init.xavier_uniform(param)

def he_init(param):
    """initialize weights with he.

    Args:
        param (network params): params to initialize.
    """    
    nn.init.kaiming_uniform_(param,nonlinearity='relu')
    nn.init.normal(param)

def weights_init(m):
    """Function to initialize weights of a nn.

    Args:
        m (network params): pass in model.parameters()
    """    
    fn = he_init
    if isinstance(m, nn.Conv2d):
        fn(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Conv3d):
        fn(m.weight.data)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fn(m.weight.data)
        if(m.bias is not None):
            m.bias.data.zero_()

def MLP(channels, dropout_ratio=0.0, batch_norm=False,act_fn=ReLU,layer_norm=False,nerf=True):
    """Automatic generation of mlp given some

    Args:
        channels (int): number of channels in input
        dropout_ratio (float, optional): dropout used after every layer. Defaults to 0.0.
        batch_norm (bool, optional): batch norm after every layer. Defaults to False.
        act_fn ([type], optional): activation function after every layer. Defaults to ReLU.
        layer_norm (bool, optional): layer norm after every layer. Defaults to False.
        nerf (bool, optional): use positional encoding (x->[sin(x),cos(x)]). Defaults to True.

    Returns:
        nn sequential layers
    """    
    if batch_norm:
        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn(),
                      Dropout(dropout_ratio),BN(channels[i]))
                  for i in range(2, len(channels)-1)]

    elif layer_norm:
        layers = [
            Seq(Lin(channels[i - 1], channels[i]), act_fn(),Dropout(dropout_ratio),LN(channels[i]))
            for i in range(2, len(channels)-1)
        ]
    else:

        layers = [Seq(Lin(channels[i - 1], channels[i]), act_fn(),Dropout(dropout_ratio))
                  for i in range(2, len(channels)-1)]

    if(nerf):
        layers.insert(0,Seq(Lin(channels[0] * 2, channels[1],bias=True)))
    else:
        layers.insert(0,Seq(Lin(channels[0], channels[1],bias=False)))
    layers.append(Seq(Lin(channels[-2], channels[-1])))
    
    layers = Seq(*layers)

    return layers

class MLPRegression(nn.Module):
    def __init__(self, input_dims, output_dims, mlp_layers=[256, 128, 128], dropout_ratio=0.0, batch_norm=False, scale_mlp_units=1.0, act_fn=ELU,layer_norm=False, nerf=False):
        """Create an instance of mlp nn model

        Args:
            input_dims (int): number of channels
            output_dims (int): output channel size
            mlp_layers (list, optional): perceptrons in each layer. Defaults to [256, 128, 128].
            dropout_ratio (float, optional): dropout after every layer. Defaults to 0.0.
            batch_norm (bool, optional): batch norm after every layer. Defaults to False.
            scale_mlp_units (float, optional): Quick way to scale up and down the number of perceptrons, as this gets multiplied with values in mlp_layers. Defaults to 1.0.
            act_fn ([type], optional): activation function after every layer. Defaults to ELU.
            layer_norm (bool, optional): layer norm after every layer. Defaults to False.
            nerf (bool, optional): use positional encoding (x->[sin(x),cos(x)]). Defaults to False.
        """        
        super(MLPRegression, self).__init__()

        # Scale units:
        scaled_mlp_layers = [int(i * scale_mlp_units) for i in mlp_layers]
        scaled_mlp_layers.append(output_dims)
        scaled_mlp_layers.insert(0,input_dims)
        
        self.mlp_layers = MLP(scaled_mlp_layers, dropout_ratio,batch_norm=batch_norm,act_fn=act_fn, layer_norm=layer_norm, nerf=nerf)

        self.nerf = nerf
    
    def forward(self, x, *args):
        """forward pass on network."""        

        if(self.nerf):
            inp = torch.cat((torch.sin(x), torch.cos(x)),1)
        else:
            inp = x
        y = self.mlp_layers(inp)
        return y
    def reset_parameters(self):
        """Use this function to initialize weights. Doesn't help much for mlp.
        """        
        self.apply(weights_init)

def scale_to_base(data, norm_dict, key):
    """Scale the tensor back to the orginal units.  

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    scaled_data = torch.mul(data,norm_dict[key]['std']) + norm_dict[key]['mean']
    return scaled_data
    
def scale_to_net(data, norm_dict, key):
    """Scale the tensor network range

    Args:
        data (tensor): input tensor to scale
        norm_dict (Dict): normalization dictionary of the form dict={key:{'mean':,'std':}}
        key (str): key of the data

    Returns:
        tensor : output scaled tensor
    """    
    
    scaled_data = torch.div(data - norm_dict[key]['mean'],norm_dict[key]['std'])
    scaled_data[scaled_data != scaled_data] = 0.0
    return scaled_data
