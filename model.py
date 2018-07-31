"""
Code for the model structures.
"""
import os

import pickle

import math
import torch
from torch.autograd import Variable
from torch.nn import Module, Conv2d, MaxPool2d, ConvTranspose2d, BatchNorm2d, Parameter
from torch.nn.functional import leaky_relu, tanh

from settings3 import Settings
from hardware import load, gpu


class Discriminator1(Module):
    """
    A CNN that produces a density map and a count.
    """
    def __init__(self):
        super().__init__()

        self.conv1 = Conv2d(3, 32, kernel_size=7, padding=3)
        self.max_pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(self.conv1.out_channels, 32, kernel_size=7, padding=3)
        self.max_pool2 = MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = Conv2d(self.conv2.out_channels, 64, kernel_size=5, padding=2)
 
        self.convbigk = Conv2d(30, 1000, kernel_size=18)
        self.conv5 = Conv2d(self.convbigk.out_channels, 400, kernel_size=1)
        
        self.count_conv = Conv2d(self.conv5.out_channels, 1, kernel_size=1)
        
        self.density_conv = Conv2d(self.conv5.out_channels, 324, kernel_size=1)

        self.feature_layer = None
        
        self.branchx11 = Conv2d(3, 16, kernel_size=9, padding=4)
        self.max_pool11 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx12 = Conv2d(16, 32, kernel_size=7, padding=3)
        self.max_pool12 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx13 = Conv2d(32, 16, kernel_size=7, padding=3)
        self.branchx14 = Conv2d(16, 8, kernel_size=7, padding=3)
        
        self.branchx21 = Conv2d(3, 20, kernel_size=7, padding=3)
        self.max_pool21 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx22 = Conv2d(20, 40, kernel_size=5, padding=2)
        self.max_pool22 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx23 = Conv2d(40, 20, kernel_size=5, padding=2)
        self.branchx24 = Conv2d(20, 10, kernel_size=5, padding=2)
        
        self.branchx31 = Conv2d(3, 24, kernel_size=5, padding=2)
        self.max_pool31 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx32 = Conv2d(24, 48, kernel_size=3, padding=1)
        self.max_pool32 = MaxPool2d(kernel_size=2, stride=2)
        self.branchx33 = Conv2d(48, 24, kernel_size=3, padding=1)
        self.branchx34 = Conv2d(24, 12, kernel_size=3, padding=1)
        
        self.fuse = Conv2d(30, 1, kernel_size=1, padding=0)


    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        """
        The forward pass of the network.

        :param x: The input images.
        :type x: torch.autograd.Variable
        :return: The predicted density labels.
        :rtype: torch.autograd.Variable
        """
        x1=x
        x2=x
        x3=x
         
        x11 = leaky_relu (self.branchx11(x1))
#        print('branchx11 =',x11.shape)
        m11 = self.max_pool11(x11)
#        print('max_pool11 =',m11.shape)
        x12 = leaky_relu ( self.branchx12(m11))
#        print('branchx12 =',x12.shape)
        m12 = self.max_pool12(x12)
#        print('max_pool12 =',m12.shape)
        x13 = leaky_relu (self.branchx13(m12))
#        print('branchx13 =',x13.shape)
        x14 = leaky_relu (self.branchx14(x13))
#        print('branchx14 =',x14.shape)
        
#        print('branch 2 begins ________________________')
        
        x21 =leaky_relu (self.branchx21(x2))
#        print('branchx21 =',x21.shape)
        m21 = self.max_pool21(x21)
#        print('max_pool21 =',m21.shape)
        x22 = leaky_relu (self.branchx22(m21))
#        print('branchx22 =',x22.shape)
        m22 = self.max_pool22(x22)
#        print('max_pool22 =',m22.shape)
        x23 =leaky_relu (self.branchx23(m22))
#        print('branchx23 =',x23.shape)
        x24 =leaky_relu (self.branchx24(x23))
#        print('branchx24 =',x24.shape)        
  
#        print('branch 3 begins ________________________')      
        
        x31 =leaky_relu (self.branchx31(x3))
#        print('branchx31 =',x31.shape)
        m31 = self.max_pool31(x31)
#        print('max_pool31 =',m31.shape)
        x32 =leaky_relu (self.branchx32(m31))
#        print('branchx32 =',x32.shape)
        m32 = self.max_pool32(x32)
#        print('max_pool32 =',m32.shape)
        x33 =leaky_relu (self.branchx33(m32))
#        print('branchx33 =',x33.shape)
        x34 =leaky_relu (self.branchx34(x33))
#        print('branchx34 =',x34.shape)        
        
        xconcat = torch.cat((x14,x24,x34),1)
#        print('xconcat =',xconcat.shape) 
        
        xer = leaky_relu(self.convbigk(xconcat))
#        print('biggest xer =', xer.shape)

        xfuse = self.fuse(xconcat)
#        print('xfuse =',xfuse.shape) 
        
#        print('x input = ',x.shape)
#        x = leaky_relu(self.conv1(x))
#        print('x leaky relu = ',x.shape)
#
#        x = self.max_pool1(x)
#        print('x max pool = ',x.shape)
#
#        x = leaky_relu(self.conv2(x))
#        print('x leaky relu again = ',x.shape)
#
#        x = self.max_pool2(x)
#        print('x max pool2 = ',x.shape)
#
#        x = leaky_relu(self.conv3(x))
##        print('leaky relu1 = ',x.shape)
#
#        x = leaky_relu(self.conv4(x))
#        print('leaky relu2 =', x.shape)
#   
        passed = leaky_relu(self.conv5(xer))
#        print('pass =', passed.shape)
        
        self.feature_layer = x
#        print('feature layer = ', x.shape)
        
        x_count = leaky_relu(self.count_conv(passed)).view(-1)
#        print('x-count =',x_count.shape)
#       x_density = xfuse

        x_density = leaky_relu(self.density_conv(passed)).view(-1, 18, 18)
#        print('jointcnn output shape =',x_density.shape)
        return x_density, x_count


class Conv2df(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = leaky_relu(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Generator(Module):
    """
    A generator for producing crowd images.
    """
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = ConvTranspose2d(100, 64, kernel_size=18)
        self.conv_transpose2 = ConvTranspose2d(self.conv_transpose1.out_channels, 32, kernel_size=4, stride=2,
                                               padding=1)
        self.conv_transpose3 = ConvTranspose2d(self.conv_transpose2.out_channels, 3, kernel_size=4, stride=2,
                                               padding=1)

    def forward(self, z):
        """
        The forward pass of the generator.

        :param z: The input images.
        :type z: torch.autograd.Variable
        :return: Generated images.
        :rtype: torch.autograd.Variable
        """
        z = z.view(-1, 100, 1, 1)
        z = leaky_relu(self.conv_transpose1(z))
        z = leaky_relu(self.conv_transpose2(z))
        z = tanh(self.conv_transpose3(z))
#        print('z=',z.shape)
        return z

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)


class Predictor(Module):
    """
    The extra predictor layer.
    """
    def __init__(self):
        super().__init__()
        self.exponent = Parameter(torch.Tensor([0]))
        self.register_buffer('e', torch.Tensor([math.e]))

    def forward(self, y):
        """
        The forward pass of the predictor.

        :param y: Person counts.
        :type y: torch.autograd.Variable
        :return: Scaled person counts.
        :rtype: torch.autograd.Variable
        """
        y = y * Variable(self.e).pow(self.exponent)
        return y

    def __call__(self, *args, **kwargs):
        """
        Defined in subclass just to allow for type hinting.

        :return: The predicted labels.
        :rtype: torch.autograd.Variable
        """
        return super().__call__(*args, **kwargs)


class GAN(Module):
    """
    The full GAN.
    """
    def __init__(self):
        super().__init__()
        self.D = Discriminator1()
        self.G = Generator()
        self.P = Predictor()

    def forward(self, x):
        """Forward pass not implemented here."""
        raise NotImplementedError


def save_trainer(trial_directory, model, optimizer, epoch, step, prefix=None):
    """
    Saves all the information needed to continue training.

    :param trial_directory: The directory path to save the data to.
    :type trial_directory: str
    :param model: The model to save.
    :type model: torch.nn.Module
    :param optimizer: The optimizer to save.
    :type optimizer: torch.optim.optimizer.Optimizer
    :param epoch: The number of epochs completed.
    :type epoch: int
    :param step: The number of steps completed.
    :type step: int
    :param prefix: A prefix to append to the model file names.
    :type prefix: str
    """
    model_path = 'model {}'.format(epoch)
    optimizer_path = 'optimizer {}'.format(epoch)
    meta_path = 'meta {}'.format(epoch)
    if prefix:
        model_path = prefix + ' ' + model_path
        optimizer_path = prefix + ' ' + optimizer_path
        meta_path = prefix + ' ' + meta_path
    torch.save(model.state_dict(), os.path.join(trial_directory, model_path))
    torch.save(optimizer.state_dict(), os.path.join(trial_directory, optimizer_path))
    with open(os.path.join(trial_directory, meta_path), 'wb') as pickle_file:
        pickle.dump({'epoch': epoch, 'step': step}, pickle_file)


def load_trainer(prefix=None, settings=None):
    """
    Saves all the information needed to continue training.

    :param prefix: A prefix to append to the model file names.
    :type prefix: str
    :return: The model and optimizer state dict and the metadata for the training run.
    :rtype: dict[torch.Tensor], dict[torch.Tensor], int, int
    """
    if settings is None:
        settings = Settings()

############### model path and optimizer path , replacing model with optimizer and meta########################

    model_path = settings.load_model_path
    optimizer_path = settings.load_model_path.replace('model', 'optimizer')
    meta_path = settings.load_model_path.replace('model', 'meta')

############################ adding prefix to the path #####################################

    if prefix:
        model_path = os.path.join(os.path.split(model_path)[0], prefix + ' ' + os.path.split(model_path)[1])
        optimizer_path = os.path.join(os.path.split(optimizer_path)[0], prefix + ' ' + os.path.split(optimizer_path)[1])
        meta_path = os.path.join(os.path.split(meta_path)[0], prefix + ' ' + os.path.split(meta_path)[1])
######################## Loading path ####################
    model_state_dict = load(model_path)
    optimizer_state_dict = torch.load(optimizer_path)
####################### Open pickle file #################
    with open(meta_path, 'rb') as pickle_file:
        metadata = pickle.load(pickle_file)
    if settings.restore_mode == 'continue':
        step = metadata['step']
        epoch = metadata['epoch']
    else:
        step = 0
        epoch = 0
    return model_state_dict, optimizer_state_dict, epoch, step
