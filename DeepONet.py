# %%
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import argparse
import random
import os
import time 
from termcolor import colored
from scipy.io import loadmat
import urllib.request

import sys
from deeponet_derivative import KANBranchNet, KANTrunkNet
from networks import *
import efficient_kan
# import kan #This is deprecated import of original KAN. 

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9})
import seaborn as sns
sns.set_style("white")
sns.set_style("ticks")

import warnings
warnings.filterwarnings("ignore")

# %%
class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs):
        """
        bs    :  Batch size.
        m     :  Number of sensors on each input IC field. # IC:initial condition
        neval :  Number of points at which output field is evaluated for a given input IC field sample = nt*nx
        p     :  Number of output neurons in both branch and trunk net.   
        
        branch inputs shape: (bs, m) 
        trunk inputs shape : (neval, 2) # 2 corresponds to t and x
        
        shapes:  inputs shape         -->      outputs shape
        branch:  (bs x m)             -->      (bs x p)
        trunk:   (neval x 2)          -->      (neval x p)
        
        outputs shape: (bs x neval).
        """
        
        branch_outputs = self.branch_net(branch_inputs)
        trunk_outputs = self.trunk_net(trunk_inputs)
        
        results = torch.einsum('ik, lk -> il', branch_outputs, trunk_outputs)
        
        return results
    
# %%
class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)

def download_data(data_path):
    '''Downloads the Burgers dataset.'''
    url = "https://drive.usercontent.google.com/download?id=10squkpIibNNOslxoOYzXV3YQ8AYk858t&export=download&authuser=0&confirm=t&uuid=ae9056a2-5076-4bde-9aa6-19d36c09055c&at=AENtkXY8hqWeFT1DySKsQrA7DfLj:1730901209755"
    filename = "Burger.mat"
    print(f'Downloading {filename} from {url}...')
    urllib.request.urlretrieve(url, data_path)
    print(f'{filename} downloaded.')

def create_model(modeltype, mode, device):
    '''
    Utility function that creates a specified DeepONet model and sends to device.
    Used in error analysis and other codes to load existing model weights.
    Loads in data in order to create model, so slightly redundant with load_data. 
    '''
    seed = 0
    try:
        data = loadmat('Burger.mat') # Load the .mat file
        #Data has tspan, input, output in its keys.
    except Exception as e:
        print("Creating model when data has not been downloaded.")
        print("Please use download_data function to fetch .mat files.")

    #print(data)
    # print(data['tspan'].shape)
    # print(data['input'].shape)  # Initial conditions: Gaussian random fields, Nsamples x 101, each IC sample is (1 x 101)
    # print(data['output'].shape) # Time evolution of the solution field: Nsamples x 101 x 101.
    #                             # Each field is 101 x 101, rows correspond to time and columns respond to location.
    #                             # First row corresponds to solution at t=0 (1st time step)
    #                             # and next  row corresponds to solution at t=0.01 (2nd time step) and so on.
    #                             # last row correspond to solution at t=1 (101th time step).

    # %%
    # Convert NumPy arrays to PyTorch tensors
    inputs = torch.from_numpy(data['input']).float().to(device)
    outputs = torch.from_numpy(data['output']).float().to(device)

    t_span = torch.from_numpy(data['tspan'].flatten()).float().to(device)
    x_span = torch.linspace(0, 1, data['output'].shape[2]).float().to(device)
    nt, nx = len(t_span), len(x_span) # number of discretizations in time and location.
    print("nt =",nt, ", nx =",nx)
    print("Shape of t-span and x-span:",t_span.shape, x_span.shape)
    print("t-span:", t_span)
    print("x-span:", x_span)

    # Estimating grid points
    T, X = torch.meshgrid(t_span, x_span)
    # print(T)
    # print(X)
    grid = torch.vstack((T.flatten(), X.flatten())).T
    print("Shape of grid:", grid.shape) # (nt*nx, 2)
    print("grid:", grid) # time, location

    # Split the data into training (2000) and testing (500) samples
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=500, random_state=seed)

    
    # Check the shapes of the subsets
    print("Shape of inputs_train:", inputs_train.shape)
    print("Shape of inputs_test:", inputs_test.shape)
    print("Shape of outputs_train:", outputs_train.shape)
    print("Shape of outputs_test:", outputs_test.shape)
    print('#'*100)

    """
    input_neurons_branch: Number of input neurons in the branch net.
    input_neurons_trunk: Number of input neurons in the trunk net.
    p: Number of output neurons in both the branch and trunk net.
    """
    p = 100 # Number of output neurons in both the branch and trunk net.

    input_neurons_branch = nx # m
    if mode=='shallow':
        if modeltype == 'efficient_kan':
            # branch_net = KAN(layers_hidden=[input_neurons_branch] + [100]*6 + [p])
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            # branch_net = DenseNet(layersizes=[input_neurons_branch] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net = DenseNet(layersizes=[input_neurons_branch]+[1000]+[p], activation=nn.SiLU())
    elif mode=='deep':
        if modeltype == 'efficient_kan':
            # branch_net = KAN(layers_hidden=[input_neurons_branch] + [100]*6 + [p])
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [100]*3 + [p])
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='legendre_kan', layernorm=False)
        else:
            # branch_net = DenseNet(layersizes=[input_neurons_branch] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net = DenseNet(layersizes=[input_neurons_branch]+[128]*6+[p], activation=nn.SiLU())
    else:
        print("Invalid architecture mode passed, must be one of 'shallow' or 'deep'.")
    branch_net.to(device)
    # print(branch_net)
    # print('BRANCH-NET SUMMARY:')
    # # summary(branch_net, input_size=(input_neurons_branch,))  
    # print('#'*100)

    # 2 corresponds to t and x
    input_neurons_trunk = 2
    trunk_grid_size = 20 #Modify this if needed.
    if mode=='deep':
        if modeltype == 'efficient_kan':
            # trunk_net = KAN(layers_hidden=[input_neurons_trunk] + [100]*6 + [p]) 
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [100]*3 + [p])
        elif modeltype == 'cheby':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='legendre_kan', layernorm=False)
        else:
            # trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net = DenseNet(layersizes=[input_neurons_trunk]+[128]*6+[p], activation=nn.SiLU())
    elif mode == 'shallow':
        if modeltype == 'efficient_kan':
            # trunk_net = KAN(layers_hidden=[input_neurons_trunk] + [100]*6 + [p]) 
            hidden = 2*input_neurons_trunk+1
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [hidden]*1 + [p], grid_size=trunk_grid_size)
        elif modeltype == 'cheby':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            # trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net = DenseNet(layersizes=[input_neurons_trunk]+[1000]+[p], activation=nn.SiLU())
    trunk_net.to(device)
    # print(trunk_net)
    # print('TRUNK-NET SUMMARY:')
    # # summary(trunk_net, input_size=(input_neurons_trunk,))
    # print('#'*100)

    model = DeepONet(branch_net, trunk_net)
    model.to(device)

    return model

def test_error_analysis(inputs_test, outputs_test, grid, nx, model, modeltype, output_dir):
    mse_list = []
    l2err_list = []

    for i in range(inputs_test.shape[0]):
        
        branch_inputs = inputs_test[i].reshape(1, nx) # (bs, m) = (1, nx) 
        trunk_inputs = grid # (neval, 2) = (nt*nx, 2)

        prediction_i = model(branch_inputs, trunk_inputs).cpu() # (bs, neval) = (1, nt*nx)
        target_i = outputs_test[i].reshape(1, -1).cpu()
        mse_i = F.mse_loss(prediction_i, target_i)
        mse_list.append(mse_i.item())

        # print(f"Pred shape: {prediction_i.shape}", f"Target shape: {target_i.shape}")

        num = np.linalg.norm(prediction_i.detach().numpy() - target_i.detach().numpy(), ord=2)
        denom = np.linalg.norm(target_i.detach().numpy(), ord=2)
        l2err_list.append(num/denom)
    
    return mse_list, l2err_list

def load_data(device):
    data = loadmat('data/Burger.mat') # Load the .mat file
    seed=0
    #print(data)
    print(data['tspan'].shape)
    print(data['input'].shape)  # Initial conditions: Gaussian random fields, Nsamples x 101, each IC sample is (1 x 101)
    print(data['output'].shape) # Time evolution of the solution field: Nsamples x 101 x 101.
                                # Each field is 101 x 101, rows correspond to time and columns respond to location.
                                # First row corresponds to solution at t=0 (1st time step)
                                # and next  row corresponds to solution at t=0.01 (2nd time step) and so on.
                                # last row correspond to solution at t=1 (101th time step).

    # %%
    # Convert NumPy arrays to PyTorch tensors
    inputs = torch.from_numpy(data['input']).float().to(device)
    outputs = torch.from_numpy(data['output']).float().to(device)

    t_span = torch.from_numpy(data['tspan'].flatten()).float().to(device)
    x_span = torch.linspace(0, 1, data['output'].shape[2]).float().to(device)
    nt, nx = len(t_span), len(x_span) # number of discretizations in time and location.
    print("nt =",nt, ", nx =",nx)
    print("Shape of t-span and x-span:",t_span.shape, x_span.shape)
    print("t-span:", t_span)
    print("x-span:", x_span)

    # Estimating grid points
    T, X = torch.meshgrid(t_span, x_span)
    # print(T)
    # print(X)
    grid = torch.vstack((T.flatten(), X.flatten())).T
    print("Shape of grid:", grid.shape) # (nt*nx, 2)
    print("grid:", grid) # time, location

    # Split the data into training (2000) and testing (500) samples
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=500, random_state=seed)


    print("Burgers data loaded.")
    return inputs_train, inputs_test, outputs_train, outputs_test, grid, nt, nx, T, X, t_span, x_span

def plot_results(model, modeltype, inputs_test, outputs_test, grid, nt, nx, T, X, t_span, x_span, plots_resultdir):
    for i in range(inputs_test.shape[0]):

        branch_inputs = inputs_test[i].reshape(1, nx) # (bs, m) = (1, nx) 
        trunk_inputs = grid # (neval, 2) = (nt*nx, 2)

        prediction_i = model(branch_inputs, trunk_inputs).cpu() # (bs, neval) = (1, nt*nx)
        target_i = outputs_test[i].reshape(1, -1).cpu()

        if (i+1) % 10 == 0:
            fig = plt.figure(figsize=(15,4))
            plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
            
            ax = fig.add_subplot(1, 4, 1)    
            ax.plot(x_span.cpu().detach().numpy(), inputs_test[i].cpu().detach().numpy())
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$s(t=0, x)$')
            plt.tight_layout()
            
            ax = fig.add_subplot(1, 4, 2)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), outputs_test[i].cpu().detach().numpy(), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$True \ field$',fontsize=14)
            plt.tight_layout()

            ax = fig.add_subplot(1, 4, 3)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), prediction_i.reshape(nt, nx).cpu().detach().numpy(), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$Predicted \ field$',fontsize=14)  
            plt.tight_layout()
            
            ax = fig.add_subplot(1, 4, 4)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), np.abs(outputs_test[i].cpu().detach().numpy() - prediction_i.reshape(nt, nx).cpu().detach().numpy()), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$Absolute \ error$',fontsize=14)  
            plt.tight_layout()

            # fig.suptitle(f"Example {i} Test Results")

            plt.savefig(os.path.join(plots_resultdir,'Test_Sample_'+str(i+1)+f'_{modeltype}.png'))
            plt.close()

def main():
    # %%
    cluster = False
    save = True

    #Change here from Hassan's original: add command line argument for model type.
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--model', dest='modeltype', type=str, default='densenet',
                            help='Model type.',
                            choices=['densenet', 'efficient_kan', 'original_kan', 'cheby', 'jacobi', 'legendre'])
    model_parser.add_argument('--mode', dest='mode', type=str, default='shallow',
                            help='Network architecture mode.',
                            choices=['shallow', 'deep'])
    model_parser.add_argument('--noise', dest='noise', type=float, default=0.0,
                            help='Input perturbation noise.')
    model_parser.add_argument('--lr', dest='lr',type=float, default=1e-4,
                                help='Learning rate for optimizer.')
    model_parser.add_argument('-scheduler', action='store_true', help='Enables learning rate step scheduler.')
    model_parser.add_argument('-tuning', action='store_true', help='Flag to indicate tuning mode.')
    model_parser.add_argument('--trunk_grid_size', type=int, default=5, help='Grid size for KAN trunk net.')
    model_parser.add_argument('--trunk_hidden_size', type=int, default=0, help='Hidden size for non-standard shallow trunk network')
    args = model_parser.parse_args()
    modeltype = args.modeltype
    mode = args.mode
    noise = args.noise
    lr = args.lr
    scheduler_enabled = args.scheduler
    tuning = args.tuning
    trunk_grid_size = args.trunk_grid_size #specify a grid size for the trunk network. 
    trunk_hidden_size = args.trunk_hidden_size #specify a hidden layer size for the trunk network. 0 if we use default 2n+1
    print(f"Running with modeltype {modeltype}, architecture mode {mode}.")
    print(f"Learning rate input is set as {lr}.")
    print(f"LR SCHEDULER: {scheduler_enabled}")
    if tuning:
        print(f"TUNING MODE: trunk grid {trunk_grid_size}, trunk hidden size {trunk_hidden_size}")
    # modeltype = "efficient_kan" # "densenet"  #

    # %%
    if cluster == True:
        parser = argparse.ArgumentParser()
        parser.add_argument('-seed', dest='seed', type=int, default=0, help='Seed number.')
        args = parser.parse_args()

        # Print all the arguments
        for arg in vars(args):
            print(f'{arg}: {getattr(args, arg)}')

        seed = args.seed
        
    if cluster == False:
        seed = 0 # Seed number.

    if save == True:
        if not tuning:
            resultdir = os.path.join(os.getcwd(), f'DeepONet_results/{mode}', 'seed='+str(seed))
            plots_resultdir = os.path.join(resultdir, f'plots/{mode}/{modeltype}') 
        else:
            resultdir = os.path.join(os.getcwd(), f'DeepONet_results/kan_tuning/{mode}', 'seed='+str(seed))
            plots_resultdir = os.path.join(resultdir, f'plots/kan_tuning/{mode}/{modeltype}') 
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        if not os.path.exists(plots_resultdir):
            os.makedirs(plots_resultdir)

    if save == True and cluster == True:
        orig_stdout = sys.stdout
        q = open(os.path.join(resultdir, 'output-'+'seed='+str(seed)+'.txt'), 'w')
        sys.stdout = q
        print ("------START------")
    
    if not noise==0.0:
        resultdir = os.path.join(resultdir, f'noise_{noise}')
        plots_resultdir = os.path.join(plots_resultdir, f'noise_{noise}')
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)
        if not os.path.exists(plots_resultdir):
            os.makedirs(plots_resultdir)

    print('seed = '+str(seed))

    # %%
    start = time.time()
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    data_path = 'Burger.mat'
    if not os.path.exists(data_path):
        download_data(data_path)

    # %%
    # Load the data
    data = loadmat('Burger.mat') # Load the .mat file
    #print(data)
    print(data['tspan'].shape)
    print(data['input'].shape)  # Initial conditions: Gaussian random fields, Nsamples x 101, each IC sample is (1 x 101)
    print(data['output'].shape) # Time evolution of the solution field: Nsamples x 101 x 101.
                                # Each field is 101 x 101, rows correspond to time and columns respond to location.
                                # First row corresponds to solution at t=0 (1st time step)
                                # and next  row corresponds to solution at t=0.01 (2nd time step) and so on.
                                # last row correspond to solution at t=1 (101th time step).

    # %%
    # Convert NumPy arrays to PyTorch tensors
    inputs = torch.from_numpy(data['input']).float().to(device)
    outputs = torch.from_numpy(data['output']).float().to(device)

    t_span = torch.from_numpy(data['tspan'].flatten()).float().to(device)
    x_span = torch.linspace(0, 1, data['output'].shape[2]).float().to(device)
    nt, nx = len(t_span), len(x_span) # number of discretizations in time and location.
    print("nt =",nt, ", nx =",nx)
    print("Shape of t-span and x-span:",t_span.shape, x_span.shape)
    print("t-span:", t_span)
    print("x-span:", x_span)

    # Estimating grid points
    T, X = torch.meshgrid(t_span, x_span)
    # print(T)
    # print(X)
    grid = torch.vstack((T.flatten(), X.flatten())).T
    print("Shape of grid:", grid.shape) # (nt*nx, 2)
    print("grid:", grid) # time, location

    # Split the data into training (2000) and testing (500) samples
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=500, random_state=seed)

    #Added by Raghav: random, noisy perturbation of the training inputs.
    # inputs_train = inputs_train + np.random.normal(loc=0.0, scale=noise, size=inputs_train.shape)
    inputs_noise = torch.randn_like(inputs_train)*noise
    inputs_train = inputs_train + inputs_noise
    
    # Check the shapes of the subsets
    print("Shape of inputs_train:", inputs_train.shape)
    print("Shape of inputs_test:", inputs_test.shape)
    print("Shape of outputs_train:", outputs_train.shape)
    print("Shape of outputs_test:", outputs_test.shape)
    print('#'*100)

    """
    input_neurons_branch: Number of input neurons in the branch net.
    input_neurons_trunk: Number of input neurons in the trunk net.
    p: Number of output neurons in both the branch and trunk net.
    """
    p = 100 # Number of output neurons in both the branch and trunk net.

    input_neurons_branch = nx # m
    if mode=='shallow':
        if modeltype == 'efficient_kan':
            # branch_net = KAN(layers_hidden=[input_neurons_branch] + [100]*6 + [p])
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            # branch_net = DenseNet(layersizes=[input_neurons_branch] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net = DenseNet(layersizes=[input_neurons_branch]+[1000]+[p], activation=nn.SiLU())
    elif mode=='deep':
        if modeltype == 'efficient_kan':
            # branch_net = KAN(layers_hidden=[input_neurons_branch] + [100]*6 + [p])
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [100]*3 + [p])
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*3, p, modeltype='legendre_kan', layernorm=False)
        else:
            # branch_net = DenseNet(layersizes=[input_neurons_branch] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            branch_net = DenseNet(layersizes=[input_neurons_branch]+[128]*6+[p], activation=nn.SiLU())
    else:
        print("Invalid architecture mode passed, must be one of 'shallow' or 'deep'.")
    branch_net.to(device)
    # print(branch_net)
    print('BRANCH-NET SUMMARY:')
    # summary(branch_net, input_size=(input_neurons_branch,))  
    print('#'*100)

    # 2 corresponds to t and x
    input_neurons_trunk = 2
    if mode=='deep':
        if modeltype == 'efficient_kan':
            # trunk_net = KAN(layers_hidden=[input_neurons_trunk] + [100]*6 + [p]) 
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [100]*3 + [p])
        elif modeltype == 'cheby':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*3, p, modeltype='legendre_kan', layernorm=False)
        else:
            # trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net = DenseNet(layersizes=[input_neurons_trunk]+[128]*6+[p], activation=nn.SiLU())
    elif mode == 'shallow':
        if modeltype == 'efficient_kan':
            # trunk_net = KAN(layers_hidden=[input_neurons_trunk] + [100]*6 + [p]) 
            if not tuning:
                trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p])
            else:
                hidden = 2*input_neurons_trunk+1 if trunk_hidden_size == 0 else trunk_hidden_size
                #Change the below grid size arg to trunk_grid_size if wanting to use the flag-provided value. We hard-code 20 after our experiments.
                trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [hidden]*1 + [p], grid_size=20)
        elif modeltype == 'cheby':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            # trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [100]*6 + [p], activation=nn.SiLU()) #nn.LeakyReLU() #nn.Tanh()
            trunk_net = DenseNet(layersizes=[input_neurons_trunk]+[1000]+[p], activation=nn.SiLU())
    trunk_net.to(device)
    # print(trunk_net)
    print('TRUNK-NET SUMMARY:')
    # summary(trunk_net, input_size=(input_neurons_trunk,))
    print('#'*100)

    model = DeepONet(branch_net, trunk_net)
    model.to(device)

    # %%
    def count_learnable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    num_learnable_parameters = count_learnable_parameters(branch_net) + count_learnable_parameters(trunk_net)
    print("Total number of learnable parameters:", num_learnable_parameters)

    # %%
    # print('Shape of train data')
    # print(inputs_train.shape, outputs_train.shape)
    # print('#'*100)

    bs = 256 # Batch size
    # Calculate the number of batches
    num_batches = len(inputs_train) // bs
            
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    iteration_list, loss_list, learningrates_list = [], [], []
    test_loss_list = []
    iteration = 0

    n_epochs = 25000 #1000 #800 # 10

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.85) # gamma=0.8

    for epoch in range(n_epochs):
        
        # Shuffle the train data using the generated indices
        num_samples = len(inputs_train)
        indices = torch.randperm(num_samples).to(device) # Generate random permutation of indices
        inputs_train_shuffled = inputs_train[indices]
        outputs_train_shuffled = outputs_train[indices]
        
        # Initialize lists to store batches
        inputs_train_batches = []
        outputs_train_batches = []
        # Split the data into batches
        for i in range(num_batches):
            start_idx = i * bs
            end_idx = (i + 1) * bs
            inputs_train_batches.append(inputs_train_shuffled[start_idx:end_idx])
            outputs_train_batches.append(outputs_train_shuffled[start_idx:end_idx])
        # Handle leftover data into the last batch
        if len(inputs_train_shuffled) % bs != 0:
            start_idx = num_batches * bs
            inputs_train_batches.append(inputs_train_shuffled[start_idx:])
            outputs_train_batches.append(outputs_train_shuffled[start_idx:])
        
        for i, (inputs_batch, outputs_batch) in enumerate(zip(inputs_train_batches, outputs_train_batches)):
            #print(f"Shape of inputs_train_batch[{i}]:", inputs_batch.shape) # (bs, nx)
            #print(f"Shape of outputs_train_batch[{i}]:", outputs_batch.shape) # (bs, nt, nx)
            
            branch_inputs = inputs_batch # (bs, m) = (bs, nx) 
            
            trunk_inputs = grid # (neval, 2) = (nt*nx, 2)
                
            outputs_needed = outputs_batch.reshape(-1, nt*nx) # (bs, neval) = (bs, nt*nx)

            # print(branch_inputs.shape, trunk_inputs.shape, outputs_needed.shape)   
            # print('*********')

            optimizer.zero_grad()
            predicted_values = model(branch_inputs, trunk_inputs) # (bs, nt*nx)
            target_values = outputs_needed # (bs, nt*nx)
            loss = nn.MSELoss()(predicted_values, target_values)
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print('Epoch %s:' % epoch, 'Batch %s:' % i, 'loss = %f,' % loss,
                    'learning rate = %f' % optimizer.state_dict()['param_groups'][0]['lr']) 
            
            iteration_list.append(iteration)
            loss_list.append(loss.item())
            learningrates_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            iteration+=1
        if scheduler_enabled:
            scheduler.step()
        
        #do predictions and get back test losses
        mse_list = []
        model.eval()
        for i in range(inputs_test.shape[0]):
        
            branch_inputs = inputs_test[i].reshape(1, nx) # (bs, m) = (1, nx) 
            trunk_inputs = grid # (neval, 2) = (nt*nx, 2)

            prediction_i = model(branch_inputs, trunk_inputs).cpu() # (bs, neval) = (1, nt*nx)
            target_i = outputs_test[i].reshape(1, -1).cpu()
            mse_i = F.mse_loss(prediction_i, target_i)
            mse_list.append(mse_i.item())
            test_loss_list.append(mse_i.item())

        if epoch % 10000 == 0 and save == True:
            if not tuning:
                torch.save(model.state_dict(), os.path.join(resultdir, f'model_state_dict_{modeltype}_epoch{epoch}.pt'))
            else:
                torch.save(model.state_dict(), os.path.join(resultdir, f'model_state_dict_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}_epoch{epoch}.pt'))
        
    if save == True:
        if not tuning:
            np.save(os.path.join(resultdir, f'iteration_list_{modeltype}.npy'), np.asarray(iteration_list))
            np.save(os.path.join(resultdir, f'loss_list_{modeltype}.npy'), np.asarray(loss_list))
            np.save(os.path.join(resultdir, f'test_loss_list_{modeltype}.npy'), np.asarray(test_loss_list))
            np.save(os.path.join(resultdir, f'learningrates_list_{modeltype}.npy'), np.asarray(learningrates_list))
        else:
            np.save(os.path.join(resultdir, f'iteration_list_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.npy'), np.asarray(iteration_list))
            np.save(os.path.join(resultdir, f'loss_list_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.npy'), np.asarray(loss_list))
            np.save(os.path.join(resultdir, f'test_loss_list_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.npy'), np.asarray(test_loss_list))
            np.save(os.path.join(resultdir, f'learningrates_list_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.npy'), np.asarray(learningrates_list))
        
    plt.figure()
    print(f"Test loss list length: {len(test_loss_list)}")
    plt.plot(iteration_list, loss_list, 'g', label = 'training loss', lw=0.75)
    # plt.plot(iteration_list, test_loss_list, 'r', label = 'test loss', lw=0.75)
    plt.yscale("log")
    plt.xlabel('Iterations')
    plt.ylabel('Training/Test loss')
    plt.legend()
    plt.tight_layout()
    plt.title(f'Training/Test Loss for {modeltype}')
    if save == True:
        if not tuning:
            plt.savefig(os.path.join(resultdir, f'loss_plot_{modeltype}.jpg'), bbox_inches='tight')
        else:
            plt.savefig(os.path.join(resultdir, f'loss_plot_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.jpg'), bbox_inches='tight')

    plt.figure()
    plt.plot(iteration_list, learningrates_list, 'b', label = 'learning-rate')
    plt.xlabel('Iterations')
    plt.ylabel('Learning-rate')
    plt.legend()
    plt.tight_layout()
    plt.title(f"Learning Rate For {modeltype}")
    if save == True:
        if not tuning:
            plt.savefig(os.path.join(resultdir, f'learning-rate_plot_{modeltype}.jpg'))
        else:
            plt.savefig(os.path.join(resultdir, f'learning-rate_plot_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.jpg'))
        
    # end timer
    finish = time.time() - start  # time for network to train

    # %%
    if save == True:
        if not tuning:
            torch.save(model.state_dict(), os.path.join(resultdir, f'model_state_dict_{modeltype}.pt'))
        else:
            torch.save(model.state_dict(), os.path.join(resultdir, f'model_state_dict_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.pt'))
    # model.load_state_dict(torch.load(os.path.join(resultdir,'model_state_dict.pt')))

    # %%
    # Predictions
    mse_list = []

    for i in range(inputs_test.shape[0]):
        
        branch_inputs = inputs_test[i].reshape(1, nx) # (bs, m) = (1, nx) 
        trunk_inputs = grid # (neval, 2) = (nt*nx, 2)

        prediction_i = model(branch_inputs, trunk_inputs).cpu() # (bs, neval) = (1, nt*nx)
        target_i = outputs_test[i].reshape(1, -1).cpu()
        mse_i = F.mse_loss(prediction_i, target_i)
        mse_list.append(mse_i.item())
        

        if (i+1) % 10 == 0:
            print(colored('TEST SAMPLE '+str(i+1), 'red'))
            
            r2score = metrics.r2_score(outputs_test[i].flatten().cpu().detach().numpy(), prediction_i.flatten().cpu().detach().numpy()) 
            relerror = np.linalg.norm(outputs_test[i].flatten().cpu().detach().numpy() - prediction_i.flatten().cpu().detach().numpy()) / np.linalg.norm(outputs_test[i].flatten().cpu().detach().numpy())
            r2score = float('%.4f'%r2score)
            relerror = float('%.4f'%relerror)
            print('Rel. L2 Error = '+str(relerror)+', R2 score = '+str(r2score))
            
            fig = plt.figure(figsize=(15,4))
            plt.subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.5, wspace = 0.4, hspace = 0.1)
            
            ax = fig.add_subplot(1, 4, 1)    
            ax.plot(x_span.cpu().detach().numpy(), inputs_test[i].cpu().detach().numpy())
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$s(t=0, x)$')
            plt.tight_layout()
            
            ax = fig.add_subplot(1, 4, 2)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), outputs_test[i].cpu().detach().numpy(), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$True \ field$',fontsize=14)
            plt.tight_layout()

            ax = fig.add_subplot(1, 4, 3)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), prediction_i.reshape(nt, nx).cpu().detach().numpy(), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$Predicted \ field$',fontsize=14)  
            plt.tight_layout()
            
            ax = fig.add_subplot(1, 4, 4)  
            plt.pcolor(X.cpu().detach().numpy(), T.cpu().detach().numpy(), np.abs(outputs_test[i].cpu().detach().numpy() - prediction_i.reshape(nt, nx).cpu().detach().numpy()), cmap='jet')
            plt.colorbar()
            ax.set_xlabel(r'$x$')
            ax.set_ylabel(r'$t$')
            plt.title('$Absolute \ error$',fontsize=14)  
            plt.tight_layout()

            if save == True:
                plt.savefig(os.path.join(plots_resultdir,'Test_Sample_'+str(i+1)+f'_{modeltype}.png'))
                # plt.show()
                plt.close()
            if save == False:
                plt.show()

            print(colored('#'*230, 'green'))

    mse = sum(mse_list) / len(mse_list)
    print("Mean Squared Error Test:\n", mse)

    # %%
    print("Time (sec) to complete:\n" +str(finish)) # time for network to train
    if save == True and cluster == True:
        print ("------END------")
        sys.stdout = orig_stdout
        q.close()


if __name__ == '__main__':
    main()