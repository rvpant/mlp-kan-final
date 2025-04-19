import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
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
import sys
from networks import DenseNet  ## otherwise defined the MLP 
import matplotlib.pyplot as plt
import urllib

from deeponet_derivative import KANBranchNet, KANTrunkNet
from networks import *
import efficient_kan

class DeepONet(nn.Module):

    def __init__(self, branch_net, trunk_net):
        super().__init__()
        
        self.branch_net = branch_net
        self.trunk_net = trunk_net
    
    def forward(self, branch_inputs, trunk_inputs):
        
        branch_outputs = self.branch_net(branch_inputs)
        trunk_outputs = self.trunk_net(trunk_inputs)    
        results = torch.einsum('ik, lk -> il', branch_outputs, trunk_outputs)   
        return results

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(device):
    # load and process data
    train_data_path = 'nonlineardarcy_train.mat'
    test_data_path = 'nonlineardarcy_test.mat'
        
    data_train = loadmat(train_data_path)
    data_test = loadmat(test_data_path)    

    input_train = torch.from_numpy(data_train['f_train']).to(device).float()
    output_train = torch.from_numpy(data_train['u_train']).to(device).float()
    x_train = torch.from_numpy(data_train['x']).to(device).float().t()

    input_test = torch.from_numpy(data_test['f_test']).to(device).float()
    output_test = torch.from_numpy(data_test['u_test']).to(device).float()
    x_test = torch.from_numpy(data_test['x']).to(device).float().t()

    return input_train, output_train, x_train, input_test, output_test, x_test

def plot_results(predictions, ground_truth, x, output_dir, modeltype, num_instances=3):
    plt.figure(figsize=(12, 6))
        
    for i in range(num_instances):
        plt.subplot(1, num_instances, i + 1)
        plt.plot(x.detach().cpu().numpy(), ground_truth[i].detach().cpu().numpy(), label='Ground Truth', color='blue')
        plt.plot(x.detach().cpu().numpy(), predictions[i].detach().cpu().numpy(), label='Prediction', color='red', linestyle='dashed')
        plt.title(f'Instance {i+1}')
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.suptitle(f'1D Darcy {modeltype} Predictions')
    plt.savefig(f'{output_dir}/{modeltype}_predictions_and_true.png', dpi=400,bbox_inches='tight')
    plt.close()

def test_error_analysis(predictions, ground_truth, x, output_dir, modeltype):
    
    #error analysis of the individual prediction vs true errors.
    uhat = predictions.detach().cpu().numpy().copy() #array
    u = ground_truth.detach().cpu().numpy().copy()
    abs_errors = np.abs(uhat - u)
    l2_errors = []
    print(f"Predictions shape: {np.shape(uhat)}")
    print(f"True data shape: {np.shape(u)}")
    for pred, true in zip(uhat, u):
        # print(np.shape(pred), np.shape(true))
        num = np.linalg.norm(pred-true, ord=2)
        denom = np.linalg.norm(true)
        l2_errors.append(num/denom)
    print("Absolute and L2 error shapes:")
    print(np.shape(abs_errors), np.shape(l2_errors))
    print("#"*30)

    # can add functionality here to plot the worst-case prediction if needed.
    
    return abs_errors, l2_errors

def create_model(modeltype, mode, device, adaptive=False):
    input_neurons_branch = 50 #based on dataset sampling
    input_neurons_trunk = 1
    p = 100 #standardized across our operator learning problems.
    trunk_grid_size = 20

    #Defining the model, across the two different modes.
    if mode=='shallow':
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p], grid_size = trunk_grid_size)
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            branch_net = DenseNet(layersizes=[input_neurons_branch] + [1000]*1 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [1000]*1 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # trunk_net.to(device)
    elif mode=='deep':
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [100]*3 + [p])
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [100]*3 + [p], grid_size = 20)
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='cheby_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='jacobi_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='legendre_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='legendre_kan', layernorm=False)
        else:
            branch_net = DenseNet(layersizes=[input_neurons_branch] + [256]*3 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [256]*3 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # trunk_net.to(device)
    else:
        print('Invalid architecture mode argument passed: must be one of "shallow" or "deep" (default).')
        return
    model = DeepONet(branch_net, trunk_net)
    model.to(device)

    return model

def main():
    #Change here from Hassan's original: add command line argument for model type.
    model_parser = argparse.ArgumentParser()
    model_parser.add_argument('--model', dest='modeltype', type=str, default='densenet',
                            help='Model type.',
                            choices=['densenet', 'efficient_kan', 'original_kan', 'cheby', 'jacobi', 'legendre'])
    model_parser.add_argument('--mode', dest='mode', type=str, default='shallow',
                            help='Network architecture mode.',
                            choices=['shallow', 'deep'])
    model_parser.add_argument('-adaptive', action='store_true', help='Enables adaptive activations on DenseNet.')
    model_parser.add_argument('-tuning', action='store_true', help='Flag to indicate tuning mode.')
    model_parser.add_argument('--trunk_grid_size', type=int, default=5, help='Grid size for KAN trunk net.')
    model_parser.add_argument('--trunk_hidden_size', type=int, default=0, help='Hidden size for non-standard shallow trunk network')
    
    args = model_parser.parse_args()
    modeltype = args.modeltype
    mode = args.mode
    adaptive = args.adaptive
    trunk_grid_size = args.trunk_grid_size
    trunk_hidden_size = args.trunk_hidden_size #if this is 0, then we used the standard 2n+1, else we use the provided value.
    tuning= args.tuning

    print(f"Running 1D Darcy with modeltype {modeltype}, architecture: {mode}.")
    if mode=='densenet':
        print(f"Adaptive activations on DenseNet: {adaptive}.")
    if tuning:
        print(f"TUNING MODE: trunk hidden size = {trunk_hidden_size} and trunk grid size = {trunk_grid_size}.")
        print("A trunk hidden size of 0 means that we use the default 2n+1 in the trunk net.")

    seed=0
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device: ', device)

    #defining the output directory
    if adaptive:
        output_dir = os.path.join(os.getcwd(), f'1D_Darcy_DeepONet/{mode}_adaptive')
    elif tuning:
        output_dir = os.path.join(os.getcwd(), f'1D_Darcy_DeepONet/tuning/{mode}')
    else:
        output_dir = os.path.join(os.getcwd(), f'1D_Darcy_DeepONet/deep20/{mode}')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load and process data
    train_data_path = 'nonlineardarcy_train.mat'
    test_data_path = 'nonlineardarcy_test.mat'
        
    data_train = loadmat(train_data_path)
    data_test = loadmat(test_data_path)    

    input_train = torch.from_numpy(data_train['f_train']).to(device).float()
    output_train = torch.from_numpy(data_train['u_train']).to(device).float()
    x_train = torch.from_numpy(data_train['x']).to(device).float().t()

    input_test = torch.from_numpy(data_test['f_test']).to(device).float()
    output_test = torch.from_numpy(data_test['u_test']).to(device).float()
    x_test = torch.from_numpy(data_test['x']).to(device).float().t()

    input_neurons_branch = 50 #based on dataset sampling
    input_neurons_trunk = 1
    p = 100 #standardized across our operator learning problems.


    #Defining the model, across the two different modes.
    if mode=='shallow':
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [2*input_neurons_branch+1]*1 + [p])
            if tuning:
                if trunk_hidden_size == 0:
                    trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p], grid_size = trunk_grid_size)
                else:
                    trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [trunk_hidden_size]*1 + [p], grid_size = trunk_grid_size)
            else:
                trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [2*input_neurons_trunk+1]*1 + [p])
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='cheby_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='jacobi_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, 2*input_neurons_branch+1, p, modeltype='legendre_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, 2*input_neurons_trunk+1, p, modeltype='legendre_kan', layernorm=False)
        else:
            branch_net = DenseNet(layersizes=[input_neurons_branch] + [1000]*1 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [1000]*1 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # trunk_net.to(device)
    elif mode=='deep':
        if modeltype=='efficient_kan':
            branch_net = efficient_kan.KAN(layers_hidden=[input_neurons_branch] + [100]*3 + [p])
            trunk_net = efficient_kan.KAN(layers_hidden=[input_neurons_trunk] + [100]*3 + [p], grid_size = 20)
        elif modeltype == 'cheby':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='cheby_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='cheby_kan', layernorm=False)
        elif modeltype == 'jacobi':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='jacobi_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='jacobi_kan', layernorm=False)
        elif modeltype == 'legendre':
            branch_net = KANBranchNet(input_neurons_branch, [2*input_neurons_branch+1]*2, p, modeltype='legendre_kan', layernorm=False)
            trunk_net = KANTrunkNet(input_neurons_trunk, [2*input_neurons_trunk+1]*2, p, modeltype='legendre_kan', layernorm=False)
        else:
            branch_net = DenseNet(layersizes=[input_neurons_branch] + [256]*3 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # branch_net.to(device)
            trunk_net = DenseNet(layersizes=[input_neurons_trunk] + [256]*3 + [p], activation=nn.ReLU(), adapt_activation=adaptive) #nn.LeakyReLU() #nn.Tanh()
            # trunk_net.to(device)
    else:
        print('Invalid architecture mode argument passed: must be one of "shallow" or "deep" (default).')
        return
    model = DeepONet(branch_net, trunk_net)
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)

    # Training loop
    epochs = 4000
    batch_size = 32
    train_losses = []
    test_losses = []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(input_train.size(0))

        for i in range(0, input_train.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_input, batch_output, batch_x = input_train[indices], output_train[indices], x_train
            
            optimizer.zero_grad()
            predictions = model(batch_input, batch_x)
            loss = criterion(predictions, batch_output)
            loss.backward()
            optimizer.step()
        # scheduler.step()

        # Track training loss
        train_losses.append(loss.item())
        if (epoch+1) % (epochs/10) == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        predictions_test = model(input_test, x_test)
        test_loss = criterion(predictions_test, output_test)
        test_losses.append(test_loss.item())
        print(f'Test Loss: {test_loss.item():.6f}')
        
        # Calculate additional metrics
        rmse = torch.sqrt(test_loss).item()
        print(f'Test RMSE: {rmse:.6f}')

    # Save model
    if not tuning:
        torch.save(model.state_dict(), f'{output_dir}/deeponet_model_{modeltype}.pt')
        np.save(f'{output_dir}/deeponet_model_{modeltype}_loss_list.npy', np.asarray(train_losses))
        np.save(f'{output_dir}/deeponet_model_{modeltype}_test_loss_list.npy', np.asarray(test_losses))
    else:
        torch.save(model.state_dict(), f'{output_dir}/deeponet_model_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}.pt')
        np.save(f'{output_dir}/deeponet_model_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}_loss_list.npy', np.asarray(train_losses))
        np.save(f'{output_dir}/deeponet_model_{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}_test_loss_list.npy', np.asarray(test_losses))

    # Save loss plot.
    fig, ax = plt.subplots()
    ax.plot(np.arange(epochs), train_losses)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_yscale("log")
    ax.set_title(f"{modeltype} Training Loss")
    if not tuning:
        fig.savefig(f"{output_dir}/{modeltype}_train_loss.jpg")
    else:
        fig.savefig(f"{output_dir}/{modeltype}_grid{trunk_grid_size}_hidden{trunk_hidden_size}_train_loss.jpg")



    def plot_predictions(predictions, ground_truth, x, num_instances=3):
        plt.figure(figsize=(12, 6))
        
        for i in range(num_instances):
            plt.subplot(1, num_instances, i + 1)
            plt.plot(x.cpu().numpy(), ground_truth[i].cpu().numpy(), label='Ground Truth', color='blue')
            plt.plot(x.cpu().numpy(), predictions[i].cpu().numpy(), label='Prediction', color='red', linestyle='dashed')
            plt.title(f'Instance {i+1}')
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.legend()
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(f'1D Darcy {modeltype} Predictions')
        plt.savefig(f'{output_dir}/{modeltype}_predictions_and_true.png', dpi=400,bbox_inches='tight')
        plt.show()

    # Evaluation and plotting
    model.eval()
    with torch.no_grad():
        predictions_test = model(input_test, x_test)  
        # Plot the first 3 instances
        plot_predictions(predictions_test, output_test, x_test, num_instances=3)
        try:
            l2errs = np.linalg.norm(predictions_test.cpu().numpy() - output_test.cpu().numpy(), axis=1) / np.linalg.norm(output_test.cpu().numpy(), axis=1)
            print(f"Mean l2 error on test set: {np.mean(l2errs)}")
        except Exception as e:
            print(f"Error raised in L2 error calculation: {e}")
            pass

    print("Main run complete.")
    return

if __name__ == '__main__':
    main()