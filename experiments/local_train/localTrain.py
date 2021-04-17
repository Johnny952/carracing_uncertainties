from mixtureDist import Mixture
from model import Net

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF



BATCH_SIZE = 32
EPOCH = 100

range_ = (-1, 14)
n_grid = 2000
n_train = 3200
prob = 0.2

kl_sampling = 1000
import torch.nn as nn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(0)
if use_cuda:
    torch.cuda.manual_seed(0)


def sampler(num, mean1=2, std1=0.25, mean2=8, std2=1, p=0.5):
    p = (torch.rand(num) > p).float()

    mean1 = torch.from_numpy(np.array([mean1]*num)).float()
    mean2 = torch.from_numpy(np.array([mean2]*num)).float()
    std1 = torch.from_numpy(np.array([std1]*num)).float()
    std2 = torch.from_numpy(np.array([std2]*num)).float()

    samples1 = torch.normal(mean1, std1)
    samples2 = torch.normal(mean2, std2)
    
    return p*samples1 + (1-p)*samples2

def real_funcion(x):
    return x + 6*torch.sin(x) #+ 0.2*torch.rand(x.size())




x = torch.unsqueeze(torch.linspace(range_[0], range_[1], n_grid), dim=1)  # x data (tensor), shape=(100, 1)
y = real_funcion(x)
x_train = torch.unsqueeze(sampler(n_train), dim=1)
y_train = real_funcion(x_train)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)
plt.figure('Figura1')#, figsize=(10,4)
plt.plot(x.data.numpy(), y.data.numpy(), color = "blue")
plt.scatter(x_train.data.numpy(), y_train.data.numpy(), color = "red")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
#plt.savefig('curve_2.png')









kernel = ConstantKernel(8.0, constant_value_bounds="fixed") * RBF(0.3, length_scale_bounds="fixed")

gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(x_train.cpu().data.numpy(), y_train.cpu().data.numpy())
(gpr_mean_train, gpr_std_train) = gpr.predict(x_train.cpu().data.numpy(), return_std=True)
(gpr_mean, gpr_std) = gpr.predict(x.cpu().data.numpy(), return_std=True)

fig, ax = plt.subplots()#
plt.cla()
ax.set_title('Análisis de incertezas con Procesos Gaussianos')#, fontsize=35
ax.set_xlabel('Variable dependiente')#, fontsize=24
ax.set_ylabel('Variable independiente')#, fontsize=24
#ax.set_xlim(-11.0, 13.0)
#ax.set_ylim(-1.1, 1.2)
ax.plot(x.data.numpy(), y.data.numpy(), label="Real", color = "blue", alpha=1)
ax.plot(x.data.numpy(), gpr_mean.squeeze(), label="Predicción Grilla", color = "green", alpha=0.6)
#ax.scatter(x_train.data.numpy(), y_train.data.numpy(), label="Entrenamiento Real", color = "red", alpha=0.5)
ax.scatter(x_train.data.numpy(), gpr_mean_train.squeeze(), label="Predicción Entrenamiento", color = "red", alpha=0.5, s=10)
ax.fill_between(x.squeeze().data.numpy(), (gpr_mean.squeeze() - gpr_std), (gpr_mean.squeeze() + gpr_std), color="green", alpha=0.2, label="Incerteza")
#with torch.no_grad():
#    prediction = net(x.to(device))     # input x and predict based on x
#ax.plot(x.data.numpy(), prediction.cpu().data.numpy(), label="Predicción", color='green', alpha=0.5)
#with torch.no_grad():
#    prediction = net(x_train.to(device))     # input x and predict based on x
#ax.scatter(x_train.data.numpy(), prediction.cpu().data.numpy(), label="Predicción Entrenamiento", color='red', alpha=1)
#plt.savefig('curve_2_model_3_batches.png')
plt.legend()
#plt.show()




def loss_(x, net, kl_sampling=20):
    mix = Mixture(x, dev=0.05, device=device)   

    samples = mix.sample(kl_sampling)
    logp = mix.logp(samples)
    _, net_logp = net(samples)

    mseloss = nn.MSELoss()
    return mseloss(logp.float(), net_logp.float().squeeze())

def train(EPOCH, net, x_train, y_train, BATCH_SIZE, lr=0.01, kl_sampling=20):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    torch_dataset = Data.TensorDataset(x_train, y_train)

    loader = Data.DataLoader(
        dataset=torch_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=2,)

    #my_images = []
    #fig, ax = plt.subplots(figsize=(16,10))

    loss_array = []

    # start training
    for epoch in range(EPOCH):
        running_loss = 0
        for step, (batch_x, batch_y) in enumerate(loader): # for each training step
            
            b_x = Variable(batch_x).to(device)
            b_y = Variable(batch_y).to(device)

            prediction, _ = net(b_x)     # input x and predict based on x

            loss = loss_func(prediction, b_y)     # must be (1. nn output, 2. target)
            loss += 5*loss_(b_x, net, kl_sampling=kl_sampling)

            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients

            running_loss += loss
            """
            if step == 1:
                # plot and show learning process
                plt.cla()
                ax.set_title('Regression Analysis - model 3 Batches', fontsize=35)
                ax.set_xlabel('Independent variable', fontsize=24)
                ax.set_ylabel('Dependent variable', fontsize=24)
                #ax.set_xlim(-11.0, 13.0)
                #ax.set_ylim(-1.1, 1.2)
                ax.scatter(b_x.cpu().data.numpy(), b_y.cpu().data.numpy(), color = "blue", alpha=0.2)
                ax.scatter(b_x.cpu().data.numpy(), prediction.cpu().data.numpy(), color='green', alpha=0.5)
                ax.text(8, 4, 'Epoch = %d' % (epoch+1),
                        fontdict={'size': 24, 'color':  'red'})
                ax.text(8, 2, 'Loss = %.4f' % loss.cpu().data.numpy(),
                        fontdict={'size': 24, 'color':  'red'})

                # Used to return the plot as an image array 
                # (https://ndres.me/post/matplotlib-animated-gifs-easily/)
                fig.canvas.draw()       # draw the canvas, cache the renderer
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                my_images.append(image)
            """
        loss_array.append(running_loss.cpu().data.numpy())
        print("\rEpoch: {}/{}\tLoss: {}".format(epoch+1, EPOCH, running_loss), end="")
    print("\n")
    return np.stack(loss_array)

net = Net().to(device)
train(EPOCH, net, x_train, y_train, BATCH_SIZE, lr=0.01, kl_sampling=kl_sampling)

with torch.no_grad():
    pred_x = net(x.to(device))
    pred_trainx = net(x_train.to(device))

uncertainties =  2*(1 - np.exp(pred_x[1].cpu().data.numpy()))

fig, ax = plt.subplots()#
plt.cla()
ax.set_title('Análisis de incertezas con Red Neuronal')#, fontsize=35
ax.set_xlabel('Variable dependiente')#, fontsize=24
ax.set_ylabel('Variable independiente')#, fontsize=24
ax.plot(x.data.numpy(), y.data.numpy(), label="Real", color = "blue", alpha=1)
ax.plot(x.data.numpy(), pred_x[0].cpu().data.numpy(), label="Predicción Grilla", color = "green", alpha=0.6)
ax.scatter(x_train.data.numpy(), pred_trainx[0].cpu().data.numpy(), label="Predicción Entrenamiento", color = "red", alpha=0.5, s=10)

ax.fill_between(x.squeeze().data.numpy(), (pred_x[0].cpu().data.numpy() - uncertainties).squeeze(), (pred_x[0].cpu().data.numpy() + uncertainties).squeeze(), color="green", alpha=0.2, label="Incerteza")
plt.legend()




plt.figure()
plt.plot(x.cpu().data.numpy(), uncertainties/2)



plt.show()