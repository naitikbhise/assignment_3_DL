from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
from torch.functional import F
from torch.optim import Adam
from torch.autograd import Variable

def get_data_loader(dataset_location, batch_size):
    URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
    # start processing
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    splitdata = []
    for splitname in ["train", "valid", "test"]:
        filename = "binarized_mnist_%s.amat" % splitname
        filepath = os.path.join(dataset_location, filename)
        utils.download_url(URL + filename, dataset_location)
        with open(filepath) as f:
            lines = f.readlines()
        x = lines_to_np_array(lines).astype('float32')
        x = x.reshape(x.shape[0], 1, 28, 28)
        # pytorch data loader
        dataset = data_utils.TensorDataset(torch.from_numpy(x))
        dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
        splitdata.append(dataset_loader)
    return splitdata

train1, valid, test = get_data_loader("binarized_mnist", 64)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 256, kernel_size=(5, 5)),
            nn.ELU())
        self.en_mean = nn.Linear(256, 100)
        self.en_log_var = nn.Linear(256, 100)
        self.fully = nn.Linear(100, 256)
        self.decoder = nn.Sequential(

            nn.ELU(),
            nn.Conv2d(256, 64, kernel_size=(5, 5), padding=4),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2),
            nn.ELU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 16, kernel_size=(3, 3), padding=2),
            nn.ELU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=2)
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)

        eva = self.en_mean(x)
        log_var = self.en_log_var(x)

        return eva, log_var

    def sampler(self, eva, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + eva

    def decode(self, z):
        z = self.fully(z)
        z = z.view(-1, 256, 1, 1)
        z = self.decoder(z)
        return z

    def forward(self, x):
        eva, log_var = self.encode(x)
        z = self.sampler(eva, log_var)
        return self.decode(z), eva, log_var


model = VAE()

print(model)

if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
else:
    device = torch.device("cpu")
    use_cuda = False

print(device)

model = VAE()
model = model.to(device)


def loss_function(recon_x, x, eva, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='sum')

    KLD = 0.5 * torch.sum(-1 - log_var + eva.pow(2) + log_var.exp())

    return BCE + KLD


def trainVAE(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train1):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, eva, log_var = model(data)
        # print(eva)
        # print(log_var)
        loss = loss_function(recon_batch, data, eva, log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    train_loss = - train_loss / len(train1.dataset)
    print('====> Epoch: {} ELBO for training: {:.4f}'.format(
        epoch, train_loss))

def testVAE(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(valid):
        data = data.to(device)
        recon_batch, eva, log_var = model(data)
        #print(eva)
        #print(log_var)
        loss = loss_function(recon_batch, data, eva, log_var)
        test_loss += loss.item()
    test_loss /= len(valid.dataset)
    print('====> Epoch: {} ELBO for validation: {:.4f}'.format(
          epoch, -test_loss))

def test1VAE(epoch):
    model.eval()
    test_loss = 0
    for batch_idx, data in enumerate(test):
        data = data.to(device)
        recon_batch, eva, log_var = model(data)
        #print(eva)
        #print(log_var)
        loss = loss_function(recon_batch, data, eva, log_var)
        test_loss += loss.item()
    test_loss /= len(test.dataset)
    print('====> Epoch: {} ELBO for test: {:.4f}'.format(
          epoch, -test_loss))

optimizer = torch.optim.Adam(model.parameters(), lr = 3*1e-4)
epochs = 20
for epoch in range(1, epochs + 1):
    trainVAE(epoch)
    testVAE(epoch)
    test1VAE(epoch)

model = VAE()
model = model.to(device)

from torch.distributions.normal import Normal
import torch.distributions as d
import math



def log_likelihood(model, data, K=200, L=100):
    log_probs = []  # build a new list to store the logp
    with torch.no_grad():
        for batch_idx, data in enumerate(data):
            data = data.to(device)  # link the data to the device
            eva, log_var = model.encode(data)  # get the mean and log-variance from decode
            eva = eva.to(device)
            log_var = log_var.to(device)  # link the mean and log-variance to device
            normal = d.Normal(eva, torch.exp(0.5 * log_var))  # get the normal distribution
            std_nor = d.Normal(torch.zeros(L).to(device),
                               torch.ones(L).to(device))  # get the standard normal distribution
            for _ in range(K):
                # eva, log_var = model(data)
                z = model.sampler(eva, log_var)
                recon_x = model.decode(z).to(device)
                # normal = d.Normal(eva, torch.exp(0.5 * log_var))
                logp_xz = -torch.nn.functional.binary_cross_entropy(torch.sigmoid(recon_x
                                                                                  ), data,
                                                                    reduction='sum')  # get logp(x|z)
                # a = log_prob(z)
                logp_z = torch.sum(std_nor.log_prob(z), dim=1)  # get logp(z)
                logq_zx = torch.sum(normal.log_prob(z), dim=1)  # get logq(z|x)
                logp_x1 = logp_xz + logp_z - logq_zx - math.log(
                    K)  # calculate logp(x) using the equation in the assignment
                logp_x1 = torch.max(logp_x1)  # find the maximum value
                log_probs.append(logp_x1)  # add the value of logp(x) in the list
            # print(log_probs)
            logp_x = torch.logsumexp(torch.stack(log_probs).to(device),
                                     0).mean()  # using logsumexp and get the mean value
            # print (logp_x)
    return logp_x

a = log_likelihood(model, test, K = 200, L = 100)
b = log_likelihood(model, valid, K = 200, L = 100)
print("log_likelihood for validation: {:.4f}".format(b/100))
print("log_likelihood for test: {:.4f}".format(a/100))
