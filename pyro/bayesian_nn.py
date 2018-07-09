import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data = pd.read_csv('ethusd_tweets.csv')[::-1]

N = 360

highp = pd.to_numeric(data.loc[:, 'High'][-N:])
lowp = pd.to_numeric(data.loc[:, 'Low'][-N:])
openp = pd.to_numeric(data.loc[:, 'Open'][-N:])
closep = pd.to_numeric(data.loc[:, 'Close'][-N:])
tweets = pd.to_numeric(data.loc[:, 'Tweets'].replace('null', 0)[-N:])
volume = pd.to_numeric(data.loc[:, 'Volume'][-N:])
marketcap = pd.to_numeric(data.ix[:, 'Market Cap'][-N:])

normal_close = closep

highp = highp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
lowp = lowp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
openp = openp.pct_change().replace(np.nan, 0).replace(np.inf, 0)
closep = closep.pct_change().replace(np.nan, 0).replace(np.inf, 0)
tweets = tweets.pct_change().replace(np.nan, 0).replace(np.inf, 0)
volume = volume.pct_change().replace(np.nan, 0).replace(np.inf, 0)
marketcap = marketcap.pct_change().replace(np.nan, 0).replace(np.inf, 0)

normal_close = np.array(normal_close)
highp = np.array(highp)
lowp = np.array(lowp)
openp = np.array(openp)
closep = np.array(closep)
tweets = np.array(tweets)
volume = np.array(volume)
marketcap = np.array(marketcap)

WINDOW = 7
STEP = 1
FORECAST = 1

X, Y = [], []
for i in range(0, len(openp), STEP):
    try:
        o = openp[i:i+WINDOW]
        h = highp[i:i+WINDOW]
        l = lowp[i:i+WINDOW]
        c = closep[i:i+WINDOW]
        v = volume[i:i+WINDOW]
        t = tweets[i:i+WINDOW]
        m = marketcap[i:i+WINDOW]

#         y_i = (normal_close[i+WINDOW+FORECAST] - normal_close[i+WINDOW]) / normal_close[i+WINDOW]
        y_i = closep[i+WINDOW+FORECAST]
        x_i = np.column_stack((o, h, l, c, v, t, m))
        x_i = x_i.flatten()

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)

import torch
import torch.nn as nn
from torch.nn.functional import normalize  # noqa: F401
import torch.nn.functional as F

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal, Bernoulli  # noqa: F401
from pyro.infer import SVI
from pyro.optim import Adam
from pyro.infer import Trace_ELBO

X_train, Y_train = X[:-30], Y[:-30]

for i in range(len(X_train)):
    np.insert(X_train[i], len(X_train[0]), Y_train[i])

data = Variable(torch.Tensor(X_train))

pyro.get_param_store().clear()
X_train, Y_train = Variable(torch.Tensor(X[:-30])), Variable(torch.Tensor(Y[:-30]))
X_test, Y_test = Variable(torch.Tensor(X[-30:])), Variable(torch.Tensor(Y[-30:]))
#data = torch.cat((X_train, Y_train), 1)


# get array of batch indices
def get_batch_indices(N, batch_size):
    all_batches = np.arange(0, N, batch_size)
    if all_batches[-1] != N:
        all_batches = list(all_batches) + [N]
    return all_batches

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, 1)   # output layer

    def forward(self, x):
        x = self.hidden(x)
        x = self.predict(x)
        return x

first_layer = len(X_train.data.numpy()[0])-1
second_layer = 25

softplus = nn.Softplus()
regression_model = Net(first_layer, second_layer)

def model(data):

    mu = Variable(torch.zeros(second_layer, first_layer)).type_as(data)
    sigma = Variable(torch.ones(second_layer, first_layer)).type_as(data)
    bias_mu = Variable(torch.zeros(second_layer)).type_as(data)
    bias_sigma = Variable(torch.ones(second_layer)).type_as(data)
    w_prior, b_prior = Normal(mu, sigma), Normal(bias_mu, bias_sigma)

    mu2 = Variable(torch.zeros(1, second_layer)).type_as(data)
    sigma2 = Variable(torch.ones(1, second_layer)).type_as(data)
    bias_mu2 = Variable(torch.zeros(1)).type_as(data)
    bias_sigma2 = Variable(torch.ones(1)).type_as(data)
    w_prior2, b_prior2 = Normal(mu2, sigma2), Normal(bias_mu2, bias_sigma2)

    priors = {'hidden.weight': w_prior,
              'hidden.bias': b_prior,
              'predict.weight': w_prior2,
              'predict.bias': b_prior2}

    # lift module parameters to random variables sampled from the priors
    lifted_module = pyro.random_module("module", regression_model, priors)
    # sample a regressor (which also samples w and b)
    lifted_reg_model = lifted_module()

    with pyro.iarange("map", N, subsample=data):
        x_data = data[:, :-1]
        y_data = data[:, -1]
        # run the regressor forward conditioned on inputs
        prediction_mean = lifted_reg_model(x_data).squeeze()
        pyro.sample("obs",
                    Normal(prediction_mean, Variable(torch.ones(data.size(0))).type_as(data)),
                    obs=y_data.squeeze())


def guide(data):

    w_mu = Variable(torch.randn(second_layer, first_layer).type_as(data.data), requires_grad=True)
    w_log_sig = Variable(0.1 * torch.ones(second_layer, first_layer).type_as(data.data), requires_grad=True)
    b_mu = Variable(torch.randn(second_layer).type_as(data.data), requires_grad=True)
    b_log_sig = Variable(0.1 * torch.ones(second_layer).type_as(data.data), requires_grad=True)

    # register learnable params in the param store
    mw_param = pyro.param("guide_mean_weight", w_mu)
    sw_param = softplus(pyro.param("guide_log_sigma_weight", w_log_sig))
    mb_param = pyro.param("guide_mean_bias", b_mu)
    sb_param = softplus(pyro.param("guide_log_sigma_bias", b_log_sig))

    # gaussian guide distributions for w and b
    w_dist = Normal(mw_param, sw_param)
    b_dist = Normal(mb_param, sb_param)

    w_mu2 = Variable(torch.randn(1, second_layer).type_as(data.data), requires_grad=True)
    w_log_sig2 = Variable(0.1 * torch.randn(1, second_layer).type_as(data.data), requires_grad=True)
    b_mu2 = Variable(torch.randn(1).type_as(data.data), requires_grad=True)
    b_log_sig2 = Variable(0.1 * torch.ones(1).type_as(data.data), requires_grad=True)

    # register learnable params in the param store
    mw_param2 = pyro.param("guide_mean_weight2", w_mu2)
    sw_param2 = softplus(pyro.param("guide_log_sigma_weight2", w_log_sig2))
    mb_param2 = pyro.param("guide_mean_bias2", b_mu2)
    sb_param2 = softplus(pyro.param("guide_log_sigma_bias2", b_log_sig2))

    # gaussian guide distributions for w and b
    w_dist2 = Normal(mw_param2, sw_param2)
    b_dist2 = Normal(mb_param2, sb_param2)

    dists = {'hidden.weight': w_dist,
              'hidden.bias': b_dist,
              'predict.weight': w_dist2,
              'predict.bias': b_dist2}

    # overloading the parameters in the module with random samples from the guide distributions
    lifted_module = pyro.random_module("module", regression_model, dists)
    # sample a regressor
    return lifted_module()

# instantiate optim and inference objects
optim = Adam({"lr": 0.001})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

N = len(X_train)

for j in range(3000):
    epoch_loss = 0.0
    perm = torch.randperm(N)
    data = data[perm]
    all_batches = get_batch_indices(N, 64)
    for ix, batch_start in enumerate(all_batches[:-1]):
        batch_end = all_batches[ix+1]
        batch_data = data[batch_start:batch_end]
        epoch_loss += svi.step(batch_data)
    if j% 100:
        print(j, "avg loss {}".format(epoch_loss/float(N)))

preds = []
for i in range(100):
    new_data = X_test.data.numpy()
    new_data_ = []
    for arr in new_data:
        arr = np.delete(arr,-1)
        new_data_.append(arr)
    new_data_ = np.array(new_data_)
    new_data_ = Variable(torch.Tensor(new_data_))
    print (new_data_.shape)
    sampled_reg_model = guide(new_data_)
    pred = sampled_reg_model(new_data_).data.numpy().flatten()
    preds.append(pred)


preds = np.array(preds)
mean = np.mean(preds, axis=0)
std = np.std(preds, axis=0) / 10
y_test = Y_test.data.numpy()
x = np.arange(len(y_test))

plt.figure()
plt.plot(x, y_test)
plt.plot(x, mean, linestyle = '--')
plt.fill_between(x, mean-std, mean+std, alpha = 0.3, color = 'orange')
plt.show()

#t = np.random.random([4,3])
#v = np.random.random([4,])

#tt = torch.tensor(t)
#vv = torch.tensor(v)
#print tt
#torch.cat((tt, vv), 1)
#ind = torch.randperm(4)
#print ind
#print ind.shape
#print tt[ind]
#X = tt[:, :-1]
#Y = tt[:, -1]
#print (X)
#print (Y)
