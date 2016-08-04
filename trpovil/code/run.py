# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:31:29 2016

@author: dabrowski
"""


import theano
import theano.tensor as T
import theano.tensor.nlinalg as Tla
import lasagne       # the library we're using for NN's
# import the nonlinearities we might use 
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
from numpy.random import *
import matplotlib
from matplotlib import pyplot as plt

import cPickle
import sys
import time
# import kmeans clustering algorithm from scikit-learn
from sklearn.cluster import KMeans 

sys.path.append('lib/') 
from GenerativeModel import *       # Class file for generative models. 
from RecognitionModel import *      # Class file for recognition models
from NVIL import BuildModelNVIL                  # The meat of the algorithm - define the cost function and initialize Gen/Rec model
from TRPOVIL import BuildModelTRPO

# import our covariance-plotting software
from plot_cov import *

print "Imports done"

theano.config.optimizer = 'fast_compile' 

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)

xDim = 5 # number of latent classes
yDim = 2 # dimensionality of Gaussian observations
_N = 2048 * 8 # number of datapoints to generate
gmm = MixtureOfGaussians(dict([]), xDim, yDim)  # instantiate our 'true' generative model
[xsamp, ysamp] = gmm.sampleXY(_N)

# center our simluated data around the mean
ysamp_mean = ysamp.mean(axis=0, dtype=theano.config.floatX)
ytrain = ysamp - ysamp_mean

# Initialize generative model at the k-means solution, just 1 iteration, best of 10
km = KMeans(n_clusters=xDim, n_init=10, max_iter=1)
kmpred = km.fit_predict(ytrain)

km_mu = np.zeros([xDim, yDim])
km_chol = np.zeros([xDim, yDim, yDim])
for cl in np.unique(kmpred):
    km_mu[cl] = ytrain[kmpred == cl].mean(axis=0)
    km_chol[cl] = np.linalg.cholesky(np.cov(ytrain[kmpred == cl].T))


print "KMeans done"

model = BuildModelTRPO(dict([]), MixtureOfGaussians, xDim, yDim, trpo_batch = 1024, trpo_iter_start = 50, sgd_batch = 64, max_epochs=2)

# Initialize with a "broken" solution

model.mprior.mu.set_value((0.1 + km_mu *1.2).astype(theano.config.floatX))
model.mprior.RChol.set_value(km_chol.astype(theano.config.floatX))
km_pi = np.histogram(kmpred,bins=xDim)[0]/(1.0*kmpred.shape[0])
model.mprior.pi_un.set_value(km_pi.astype(theano.config.floatX))

# Initialize with *true* means and covariances
#model.mprior.mu.set_value(gmm.mu.get_value()-ysamp_mean)
#model.mprior.RChol.set_value(gmm.RChol.get_value())
#model.mprior.pi_un.set_value(gmm.pi_un.get_value())

clr = ['b', 'r', 'c','g','m','o']

plt.figure()
plt.subplot(121)
plt.plot(ysamp[:,0], ysamp[:,1],'k.', alpha=.1)
plt.hold('on')
for ii in xrange(xDim):
    Rc= gmm.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), gmm.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('True Distribution')
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.subplot(122)
plt.hold('on')
plt.plot(ytrain[:,0], ytrain[:,1],'k.', alpha=.1)
for ii in xrange(xDim):
    Rc= model.mprior.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), model.mprior.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('Initialization Distributions')    
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.savefig("init.png")


print ysamp.shape

# Fit the model
tstart = time.time()
costs = model.fit(ytrain, learning_rate = 3e-4)
tend = time.time()
t1 = tend - tstart
print "TOOK TOTAL:", t1




clr = ['b', 'r', 'c','g','m','o']

plt.figure()
plt.subplot(121)
plt.plot(ysamp[:,0], ysamp[:,1],'k.', alpha=.1)
plt.hold('on')
for ii in xrange(xDim):
    Rc= gmm.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), gmm.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('True Distribution')
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.subplot(122)
plt.hold('on')
plt.plot(ytrain[:,0], ytrain[:,1],'k.', alpha=.1)
for ii in xrange(xDim):
    Rc= model.mprior.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), model.mprior.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('TRPO Learned Distributions')    
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.savefig("learned_trpo.png")
plt.show()


#
#xlbl = xsamp.nonzero()[1]
#model.agent.set_stochastic(False) # TURN OFF STOCHASTICITY
#hsamp_np, _, _ = model.agent.batch_act_with_raw_p_and_log_density(ytrain)
#hsamp_np = np.array(map(lambda x: x.astype(np.int32), hsamp_np))
#learned_lbl = hsamp_np.argmax(axis=1)
#
#clr = ['b', 'r', 'c','g','m','o']
#
#plt.figure()
#for ii in np.random.permutation(xrange(500)):
#    plt.subplot(121)
#    plt.hold('on')
#    plt.plot(ysamp[ii,0] ,ysamp[ii,1],'.', color = clr[xlbl[ii]%5])
#    plt.subplot(122)
#    plt.hold('on')
#    plt.plot(ysamp[ii,0] ,ysamp[ii,1],'.', color = clr[learned_lbl[ii]%5])
#    
#plt.subplot(121)
#plt.title('True Label')
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#plt.subplot(122)
#plt.title('Inferred Label')
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#    
#    
#plt.savefig("labels_trpo.png")    
#plt.show()
#
#
#n = 25
#
#x = np.linspace(-3, 3, n)
#y = np.linspace(-3, 3, n)
#xv, yv = np.meshgrid(x, y)
#grid= np.vstack([xv.flatten(), yv.flatten()]).T
#
#gridlabel, _, _ = model.agent.batch_act_with_raw_p_and_log_density(grid)
#gridlabel = np.array(map(lambda x: x.astype(np.int32), gridlabel))
#gridlabel = gridlabel.argmax(axis=1)
#
#plt.figure()
#plt.hold('on')
#for ii in xrange(n*n):
#    plt.plot(grid[ii,0] ,grid[ii,1],'.', color = clr[gridlabel[ii]%5])
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#plt.title('Highest-Probability Label Over Sampled Grid')
#plt.savefig("grid_trpo.png")    
#plt.show()




# ----------------------------------

# NVIL FOR COMPARISON


# construct a BuildModel object that represents the method
opt_params = dict({'c0': -0.0, 'v0': 1.0, 'alpha': 0.9})

rec_is_training = theano.shared(value = 1) 
rec_nn = lasagne.layers.InputLayer((None, yDim))
rec_nn = lasagne.layers.DenseLayer(rec_nn, 100, nonlinearity=leaky_rectify, W=lasagne.init.Orthogonal())
rec_nn = lasagne.layers.DenseLayer(rec_nn, xDim, nonlinearity=softmax, W=lasagne.init.Orthogonal(), b=-5*np.ones(xDim, dtype=theano.config.floatX))
NN_Params = dict([('network', rec_nn)])
recDict = dict([('NN_Params'     , NN_Params)
                ])


model = BuildModelNVIL(opt_params, dict([]), MixtureOfGaussians, recDict, GMMRecognition, xDim, yDim, nCUnits = 100)

# Initialize with a "broken" solution
model.mprior.mu.set_value((0.1 + km_mu *1.2).astype(theano.config.floatX))
model.mprior.RChol.set_value(km_chol.astype(theano.config.floatX))
km_pi = np.histogram(kmpred,bins=xDim)[0]/(1.0*kmpred.shape[0])
model.mprior.pi_un.set_value(km_pi.astype(theano.config.floatX))

# Initialize with *true* means and covariances
#model.mprior.mu.set_value(gmm.mu.get_value()-ysamp_mean)
#model.mprior.RChol.set_value(gmm.RChol.get_value())
#model.mprior.pi_un.set_value(gmm.pi_un.get_value())

print ysamp.shape

# Fit the model
tstart = time.time()
costs = model.fit(ytrain, batch_size = 16, max_epochs=5, learning_rate = 3e-4)
tend = time.time()
t2 = tend - tstart
print "TOOK TOTAL:", t2


clr = ['b', 'r', 'c','g','m','o']

plt.figure()
plt.subplot(121)
plt.plot(ysamp[:,0], ysamp[:,1],'k.', alpha=.1)
plt.hold('on')
for ii in xrange(xDim):
    Rc= gmm.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), gmm.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('True Distribution')
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.subplot(122)
plt.hold('on')
plt.plot(ytrain[:,0], ytrain[:,1],'k.', alpha=.1)
for ii in xrange(xDim):
    Rc= model.mprior.RChol[ii].eval()
    plot_cov_ellipse(Rc.dot(Rc.T), model.mprior.mu[ii].eval(), nstd=2, color=clr[ii%5], alpha=.3)
    
plt.title('NVIL Learned Distributions')    
plt.ylabel(r'$x_0$')
plt.xlabel(r'$x_1$')

plt.savefig("learned_nvil.png")    
plt.show()


#
#xlbl = xsamp.nonzero()[1]
##learned_lbl = model.mrec.h.eval({model.Y:ytrain}).argmax(axis=1)
##learned_lbl = model.mrec.getSample(ytrain).argmax(axis=1)
#learned_lbl = model.mrec.h.argmax(axis=1).eval({model.Y:ytrain})
#
#clr = ['b', 'r', 'c','g','m','o']
#
#plt.figure()
#for ii in np.random.permutation(xrange(500)):
#    plt.subplot(121)
#    plt.hold('on')
#    plt.plot(ysamp[ii,0] ,ysamp[ii,1],'.', color = clr[xlbl[ii]%5])
#    plt.subplot(122)
#    plt.hold('on')
#    plt.plot(ysamp[ii,0] ,ysamp[ii,1],'.', color = clr[learned_lbl[ii]%5])
#    
#plt.subplot(121)
#plt.title('True Label')
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#plt.subplot(122)
#plt.title('Inferred Label')
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#
#plt.savefig("labels_sgd.png")    
#plt.show()
#
#
#n = 25
#
#x = np.linspace(-3, 3, n)
#y = np.linspace(-3, 3, n)
#xv, yv = np.meshgrid(x, y)
#grid= np.vstack([xv.flatten(), yv.flatten()]).T
#
#gridlabel = model.mrec.getSample(grid.astype(theano.config.floatX)).argmax(axis=1)
#
#plt.figure()
#plt.hold('on')
#for ii in xrange(n*n):
#    plt.plot(grid[ii,0] ,grid[ii,1],'.', color = clr[gridlabel[ii]%5])
#plt.ylabel(r'$x_0$')
#plt.xlabel(r'$x_1$')
#plt.title('Highest-Probability Label Over Sampled Grid')
#plt.savefig("grid_sgd.png")    
#plt.show()

print "t1=", t1, "t2=", t2