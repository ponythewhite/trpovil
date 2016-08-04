"""
The MIT License (MIT)
Copyright (c) 2016 Jacek Dabrowski

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""    


from GenerativeModel import *
from RecognitionModel import *
sys.path.append('lib/') 
from MinibatchIterator import *
from lasagne.nonlinearities import leaky_rectify, softmax, linear, tanh, rectify
from modular_rl import agentzoo

from collections import defaultdict
from collections import OrderedDict

from sklearn.cluster import KMeans

import gym

from gym.spaces import Box, Discrete

import numpy as np

import time

import scipy

class BuildModelTRPO():
    def __init__(self,
                gen_params, # dictionary of generative model parameters
                GEN_MODEL,  # class that inherits from GenerativeModel
                xDim=2, # dimensionality of latent state
                yDim=2, # dimensionality of observations
                trpo_batch = 512, # batch size for TRPO agent
                trpo_iter_start = 32, # starting number of TRPO "warmup" iterations (before SGD kicks in)
                sgd_batch = 32, # batch size of SGD
                max_epochs=1, # max number of total epochs
                ):
        
        self.trpo_batch = trpo_batch
        self.trpo_iter_start = trpo_iter_start
        self.sgd_batch = sgd_batch
        
        self.max_epochs = max_epochs
        # instantiate rng's -- dataset iterator does not obey these
        self.srng = RandomStreams(seed=234)
        self.nrng = np.random.RandomState(124)
        
        #---------------------------------------------------------
        ## actual model parameters
        self.X, self.Y = T.matrices('X','Y')   # symbolic variables for the data
        
        self.hsamp = T.lmatrix('hsamp')
        self.hRaw = T.matrix('hRaw')
        self.hLogDensity = T.matrix('hLogDensity')

        self.xDim   = xDim
        self.yDim   = yDim
        
        # instantiate "recognition model" (TRPO agent)
        self.agent = agentzoo.TrpoAgent(Box(-np.inf, np.inf, yDim), Discrete(xDim), {
            "timestep_limit": 1,
            "hid_sizes": [64,32],
            "timesteps_per_batch": trpo_batch,
            "activation": "tanh",
            "gamma": 1.0,
            "lam": 1.0,
            "max_kl": 0.01,
            "cg_damping": 0.005,
            "filter": False
        })      
        
        # instantiate our prior
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim, srng=self.srng, nrng = self.nrng)
        
    def getParams(self):
        ''' 
        Return Generative and Recognition Model parameters that are currently being trained.
        '''
        params = []        
        params = params + self.mprior.getParams()            
        return params        
        
    def get_cost(self,Y,hsamp, bSize):
        '''
        NOTE: Y and hsamp are both assumed to be symbolic Theano variables. 
        
        '''
        # evaluate the "recognition model" / agent density Q_\phi(h_i | y_i)
        q_hgy = T.reshape(self.hLogDensity, (bSize,))

        # evaluate the generative model density P_\theta(y_i , h_i)
        p_yh =  T.reshape(self.mprior.evaluateLogDensity(hsamp,Y), (bSize,)) 
        
        L = p_yh.mean() - q_hgy.mean()
        l = p_yh - q_hgy

        return [L,l,p_yh]
        
    def get_gradients(self,p_yh,l):  
        def comp_param_grad(ii, pyh, l):
            dpyh = T.grad(cost=pyh[ii], wrt = self.mprior.getParams())

            output = [t for t in dpyh]
            return output

        grads,_ = theano.map(comp_param_grad, sequences = [T.arange(self.Y.shape[0])], non_sequences = [p_yh, l] )
        
        return [g.mean(axis=0, dtype=theano.config.floatX) for g in grads]

    def update_params(self, grads, L, l):
        batch_y = T.matrix('batch_y')
        h = T.lmatrix('h')
        hLogD = T.matrix('hLogD')
        lr = T.scalar('lr')
        
        # SGD updates
        #updates = [(p, p + lr*g) for (p,g) in zip(self.getParams(), grads)]
        
        # Adam updates        
        # We negate gradients because we formulate in terms of maximization.
        updates = lasagne.updates.momentum([-g for g in grads], self.getParams(), lr)
#        updates = lasagne.updates.nesterov_momentum([-g for g in grads], self.getParams(), lr)
#        updates = lasagne.updates.adam([-g for g in grads], self.getParams(), lr) 
       
        perform_updates_params = theano.function(
                 outputs=[L, l],
                 inputs=[ theano.In(batch_y), theano.In(h), theano.In(hLogD), theano.In(lr)],
                 updates=updates,
                 givens={
                     self.Y: batch_y,
                     self.hsamp: h,
                     self.hLogDensity: hLogD
                 }
             )
        
        return perform_updates_params
    
    def fit(self, y_train, learning_rate = 3e-4):
        
        # used just to get number of batches. Actual iterators are separate for TRPO and SGD
        sgd_iterator = DatasetMiniBatchIterator(y_train, self.sgd_batch)
    
        L_trpo,l_trpo,p_yh_trpo = self.get_cost(self.Y, self.hsamp, self.trpo_batch) # cost for TRPO (different batch size)
        L_sgd,l_sgd,p_yh_sgd = self.get_cost(self.Y, self.hsamp, self.sgd_batch) # cost for SGD (different batch size)
        
        batch_y = T.matrix('batch_y')
        h = T.lmatrix('h')
        hLogD = T.matrix('hLogD')        
        
        # used for TRPO
        get_reward = theano.function(
                 outputs=[L_trpo, l_trpo],
                 inputs=[ theano.In(batch_y), theano.In(h), theano.In(hLogD)],
                 givens={
                     self.Y: batch_y,
                     self.hsamp: h,
                     self.hLogDensity: hLogD
                 }
             )
             
        grads_sgd = self.get_gradients(p_yh_sgd, l_sgd)
    
        # used for SGD
        param_updater = self.update_params(grads_sgd, L_sgd, l_sgd)

        avg_costs = []
        
        epoch = 0

        # used for controlling the warmup phase of TRPO        
        total_sgd_batch_counter = 0
        
        while epoch < self.max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ self.max_epochs))
            sys.stdout.flush()
            sgd_batch_counter = 0 # used for displaying updates
            
            # In one epoch, we want SGD to cover the whole dataset
            for it_no in xrange(sgd_iterator.n_batches):
                
                # Before an SGD iteration, we run a TRPO update (multiple during warmup phase)
                for trpo_i in xrange(np.maximum(1, self.trpo_iter_start / 4 ** total_sgd_batch_counter)):
                    trpo_iterator = DatasetMiniBatchIterator(y_train, self.trpo_batch)
                    y = trpo_iterator.first()
                    
                    # AGENT ACT                
                    hsamp_np, h_np, hLogDensity_np = self.agent.batch_act_with_raw_p_and_log_density(y)
                    hsamp_np = map(lambda x: x.astype(np.int32), hsamp_np)
                    h_np = map(lambda x: x.astype(np.float32), h_np)
                    hLogDensity_np =  map(lambda x: x.astype(np.float32), hLogDensity_np)

                    avg_cost, costs = get_reward(y, hsamp_np, hLogDensity_np)
                    print '(trpo L): (%f)\n' % (avg_cost)
                    tstart = time.time()                
                    
                    gamma = 1.0
                    lam = 1.0
                    
                    def pathlength(path):
                        return len(path["action"])                
                    
                    def discount(x, gamma):
                        assert x.ndim >= 1
                        return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    
                    def compute_advantage(vf, paths):
                        # Compute return, baseline, advantage
                        for path in paths:
                            path["return"] = discount(path["reward"], gamma)
                            b = path["baseline"] = vf.predict(path)
                            b1 = np.append(b, 0 if path["terminated"] else b[-1])
                            deltas = path["reward"] + gamma*b1[1:] - b1[:-1] 
                            path["advantage"] = discount(deltas, gamma * lam)
                        alladv = np.concatenate([path["advantage"] for path in paths])    
                        # Standardize advantage
                        std = alladv.std()
                        mean = alladv.mean()
                        for path in paths:
                            path["advantage"] = (path["advantage"] - mean) / std
                    
                    
                    
                    
                    
                    paths = []
                    
                    for i in range(self.trpo_batch):
                    # SUPPLY REWARD TO AGENT
                        data = defaultdict(list)
                        data["observation"].append(y[i])
                        data["action"].append(hsamp_np[i])
                        data["reward"].append(costs[i])
                        data["prob"].append(h_np[i])
                        data = {k:np.array(v) for (k,v) in data.iteritems()}
                        data["terminated"] = True
                        paths.append(data)

                    compute_advantage(self.agent.baseline, paths)
                    
                    def add_episode_stats(stats, paths):
                        reward_key = "reward_raw" if "reward_raw" in paths[0] else "reward"
                        episoderewards = np.array([path[reward_key].sum() for path in paths])
                        pathlengths = np.array([pathlength(path) for path in paths])
                     
                        stats["EpisodeRewards"] = episoderewards
                        stats["EpisodeLengths"] = pathlengths
                        stats["NumEpBatch"] = len(episoderewards)
                        stats["EpRewMean"] = episoderewards.mean()
                        stats["EpRewSEM"] = episoderewards.std()/np.sqrt(len(paths))
                        stats["EpRewMax"] = episoderewards.max()
                        stats["EpLenMean"] = pathlengths.mean()
                        stats["EpLenMax"] = pathlengths.max()
                        stats["RewPerStep"] = episoderewards.sum()/pathlengths.sum()
                        
                    
                    def add_prefixed_stats(stats, prefix, d):
                        for (k,v) in d.iteritems():
                            stats[prefix+"_"+k] = v

                    vf_stats = self.agent.baseline.fit(paths)
                    pol_stats = self.agent.updater(paths)
                    
                    stats = OrderedDict()
                    add_episode_stats(stats, paths)
                    add_prefixed_stats(stats, "vf", vf_stats)
                    add_prefixed_stats(stats, "pol", pol_stats)                
                    stats["TimeElapsed"] = time.time() - tstart        
                
                # Finally we run an SGD update (just one works fine)
                for sgd_i in xrange(1):
                    sgd_iterator2 = DatasetMiniBatchIterator(y_train, self.sgd_batch)
                    y_sgd = sgd_iterator2.first()
                    
                    hsamp_np, h_np, hLogDensity_np = self.agent.batch_act_with_raw_p_and_log_density(y_sgd)
                    hsamp_np = map(lambda x: x.astype(np.int32), hsamp_np)
                    h_np = map(lambda x: x.astype(np.float32), h_np)
                    hLogDensity_np =  map(lambda x: x.astype(np.float32), hLogDensity_np)
                    
                    avg_cost, costs = param_updater(y_sgd, hsamp_np, hLogDensity_np, learning_rate * (0.1 ** epoch))                
                
                    if np.mod(sgd_batch_counter, 1) == 0:
                        print '(sgd L): (%f)\n' % (avg_cost)
                    
                avg_costs.append(avg_cost)
                sgd_batch_counter += 1
                total_sgd_batch_counter += 1
                
            epoch += 1
        return avg_costs