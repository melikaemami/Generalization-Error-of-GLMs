# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:52:26 2020

@author: sdran
"""

import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
import tensorflow as tf
import tensorflow.keras.backend as K
tfk = tf.keras
tfkl = tf.keras.layers


class TestModel(object):
    def __init__(self, nw, ny, nout, test_type='linear', mod_pkg='sklearn',\
                 lam=1.,nepochs=None, batch_size=100, lr=None,\
                 verbose=0):
        """
        TestModel:  A model to test the SE against
        
        There are three basic models for the test
        * 'linear':  Linear gaussian model
        * 'logistic':  Binary classification with a logistic output
        * 'nonlinear':  Non-linear regression
        
        Parameters
        ----------
        test_type: {'linear', 'logistic', 'nonlinear'}
            Test type.  
        mod_pkg:  {'sklearn', 'tensorflow'}
            Model type
        lam: float
            L2-regularization on weights
        nw:  int
            number of features
        ny:  int
            number of measurements
        nout:  int
            number of outputs
        batch_size: int
            batch_size.  Set to 0 for full-batch gradient descent.
            Full batch will be slower, but will approach the true
            minima more exactly
        lr:  int or `None`
            learning rate.  Set to `None` for default lr
        nepochs:  int
            number of epochs.  For full-batch gradient descent,
            this is the number of steps
        verbose:  int
            verbosity level for fit routine
    
        Returns
        -------
        None.

        """
        self.test_type = test_type
        self.mod_pkg = mod_pkg
        self.lam = lam
        self.nw = nw
        self.nout = nout
        self.ny = ny
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.lr = lr
        self.verbose = verbose
        
        # Check arguments
        if not (self.test_type in ['linear', 'logistic', 'nonlinear']):
            raise ValueError('Test type %s unknown' % self.test_type)
        if not (self.mod_pkg in ['sklearn', 'tf']):
            raise ValueError('Model package %s unknown' % self.mod_pkg)      
                            
        # Test if full batch is used
        self.full_batch = (self.batch_size == 0)
        if self.full_batch:
            self.batch_size = ny
                
        # Set the default learning rates
        lr_mini_batch = {'linear': 0.01, 'logistic': 0.003,\
                         'nonlinear': 0.01}
        lr_full_batch = {'linear': 0.1, 'logistic': 0.01,\
                         'nonlinear': 0.1}
        if (self.lr is None):
            if self.full_batch: 
                self.lr = lr_full_batch[self.test_type]
            else:
                self.lr = lr_mini_batch[self.test_type]
                
        # Set default number of epochs
        if self.nepochs is None:
            if self.full_batch:
                self.nepochs = 1000
            else:
                self.nepochs = 200
            
        # Build the model
        if self.mod_pkg == 'sklearn':
            self.build_mod_sklearn()
        else:
            self.build_mod_tf()
            
    # Output functions for the non-linear regression case
    fscale = 3
    def fnl(p):
        """ 
        Output function 
        """
        u = TestModel.fscale*p
        return (np.exp(u)-1)/(1+np.exp(u))
    def fnl_grad(p):
        """ 
        Output function gradient
        """
        u = TestModel.fscale*p
        u = np.minimum(u, 10)
        grad = TestModel.fscale*2*np.exp(u)/((1+np.exp(u))**2)    
        return grad
    def fnl_tf(p):
        """
        Tensorflow implementation of the function
        """
        z = (K.exp(TestModel.fscale*p)-1)/(1+K.exp(TestModel.fscale*p))
        return z
            
                        
        
    def build_mod_sklearn(self):
        """
        Builds the model using the sklearn package
        """
        if self.test_type == 'nonlinear':
            raise ValueError('nonlinear model are not supported in ' +\
                             'the skelarn packge.  Use mod_pkg=tf')
                
        # Fit the data with sklearn's Ridge or LogisticRegression method
        if self.test_type == 'linear':                
            self.mod = Ridge(alpha=self.lam,fit_intercept=False)
        elif self.test_type == 'logistic':
            self.mod = LogisticRegression(fit_intercept=False, C=1/self.lam)
            
    def build_mod_tf(self):
        """
        Builds the model using Tensorflow
        """
        
        # Set L2 regression level.  TF will minimize,
        #
        #    ||y-p||**2 + alpha*||w||^2/nw 
        #
        # alpha = lam*nout/nw
        K.clear_session()
        alpha = self.lam/self.nout/self.ny
        if self.test_type == 'logistic':
            alpha /= 2
                
        self.mod = tfk.models.Sequential()
        self.mod.add(tfkl.Dense(self.nout,input_shape=(self.nw,),\
            name='linear', use_bias=False,\
            kernel_regularizer=tfk.regularizers.l2(alpha)) )
        if self.test_type == 'logistic':
            self.mod.add(tfkl.Activation('sigmoid'))
        elif self.test_type == 'nonlinear':
            self.mod.add(tfkl.Lambda(TestModel.fnl_tf))
            
    def fit(self,Xtr,ytr):
        """
        Fits data 
        
        Parameters
        ----------
        Xtr, ytr:  ndarrays
            Training data
        """
        # Check if output shape matches
        if (ytr.shape != (self.ny, self.nout)):
            raise ValueError('Expecting shape %s.  Received %s'\
                             % (str((self.ny,self.nout)),str(ytr.shape)))
        
        # The logistic sklearn method can only do one output
        if (self.test_type == 'logistic') and (self.mod_pkg=='sklearn'):
            if (self.nout != 1):
                raise ValueError('Logistic model with sklearn only takes ' +\
                                 'single ouptut')
            ytr = np.squeeze(ytr)    
        
        if self.mod_pkg == 'sklearn':
            # Fit the data using skelarn
            self.mod.fit(Xtr,ytr)
        
            # Store the coefficients
            self.what = self.mod.coef_.T
        else:
            # Fit the model using tensorflow
            if self.full_batch:
                opt = tfk.optimizers.Adam(lr=self.lr)
            else:
                opt = tfk.optimizers.SGD(lr=self.lr)
                
            if self.test_type == 'logistic':
                self.mod.compile(\
                    optimizer=opt, loss='binary_crossentropy',\
                    metrics=['accuracy'])
                    
            else:
                self.mod.compile(\
                    optimizer=opt, loss='mse',\
                    metrics=['mse'])
            self.hist = self.mod.fit(\
                Xtr,ytr,epochs=self.nepochs,\
                batch_size=self.batch_size,\
                verbose=self.verbose)
            self.what = self.mod.get_weights()[0]
            
    def test_grad(self,Xtr,ytr):
        """
        Tests the gradient 

        Parameters
        ----------
        Xtr : ndarray
            matrix of features
        ytr : ndarray
            matrix of responses

        Returns
        -------
        None.

        """
        ptr = Xtr.dot(self.what)
        if self.test_type == 'linear':
            e = ptr-ytr
        elif self.test_type == 'logistic':
            # The binary cross entropy loss is:
            #    log(1 + exp(-o)) - y*p
            # So, the gradient is:
            #    grad = 1/(1+exp(-p)) - y 
            e = 1/(1+np.exp(-ptr)) - ytr
        elif self.test_type == 'nonlinear':            
            e = (TestModel.fnl(ptr)-ytr)*TestModel.fnl_grad(ptr)
            
        grad_out = Xtr.T.dot(e)
        grad_in = self.lam*self.what
        grad = grad_in + grad_out
        return grad, grad_in, grad_out
        print( 'Final MSE grad = %12.4e' % np.mean(grad**2))
        
        
    def predict(self,Xts):
        """
        Predicts values on test data
        
        Parameters
        ----------
        Xts:  ndarray
            Test data
            
        Returns:
        --------
        yts_hat:  ndarray
            Predicted output
        """
        yts_hat = self.mod.predict(Xts)
        if self.test_type == 'logistic':
            nts = yts_hat.shape[0]
            yts_hat = np.reshape(yts_hat, (nts,1))
        return yts_hat
        
        
            
                  
        



        ""