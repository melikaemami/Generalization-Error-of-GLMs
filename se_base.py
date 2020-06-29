# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:28:58 2020

@author: sdran
"""
import numpy as np

class SELayer(object):
    """
    Static variables
    """
    glob_layer_num = 0
    
    
    def __init__(self,shape,layer_type='input',\
                 dtype=None,nsamp=None, name=None,\
                 layer_class_name='Layer', damp_neg=0.):
        """
        Base class for evaluating one layer of the SE.
        
        In a sequential model, layers are either `input`, 
        `middle` or `output`.  A `middle` layer describes the relations
        between two variables:  an input `p` and output `z`.  The
        description is based on two functions:  
            
            z0 = sample(p0)
            phat, zhat, alpha0n, alpha1p = est(rp,rz,gam0p, gam1n)
        
        The function `sample` produces samples of the output `z0` from
        the input `p0`, representing the *true* samples through the layer.
        
        The function `est()` produces estimates of `p0` and `z0`
        from Gaussian corrupted versions of these vectors.
        
        The input layer methods are identical except that `p0` is removed:        
        
            z0 = sample(),   
            zhat, alpha1p = est(rz,gam1n)
            
        The output layer has the `z0` used in the estimator:
            
            z0 = sample(p0)
            phat, alpha0n = est(rp,rz, z0, gam0p, gam1n)
            

                    
        Parameters
        ----------
        layer_type : 'input', 'output', 'middle'
            type of layer in a sequential model.  
        shape : `array` or list of `array`
            For `input` layer:  `shape = z0.shape`.
            Otherwise, `shape = [p0.shape, z0.shape]`
        dtype : type or list of types 
        nsamp : int
            Number of samples used for Monte Carlo simulation   
            A value `None` indicates to use one sample without
            increasing the dimension.
        damp_neg : scalar
            Damping in the reverse direction.  Value is `[0,1]`
            with 0 indicatng no damping and 1 indicating high damping
   
        """        
        if layer_type not in ['input', 'output', 'middle']:
            raise ValueError('Unknown layer type %s' % layer_type)
        self.layer_type = layer_type

        self.shape = shape    
        self.nsamp = nsamp
        if (dtype == None):
            if (self.layer_type == 'input'):
                dtype = np.float
            else:
                dtype = [np.float, np.float]
        self.dtype = dtype
        
        # Set layer name
        if name is None:
            self.name = ('Layer %d' % SELayer.glob_layer_num )
        else:
            self.name = name
        SELayer.glob_layer_num += 1
        self.layer_class_name = layer_class_name
        
        # Other parameters
        self.pp_var_min = 1e-10
        self.damp_neg = damp_neg
        self.gam0_neg_last = None
        self.tau_last = None
           
        
    def sample_shape(shape,nsamp=None):
        """
        Computes the shape of multiple samples

        Parameters
        ----------
        shape : array of ints
            base shape
        nsamp : int
            number of samples.  `nsamp is None` indicates taking one
            sample without adding dimensions

        Returns
        -------
        shape_samp :
            output sample shape

        """
        if nsamp == None:
            sample_shape = shape
        else:
            sample_shape = shape + (nsamp,)
        return sample_shape
        
    def sample(self,p0=None,nsamp=None):
        """
        Conditionally samples `z0` given `p0`.
            
        Parameters
        ----------
        p0:  samples of the input
            If `layer_type=='input'` layer, set `p0 = None`.
        nsamp : number of samples
            If `nsamp is None`, then one sample is produced
            with shape `z0.shape`.  Otherwise, `nsamp`
            samples are produced with an additional dimension
            added.
    
        Raises
        ------
        NotImplementedError
            If not implemented
    
        Returns
        -------
        z0 : Samples of the output
    
        """        
        raise NotImplementedError('Need to implement sample function')
        
    def est(self,r,gam):
        """
        Estimator function for the layer.
        
        For a middle layer, this generally implements a proximal
        operation of the form,
        
           est([rp,rz], [gamp,gamz]) := [phat,zhat]
           
        which are the miminizers of:
            
                F(p,z) + ||p-rp||^2*(gamp/2) + ||z-rz||^2*(gamz/2)
           
        
        The function also computes derivatives, `alpha=[alphap,alphaz]
            
            alphap := dzhat / drz
            alphaz := dphat / drp

        Parameters
        ----------
        r : array or list of arrays
            For an input layer, `r=rz`.  Otherwise, `r=[rp,rz]`
        gam : scalar or list of scalar
            For an input layer, `gam=gamz`.  For an output layer,
            `gam=gamp`.  For a middle layer, `gam=[gamp,gamz]`.

        Returns
        -------
        xhat:  array or list of arrays
            For an input layer, `xhat=zhat`.  For a middle layer, 
            `xhat=[phat,zhat]`.  For an output layer, `xhat=phat`.
        alpha:  scalar or list of scalars
            For an input layer, `alphaz`.  For a middle layer, 
            `[alphap,alphaz]`.  For an output layer, `alphaz`.                

        """
        raise NotImplementedError('Need to implement estimation function')
        
    def compSETrue(self,taup=None):
        """
        Computes the true signal variance,
        
            tauz = E(Z0**2)  for Z0=sample(P0),  P0 = N(0,taup)
            
        For an initial layer, `taup` is not specified.
        For an output layer, this does not need to be implemented.
        
        Parameters
        ----------
        taup:  True input variance (middle or end layer only)
        
        Returns
        -------
        tauz:  True output second moment 
        """
        
        # Use Monte-Carlo simulation for the default implementation
        return self.compSETrueSim(taup)
    
        
    def compSETrueSim(self,taup=None):
        """
        Monte Carlo based implementation of `compSETrue()`.
        
        Parameters
        ----------
        taup:  True input variance (middle or end layer only)
        
        Returns
        -------
        tauz:  True output second moment 
        """
        
        if self.layer_type == 'input':
            # Create unit Gaussians for the sampling
            zshape = SELayer.sample_shape(self.shape,self.nsamp)
            self.wq = np.random.normal(0,1,zshape)                        

            # Sample the output
            self.ztrue = self.sample(nsamp=self.nsamp)
            
            # Compute the sample mean
            tauz = np.mean(np.abs(self.ztrue)**2)
            return tauz
            
            
        else:
            # Create unit Gaussians for the sampling
            pshape = SELayer.sample_shape(self.shape[0],self.nsamp)
            zshape = SELayer.sample_shape(self.shape[1],self.nsamp)
            self.wp = np.random.normal(0,1,pshape)                        
            self.wq = np.random.normal(0,1,zshape)                        

            # Generate random Gaussian input
            pshape = SELayer.sample_shape(self.shape[0], self.nsamp)
            self.ptrue = np.random.normal(0,np.sqrt(taup),pshape)
            
            # Sample from the ouptut
            self.ztrue = self.sample(self.ptrue,nsamp=self.nsamp)
            
            # Compute the sample mean
            tauz = np.mean(np.abs(self.ztrue)**2)
            return tauz
             
            
            
    def compSEEst(self,dir='fwd',K0=None,tau1=None,gam0_pos=None,gam1_neg=None):
        """
        Compute quantities for the SE estimation
        
        Given the parameters, define the random variables:
        
            (P0,Pp) = N(0,K0), Qn = N(0,tau1)
            Z0 = sample(P0)
            Rp = P0 + Pp
            Rz = Z0 + Qn 
            Zhat, Phat, alphap, alphaz = est(Rp, Rz, gam0_pos, gam1_neg)
            Qp = (Zhat-Z0-alphaz*Qn)/(1-alphaz)
            Pn = (Phat-P0-alphap*Pp)/(1-alphap)         
            
            
        Then, if `dir=='fwd', the method returns
            K1 = cov(Z0,Qp)
            gam1_pos = gam1_neg(1/alphap - 1)
            zerr = E((Z0-Zhat)**2)
        If `dir == 'rev'`, the method returns
            tau0 = E(Pn**2)
            gam0_neg = gam0_pos(1/alphaz - 1)
            perr = E((P0-Phat)**2)
            
        For layer_type==`input`, `K0` and `gam0_pos=None`.  
        For layer_type==`output`, `tau1` and `gam1_neg=None`.  
            

        Parameters
        ----------
        K0:  (2,2) array
            Input covariance on `(P0,Pp)`
        tau1:  scalar
            Input variance on `Qn`
        gam0_pos, gam1_neg:  scalar
            Input precisions

        Returns
        -------
        K1:  (2,2) 'array':
            Forward covariance (returned if dir=='fwd')
        gam1_pos:  scalar
            Forward precision for `z`  (returned if dir=='fwd')
        zerr:  scalar
            Output estimation error (returned if dir=='fwd')
        tau0:  scalar
            Reverse variance (returned if dir=='rev')
        gam0_neg:  scalar
            Reverse precision for `z`  (returned if dir=='rev')
        perr:  scalar
            Input estimation error (returned if dir=='rev')      
        """
        
        # Default implementation uses Monte-Carlo
        return self.compSEEstSim(dir,K0,tau1,gam0_pos,gam1_neg)
        
    def compSEEstSim(self,dir='fwd',K0=None,tau1=None,gam0_pos=None,gam1_neg=None):
        """
        Monte Carlo simulation implementation of compSEEst 
            

        Parameters
        ----------
        See `compSEEst()`.
        
        Returns
        -------
        K1:  (2,2) 'array':
            Forward covariance (returned if dir=='fwd')
        gam1_pos:  scalar
            Forward precision for `z`  (returned if dir=='fwd')
        zerr:  scalar
            Output estimation error (returned if dir=='fwd')
        tau0:  scalar
            Reverse variance (returned if dir=='rev')
        gam0_neg:  scalar
            Reverse precision for `p`  (returned if dir=='rev')
        perr:  scalar
            Input estimation error (returned if dir=='rev')      
        """
        
        
        # Generate the inputs to the estimator
        if self.layer_type != 'input':
            pp_gain = K0[1,0]/K0[0,0]        
            pp_var = K0[1,1] - np.abs(K0[0,1])**2/K0[0,0]
            pp_var = np.maximum(pp_var, self.pp_var_min)
            pp = self.ptrue*pp_gain + self.wp*np.sqrt(pp_var)
            rp = self.ptrue + pp
        if self.layer_type != 'output':
            qn = np.sqrt(tau1)*self.wq
            rz = self.ztrue + qn
            
        # Call the estimator
        if self.layer_type == 'input':
            zhat, alphaz = self.est(rz,gam1_neg, self.nsamp)
            phat = None
        elif self.layer_type == 'middle':
            xhat, alpha = self.est([rp, rz], [gam0_pos, gam1_neg], self.nsamp)
            phat, zhat = xhat
            alphap, alphaz = alpha
        elif self.layer_type == 'output':
            phat, alphap = self.est(rp,gam0_pos, self.nsamp)
            zhat = None
            
        # Save the values
        self.zhat = zhat
        self.phat = phat
            
            
        # Output the variables
        if dir == 'fwd':
            
            gam1_pos = gam1_neg*(1/alphaz - 1)
            qp = (zhat - self.ztrue - alphaz*qn)/(1-alphaz)
            
            
            K00 = np.mean(np.abs(self.ztrue)**2)
            K01 = np.mean(np.conj(self.ztrue)*qp)
            K11 = np.mean(np.abs(qp)**2)
            K1 = np.array([[K00, np.conj(K01)], [K01, K11]])
            
            zerr = np.mean(np.abs(self.ztrue-zhat)**2)
                            
            return K1, gam1_pos, zerr
        
        else:
            pn = (phat - self.ptrue - alphap*pp)/(1-alphap)
            tau0 = np.mean(np.abs(pn)**2)
            gam0_neg = gam0_pos*(1/alphap - 1)
            
            # Damping
            if (self.damp_neg > 0) and not (self.gam0_neg_last is None):
                gam0_neg = (1-self.damp_neg)*gam0_neg +\
                    self.damp_neg*self.gam0_neg_last
                tau0 = (1-self.damp_neg)*tau0 +\
                    self.damp_neg*self.tau0_last
            
            self.gam0_neg_last = gam0_neg
            self.tau0_last = tau0                
            
            perr = np.mean((np.abs(self.ptrue-phat))**2)
            
            return tau0, gam0_neg, perr
                
            
        
            
            
            
            
        
   