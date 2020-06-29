"""
se_layers.py:  Layers for the SE analysis
"""
from se_base import SELayer
import numpy as np


class SEGaussInLayer(SELayer):
    def __init__(self,zshape,zvar=1.0,zmean=0,lam=1.0, **kwargs):
        """
        Gaussian input layer with Ridge regression denoiser
        
        The sampling operator is:
        
            z0  = sample() = N(zmean, zvar)
        
        The estimator is:
        
            est(rz,gamz) := argmin gamz*|z-rz|^2 + lam*|z-zmean|^2

        Parameters
        ----------
        zshape : list
            shape of `z`
        zvar : scalar 
            variance of `z`           
        zmean : scalar
            mean of `z`
        lam : scalar
            regularization constant
        
        """
        self.zvar = zvar
        self.zmean = zmean
        self.lam = lam        
        SELayer.__init__(self,shape=zshape, layer_type='input',\
                         layer_class_name='GaussIn', **kwargs)
        
    def sample(self,ptrue=None,nsamp=None):
        """
        Produces Gaussian samples
        
        Parameters
        ----------
        nsamp:  int
           number of samples
        ptrue:  None
           set to `None` since it is an input layer

        Returns
        -------
        ztrue:  Output samples
        """
        if (ptrue != None):
            raise ValueError('Conditional input must be empty')
        
        ztrue = np.random.normal(self.zmean, np.sqrt(self.zvar),self.shape)
        return ztrue
    
    def est(self,rz,gamz,nsamp=None):
        """
        Performs the minization:
            zhat = argmin_z gamz*|z-rz|^2 + lam*|z-zmean|^2

        Parameters
        ----------
        rz : array
            noisy version of z
        gamz : scalar
            precision of rz

        Returns
        -------
        zhat:  array
            Minimizer
        alphaz:  derivative
        """
        zhat = (gamz*rz + self.lam*self.zmean)/(gamz + self.lam)
        alphaz = gamz/(gamz + self.lam)
        return zhat, alphaz
    
class SEGaussOutLayer(SELayer):
    def __init__(self,pzshape,dvar0=1.0,dvar=None, **kwargs):
        """
        Gaussian output layer with L2-estimation
        
        The sampling operator is:
        
            sample(P0) produces Z0 = P0 + N(0,dvar0)
        
        The estimator is the MAP penalty with 
        
            f(p,z)=|z-p|^2/(2*dvar)

        Parameters
        ----------
        pzshape : array
            shape of `p`= shape of `z`
        dvar0 : scalar 
            true noise variance
        dvar : scalar
            noise variance used by estimator.  If set to `None`,
            then `dvar=dvar0`.
        """
        self.dvar0 = dvar0
        if dvar is None:
            dvar = dvar0
        self.dvar = dvar       
        SELayer.__init__(self,shape=[pzshape,pzshape], layer_type='output',\
                         layer_class_name='GaussOut', **kwargs)
        
    def sample(self,ptrue,nsamp=None):
        """
        Produces Gaussian output samples
        
            z0 = p0 + d,  d = N(0,dvar0)
        
        Parameters
        ----------
        ptrue : array
            Samples of the input, `p0`
        nsamp : `None` or int
            Number of samples

        Returns
        -------
        ztrue:  Output samples
        """
            
        # Compute input and ouptut shape of the samples
        pshape = SELayer.sample_shape(self.shape[0], nsamp)
        
        # Check that the input dimensions and indices match
        if (ptrue.shape != pshape):
            err_str = ('Expected input shape = %s' % str(pshape))
            raise ValueError(err_str)
            
        # Create a noise signal
        dtrue = np.random.normal(0, np.sqrt(self.dvar0),pshape)
        
        # Create a true signal
        ztrue = ptrue + dtrue
        self.y = ztrue
        
        return ztrue   
    def est(self,rp,gamp,nsamp=None):
        """
        Performs the minization:
            phat = argmin_p gamy*|y-p|^2 + gamp*|p-rp|^2

        Parameters
        ----------
        rp : array
        gamp : scalar

        Returns
        -------
        phat:  array
            Minimizer of the objective
        alphap:  scalar
            Derivative
        """
        gamy = 1/self.dvar
        
             
        # Check input shape
        pshape = SELayer.sample_shape(self.shape[0], nsamp)
        if (rp.shape != pshape):
            err_str = ('Expected input shape = %s, received %s'\
                       (str(pshape), str(rp.shape)) )
            raise ValueError(err_str)
            
        # Compute phat
        phat = (gamy*self.y + gamp*rp)/(gamy + gamp)
        
        # Compute derivative
        alphap = gamp/(gamy + gamp)
        
        return phat, alphap
            
    
    
class SELinLayer(SELayer):
    def __init__(self,shape,s,b=0,**kwargs):
        """
        SE layer for a linear transform
        
            z0 = s*p0 + b (*)
            
        Estimation corresponds to MAP penalty:
            
            argmin_{z,p}  |z-rz|^2*gamz + |p-rp|^2*gamp 
            
        subject to constraint z = s*p + b

        Parameters
        ----------
        shape : list of arrays
            `shape[0] = p0.shape`,  `shape[1] = z0.shape`
        s : 1d array
            scaling coefficients
        b : bias terms
            not yet implemented
        """
        self.s = s
        if len(s.shape) > 1:
            raise ValueError('s must be a vector for now')
        self.ns = len(s)
        self.nin = shape[0][0]
        self.nout = shape[1][0]
        
        if (self.ns > self.nin) or (self.ns > self.nout):
            raise ValueError('s must have len <= p0.shape[0] and z0.shape[0]')
        SELayer.__init__(self,shape=shape,layer_type='middle',\
                         layer_class_name='Linear', **kwargs)
   
    def sample(self,ptrue,nsamp=None):
        """
        Samples for the linear operator
        
        Creates samples 
        
            z0 = s*p0
            
        Parameters
        ----------
        ptrue : array
            Samples of the input, `p0`
        nsamp : `None` or int
            Number of samples

        Returns
        -------
        ztrue:  Output samples
        """
        
        # Compute input and ouptut shape of the samples
        pshape = SELayer.sample_shape(self.shape[0], nsamp)
        zshape = SELayer.sample_shape(self.shape[1], nsamp)
        
        # Check that the input dimensions and indices match      
        if (ptrue.shape != pshape):
            err_str = ('Expected input shape = %s' % str(pshape))
            raise ValueError(err_str)
            
        # Reshape s to have same dimensions as sample
        ndims = len(pshape)
        if (ndims > 1):
            smat_shape = (self.ns,) + (ndims-1)*(1,)
            smat = self.s.reshape(smat_shape)
        else:
            smat = self.s
            
        # Create output samples with zero padding
        ztrue = np.zeros(zshape)
        ztrue[:self.ns] = smat*ptrue[:self.ns]
                
        return ztrue
    
    def est(self,r,gam,nsamp=None):
        """
        Performs the minization:
            phat = argmin_p gamz*|s*p-rz|^2 + gamp*|p-rp|^2
            zhat = s*phat

        Parameters
        ----------
        r : list of arrays
            `r = [rp,rz]`
        gam : List of scalars
            `gam = [gamp, gamz]

        Returns
        -------
        xhat:  array
            Minimizer `[phat,zhat]`
        alpha:  array
            Derivatives `[alphap,alphaz]
        """
        # Extract inputs
        rp, rz = r
        gamp, gamz = gam
        
        # Compute input and ouptut shape of the samples
        pshape = SELayer.sample_shape(self.shape[0], nsamp)
        zshape = SELayer.sample_shape(self.shape[1], nsamp)
        
        # Check that the input dimensions and indices match      
        if (rp.shape != pshape):
            err_str = ('Expected rp shape = %s received = %s' 
                       % (str(pshape), str(rp.shape)) ) 
            raise ValueError(err_str)
        if (rz.shape != zshape):
            err_str = ('Expected rz shape = %s received = %s' 
                       % (str(zshape), str(rz.shape)) ) 
            raise ValueError(err_str)
            
            
        # Reshape s to have same dimensions as sample
        ndims = len(pshape)
        if (ndims > 1):
            smat_shape = (self.ns,) + (ndims-1)*(1,)
            smat = self.s.reshape(smat_shape)
        else:
            smat = self.s
            
        # Get reduced versions of rp and rz
        rp1 = rp[:self.ns]
        rz1 = rz[:self.ns]
            
        # phat can be written as:
        #   phat = (gamz*s*rz + gamp*rp)/(s^2*gamz + gamp)
        ssq = np.abs(smat)**2
        phat = rp
        phat[:self.ns] = (gamp*rp1 + np.conj(smat)*gamz*rz1)/(gamp + ssq*gamz)
        
        # zhat = s*phat
        zhat = np.zeros(zshape)
        zhat[:self.ns] = smat*phat[:self.ns]
        
        # Compute derivatives
        alphap = np.mean(gamp/(np.abs(self.s)**2*gamz + gamp))
        alphap = alphap*(self.ns/self.nin) + (1-self.ns/self.nin)
        
        alphaz = np.mean(gamz*np.abs(self.s)**2/(np.abs(self.s)**2*gamz + gamp))
        alphaz = self.ns/self.nout * alphaz
                            
        return [phat,zhat], [alphap, alphaz]
        
    
 
    

    



