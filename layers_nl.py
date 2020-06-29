"""
layers_nl.py:  Non-linear layers
"""
from se_base import SELayer
import numpy as np
import matplotlib.pyplot as plt


class SENLOutLayer(SELayer):
    def __init__(self,pzshape,use_prev=True,**kwargs):
        """
        Base class for a non-linear output layer
        
        The sampling operator, samp(), is defined in derived class.
        
        The estimator minimizes the MAP penalty for a function, 'f()' defined 
        by the function 'feval()'
        
        Parameters
        ----------
        pzshape : array
            shape of `p`= shape of `z`
        use_prev : Boolean
            indicates if previous value should be used as starting point
            in next iteration
        """     
        self.use_prev = use_prev
        self.plast = None
        SELayer.__init__(self,shape=[pzshape,pzshape], layer_type='output',\
                          **kwargs)
        
    def feval(self,p):
        """
        Penalty function for the class.   
        
        Parameters
        ----------
        p:  Function input
        
        Returns:
        --------
        f:  Value of the function
        fgrad:  Gradient of the function
        fhess:  Hessian of the function
        """
        raise NotImplementedError("Must implement the feval function")
        
    def prox_eval(self,p,rp,gamp):
        """
        Proximal objective function:
            
            H(p,rp,gamp) = f(p) + gamp*(p-rp)**2/2
        
        Parameters
        ----------
        p:  Function input
        
        Returns:
        --------
        f:  Value of the function
        fgrad:  Gradient of the function
        fhess:  Hessian of the function
        """
        f, fgrad, fhess = self.feval(p)
        
        d = p-rp
        H = f + np.sum(d**2)*gamp/2
        Hgrad = fgrad + gamp*d
        Hhess = fhess + gamp
        
        return H, Hgrad, Hhess
        
        
    def test_grad(self,p0=None,step=1e-6,tol=1e-2, verbose=True):
        """
        Test gradients and Hessians 
        
        Parameters
        ----------
        p0:  Value to test gradient
            If `None`, then take `p = N(0,1)`
        tol:  relative error tolerance before raising error
        """
        
        # Initial point
        if p0 is None:
            p0 = np.random.normal(0,1,pzshape)
        
        # Second point
        p1 = p0 + step*np.random.normal(0,1,pzshape)
        
        # Evaluate at two points
        f0, fgrad0, fhess0 = self.feval(p0)
        f1, fgrad1, fhess1 = self.feval(p1)
        
        # Test gradient
        df = f1 - f0
        df_est = np.sum(fgrad0*(p1-p0))
        err_grad = (np.abs(df-df_est) > \
                    tol*np.maximum(np.abs(df),np.abs(df_est)))
        if err_grad or verbose:
            print('df =   %12.4e  df_est=   %12.4e' % (df, df_est) )
        if err_grad:
            raise ValueError('Gradients do not match within tolerance.')
          
        # Test Hessian
        dgrad = np.sum(fgrad1-fgrad0)
        dgrad_est = np.sum(fhess0*(p1-p0))
        
        err_hess = (np.abs(dgrad-dgrad_est) > \
                    tol*np.maximum(np.abs(dgrad),np.abs(dgrad_est)))
        if err_hess or verbose:
            print('dgrad = %12.4e dgrad_est=%12.4e' % (dgrad, dgrad_est) )
        if err_hess:
            raise ValueError('Hessians do not match within tolerance.')
        if verbose:
            print('Gradients and Hessian passed within tolerance!')
        
        
    def est(self,rp,gamp,nsamp=None):
        """
        Performs the minization:
            
            phat = argmin_p f(p) + (gamp/2)*|p-rp|^2
                        
        via Newton´s Raphson.  
        
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
        
        if (not self.use_prev) or (self.plast is None):
            p0 = rp
        else:
            p0 = self.plast
        self.step = 1e-3
        self.nit = 20
        self.fhess_min = 0.1
        
        # History
        self.Hhist = np.zeros(self.nit)
        self.stephist = np.zeros(self.nit)
        
        # Take initial gradient and Hessian
        H, Hgrad, Hhess = self.prox_eval(p0,rp,gamp)
        
        
        for it in range(self.nit):
            
            # Take test point
            Hhess_pos = np.maximum(Hhess, self.fhess_min)
            p1 = p0 - self.step*Hgrad/Hhess_pos
            H1, Hgrad1, Hhess1 = self.prox_eval(p1,rp,gamp)
            
            # Compute expected decrease in function
            dp = p1 - p0
            dH_est = np.sum(Hgrad*dp)
                        
            # Test the point
            alpha = 0.5
            beta = 2.0
            if (H1 - H < alpha*dH_est) and (H1 < H):
                # If point passed, save value and increase step size
                p0 = p1
                H = H1
                Hgrad = Hgrad1
                Hhess = Hhess1
                
                self.step = np.minimum(beta*self.step, 1)
                
            else:
                # If point passes, reduce the step size
                self.step = np.maximum(self.step/beta, 1e-6)
                
            # Save values
            self.Hhist[it] = H
            self.stephist[it] = self.step
            
        """
        Compute the derivative.  We know,
        
            f'(p) + gamp*(p-rp) = 0
          
        Therefore, alphap*(f''(p) + gamp)  = gamp
        """
        alphap = np.mean(gamp/Hhess)
        
        self.plast = p0
        
        return p0, alphap
                
            
    
class SELogisticOutLayer(SENLOutLayer):
    
    def __init__(self,pzshape,scale=1.0, **kwargs):
        """
        Logistic output layer
        
        The sampling operator is the logistic output:
        
            P(z0=1) = 1/(1 + exp(-scale*p0)) 
        
        The estimator is the MAP penalty with 
        
            f(p,z)=-log P(z0) = log(1 + exp(scale*p)) - scale*p*y

        Parameters
        ----------
        pzshape : array
            shape of `p`= shape of `z`
        scale : scalar 
            scale factor
        """
        self.scale = scale
        SENLOutLayer.__init__(self,pzshape=pzshape,\
                              layer_class_name='LogisticOut', **kwargs)
   
    def sample(self,ptrue,nsamp=None):
        """
        Produces logistic output
        
            P(z0 =1) = 1/(1 + exp(-scale*p0)) 
        
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
            
        # Create the binary output
        prob = 1/(1 + np.exp(-ptrue))
        u = np.random.uniform(0,1,pshape)
        ztrue = (u < prob).astype(int)
        self.u = u
         
        # Save true signal
        self.y = ztrue
        return ztrue
        
    def feval(self, p):
        """
        Logistic enalty function  
            
            f(p,y) = -log P(z0) = log(1 + exp(u)) - u*y,  u = scale*p
            
        
        Parameters
        ----------
        p:  Function input
        
        Returns:
        --------
        f:  Value of the function
        fgrad:  Gradient of the function
        fhess:  Hessian of the function
        """
        
        """
        To make the implementation numerically stable, we use the fact that
        
            if u > 0, f(p,y) = log(1 + exp(-u)) + (1-y)*u
            if u < 0, f(p,y) = log(1 + exp(u)) -u*y
            
        Hence,
             f(p,y) = log(1 + exp(-|u|)) - max(u,0) - u*y
        
        """
        u = self.scale*p
        fvec = np.log(1 + np.exp(-np.abs(u))) + np.maximum(u,0) - u*self.y
        #fvec = np.log(1 + np.exp(u)) - u*self.y
        f = np.sum(fvec)
        
        # Compute gradient
        u = np.maximum(-15,np.minimum(15,u))
        pz = 1/(1+np.exp(-u)) 
        fgrad = pz - self.y 
        
        # Compute Hessian
        fhess = pz*(1-pz)
        return f, fgrad, fhess
    
class SENLGaussOutLayer(SENLOutLayer):
    
    def __init__(self,pzshape,fnl,fnl_grad,dvar0,**kwargs):
        """
        Non-linear function with additive Gaussian noise
        
        The sampling operator is a general function:
        
            z = fnl(p) + d,  d~N(0,dvar0)
        
        The estimator is the non-linear least-squares penalty:
        
            f(p,z)=(fnl(p)-y)**2/(2*dvar0)

        Parameters
        ----------
        pzshape : array
            shape of `p`= shape of `z`
        fnl : function
            non-linear output function
        fnl_grad : function
            function gradient
        dvar0 : scalar
            noise variance
        """
        self.fnl = fnl
        self.fnl_grad = fnl_grad
        self.dvar0 = dvar0
        SENLOutLayer.__init__(self,pzshape=pzshape,\
                              layer_class_name='NLGaussOut', **kwargs)
   
    def sample(self,ptrue,nsamp=None):
        """
        Produces non-linear output
        
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
        ztrue = self.fnl(ptrue) + dtrue
        self.y = ztrue
        
        return ztrue
        
    def feval(self, p):
        """
        Non-linear least squares penalty 
            
            f(p,y) = (fnl(p)-y)**2/(2*dvar0)
            
        
        Parameters
        ----------
        p:  Function input
        
        Returns:
        --------
        f:  Value of the function
        fgrad:  Gradient of the function
        fhess:  Hessian of the function
        """
        
        
        d = self.fnl(p) - self.y
        f = np.sum(d**2)/(2*self.dvar0)
        
        # Compute gradient
        fnl_gradp = self.fnl_grad(p)
        fgrad = d*fnl_gradp/self.dvar0
        
        # Approximate Hessian
        fhess = fnl_gradp**2/self.dvar0
        return f, fgrad, fhess    

if __name__ == "__main__":
    
    nz = 1000
    ns = 10
    gamp = 2
    rvar = 1.5
    pzshape = (nz,ns)
    
    layer = SELogisticOutLayer(pzshape=pzshape)
    
    # Generate inputs
    rp = np.random.normal(0,np.sqrt(rvar), pzshape)
    ptrue = rp + np.random.normal(0, np.sqrt(1/gamp), pzshape)
    
    # Generate outputs
    ztrue = layer.sample(ptrue)
    
    # Test gradient
    layer.test_grad()
    
    layer.est(rp, gamp)
    
    nit = len(layer.Hhist)
    
    plt.plot(np.arange(nit), layer.Hhist)
    
    
    
    
    
    
    
    
    
    


