"""
utils.py:  Simple utility functions for the test
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.stats import expon, beta as beta_dist


def logistic_out(p, hard=False):
    """
    Logistic output function
 
    Parameters
    ----------
    p : array
        logistic input
    hard : Boolean
        If True, use a hard classification output.  The default is False.

    Returns
    -------
    y : array
        Binary ouptut class label, 0 or 1

    """
    if hard:
        y= (p >0)
    else:
        prob = 1/(1 + np.exp(-p))
        u = np.random.uniform(0,1,p.shape)
        y = (u < prob)
    y = y.astype(int)
    return y

def compute_pvar_test(ptrue, phat, s_ts):
    """
    Compute the sample covariance for the test error.
    
    Given samples of `ptrue` and `phat`, computes the average of
    
       Pvar = Exp( s_ts*2*[ptrue, phat]*[ptrue phat].T)
       
    This matrix is used for estimating the test error

    Parameters
    ----------
    ptrue : array
        Samples of the true values 
    phat : array
        Samples of the test values
    s_ts : array
        Samples of the test eigenvalues

    Returns
    -------
    Pvar : (2,2) array
        Covariance matrix described above
    """
    # Compute the square of the eigenvalues
    s_ts_sq = s_ts**2
    
    # Reshape for broadcasting
    ndim = len(ptrue.shape)
    if (ndim > 1):
        shape = np.ones(ndim, dtype=int)
        shape[0] = len(s_ts_sq)
        s_ts_sq = np.reshape(s_ts_sq, shape)
        
    # Compute sample covariance
    P11 = np.mean(s_ts_sq*(ptrue**2))
    P12 = np.mean(s_ts_sq*(ptrue*phat))
    P22 = np.mean(s_ts_sq*(phat**2))
    Pvar = np.array([[P11,P12], [P12,P22]])
    return Pvar

def compute_test_err(Pvar, out_true, out_model, score, shape=(1000,)):
    """
    Estimates the test error using the model:
    
        test_err = E[score(Ztrue, Zest)]
        Ztrue = out_true( Ptrue )
        Zhat  = out_mod( Phat )
        
        (Ptrue, Phat)~ N(0, pvar)
        
    Parameters
    ----------
    pvar : (2,2) array
        covariance matrix of `(Ptrue, Phat)`
    out_true : function
        True output function, should take `shape` inputs and outputs
    out_model : function
        Model output function, should take `shape` inputs and outputs
    score :  score function
    shape :  shape of samples to test
    
    Returns
    -------
    test_err : scalar
        Expected test error
    """
        
    # Generate random data
    ns = np.prod(shape)
    u = np.random.normal(0,1,(ns,2))
    Pvar_sqrt = scipy.linalg.sqrtm(Pvar)
    p = u.dot(Pvar_sqrt)
    ptrue = np.reshape(p[:,0], shape)
    phat  = np.reshape(p[:,1], shape)
    
    # Compute outputs
    ztrue = out_true(ptrue)
    zhat  = out_model(phat)
    
    # Compute score
    score_avg = score(ztrue, zhat)
    return score_avg
    
    

class RandDatGen(object):
    def __init__(self,shape,dist_type='const',ssq_mean=1.0,logn_std=10,logn_corr=1.0,
                 uniform_high=10.0,uniform_low=.1,beta_dist_alpha=2,beta_dist_beta=5,
                 mismatch=False,mismatch_epsilon=.05):
        """
        Random data generator
        
        Create a random data matrix `X`.
        
        The shape of the matrix is `shape = (n,p)`.  
        The matrix `Xtr = U*diag(str)*V0` where `V0` is Haar distributed,
        `str` are eigenvalues of the training data and `U` has i.i.d. `N(0,1/p)`.
        
        There are two distribution supported for `(str,sts)`:
            
        *  `const` :  `(str,sts) = sqrt(ssq_mean)`
        *  `lognormal`:
            
               (str,sts) = A * 10**(0.1*vtr, 0.1*vts)
            
           where `(vtr,vts)` are bivariate Gaussians with standard deviations
           of `logn_std`, zero mean and correlation coefficient of `logn_corr`.  
           The constant `A` is selected to make sure 
           `E(str**2) = E(sts**2) = ssq_mean`.  
        *  `
          
            
        Parameters
        ----------
        shape:  (2,) array
           Desired shape of X
        dist_type :  'const', 'lognormal', 'uniform', 'exponential', 'beta'
            Distribution of the eigenvalues
        ssq_mean :  scalar 
            Mean value `E(str**2) = E(sts**2)`. 
        logn_var :  scalar
            Lognormal variance
        logn_corr : scalar
            Lognormal correlation coefficient   
        uniform_low : scalar
            Uniform distribution lower_bound
        uniform_high : scalar
            Uniform distribution upper_bound
        beta_dist_alpha : sclar
            Beta distribution parameter ::alpha::
        beta_dist_beta : sclar
            Beta distribution parameter ::beta::
        mismatch : boolean
            if True, there is a mismatch between training and test covariance matrices
        mismatch_epsilon : scalar
            Prob(S_tr != S_ts)
        """
        self.shape = shape
        self.dist_type = dist_type
        self.ssq_mean = ssq_mean
        self.logn_std= logn_std
        self.logn_corr = logn_corr
        self.uniform_low = uniform_low
        self.uniform_high = uniform_high
        self.beta_dist_alpha = beta_dist_alpha
        self.beta_dist_beta = beta_dist_beta
        self.mismatch = mismatch
        self.mismatch_epsilon = mismatch_epsilon
            
    def set_ssq_mean_err(self, yerr_tgt=0.05,plot=False,\
                        scale_min=0.1, scale_max=20):
        """
        Sets the scaling to obtain a target error probability
        
        This is used for logistic regression problems.  
        Given a scale `s=ssq_mean`,  the error is:
        
            err(s) = P(Yhat != Y)
            
        where `Y` is the logistic output, and `Yhat` is the hard decision:
        
            P(Y = 1) = 1/(1 + exp(-P)),  P = N(0,ssq_mean)
            Yhat = 1  if  (P > 0)
            
        The scale indicates the standard deviation to obtain a minimum 
        prediction error in a logistic model

        Parameters
        ----------
        yerr_tgt : scalar
            Target error probability
        plot : Boolean
            If routine plots error vs. scale
        """
 
        # Generate random Gaussian inputs       
        npts = int(1e4)
        p = np.random.normal(0,1,npts)
        
        # Compute hard decisions on the inputs
        y0 = (p > 0).astype(int)
        
        # Loop over scale values
        scale_test = np.linspace(scale_min,scale_max,100)
        ntest = len(scale_test)
        yerr = np.zeros(ntest)
        u = np.random.uniform(0,1,(npts,))
        for i, scale in enumerate(scale_test):
            prob = 1/(1 + np.exp(-scale*p))        
            y = (u < prob).astype(int)
            yerr[i] = np.mean(y != y0)
        
        if plot:
            plt.plot(scale_test, yerr)
            plt.grid()
            
        
        # Find the scale level to obtain the desired target
        # We flip the arguments since the interp function expects
        # input values that are increasing 
        scale_opt = np.interp(yerr_tgt, np.flip(yerr), np.flip(scale_test))
        self.ssq_mean = scale_opt**2
        
    def sample(self):
        """
        Returns a sample of the matrix

        Returns
        -------
        Xtr : (n,p) array
            training data      
        Xts : (n,p) array
            test data
        s_tr:  (p,) array
            training eigenvalues
        s_ts:  (p,) array
            test eigenvalues
        s_mp:  (p,) array
            singular values of `Utr`.  These follow the Marcenko-Pastur distribution
        V0: (p,p) array
            training and test eigenvectors          
        """
        
        # Generate random eigenvalues
        nw = self.shape[1]  # Number of features
        if self.dist_type == 'const':
            s_tr = np.tile(np.sqrt(self.ssq_mean), nw)
            s_ts = np.tile(np.sqrt(self.ssq_mean), nw)
        elif self.dist_type == 'lognormal':
            P = np.array([[1, self.logn_corr], [self.logn_corr, 1]])
            Psqrt = scipy.linalg.sqrtm(P)*self.logn_std
            v = np.random.normal(0,1,(nw,2)).dot(Psqrt)
            s = 10**(0.1*v)
            s = self.ssq_mean/np.mean(s)*s
            s_tr = np.sqrt(s[:,0])
            s_ts = np.sqrt(s[:,1])
        elif self.dist_type == 'uniform':
            s = np.random.uniform(low=self.uniform_low, high=self.uniform_high, size=nw)
            s_tr = np.sqrt(s)
            s_ts = np.sqrt(s)
        elif self.dist_type == 'exponential':
            s = np.sqrt(expon.rvs(size=nw))
            s_tr = np.sqrt(s)
            s_ts = np.sqrt(s)
        elif self.dist_type == 'beta':
            s = beta_dist.rvs(a=self.beta_dist_alpha,b=self.beta_dist_beta,size=nw)
            s_tr = np.sqrt(s)
            s_ts = np.sqrt(s)
        else:
            raise ValueError('Unknown distribution')
        
        if self.mismatch:
            e = np.random.binomial(1,self.mismatch_epsilon,s_tr.shape)
            s_ts = e*s_tr + (1-e)*(max(s_tr)-s_tr)
        
        # Generate random V0
        X = np.random.normal(0,1,(nw,nw))
        _, _, V0 = np.linalg.svd(X)
    
        # Generate random training data
        Utr = np.random.normal(0,1/np.sqrt(nw),self.shape)
        Xtr = Utr.dot(s_tr[:,None]*V0)
        s_mp = np.linalg.svd(Utr, compute_uv=False)
 
        # Generate random test data       
        Uts = np.random.normal(0,1/np.sqrt(nw),self.shape)
        Xts = Uts.dot(s_ts[:,None]*V0)

        
        return Xtr, Xts, s_tr, s_ts, s_mp, V0

if __name__ == "__main__":
    
    shape = (1000,10)
    gen = RandDatGen(shape)
    gen.set_scale_error(yerr_tgt = 0.1)


