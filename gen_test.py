"""
generalization_test.py:  Tests the generalization error of ML-VAMP
"""
import numpy as np
import matplotlib.pyplot as plt
from layers_lin import SEGaussInLayer, SEGaussOutLayer, SELinLayer
from layers_nl import SELogisticOutLayer, SENLGaussOutLayer
from mlvamp_se import MLVampSE
from utils import RandDatGen, compute_test_err, compute_pvar_test, logistic_out
import pickle
import os
import argparse
from test_model import TestModel

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='Generalization error test')
parser.add_argument('--test_type', action='store',\
        help='test type: logistic, linear, nonlinear')
parser.set_defaults(test_type='linear')

parser.add_argument('--dist_type', action='store',\
        help='eigenvalue distribution type, const or lognormal')
parser.set_defaults(dist_type='const')

parser.add_argument('--logn_std',action='store',default=5.0,type=float,\
        help='lognormal standard deviation')
    
parser.add_argument('--logn_corr',action='store',default=1.0,type=float,\
        help='lognormal training-test correlation ')
    
parser.add_argument('--save_dat', dest='save_dat', action='store_true',\
        help="Save data to a file")
parser.set_defaults(save_dat=False)

parser.add_argument('--save_dir', action='store',\
        help='Directory to place results')
parser.set_defaults(save_dir='data')

parser.add_argument('--ntest',action='store',default=10,type=int,\
        help='Number of trials per parameter value')
parser.add_argument('--batch_num',action='store',default=0,type=int,\
        help='Batch number for output files when running in HPC')    
    
parser.add_argument('--reg_scale', action='store',default=1,type=float,\
        help='regularizer scaling relative to MAP value')
parser.add_argument('--plot', action='store_true',\
        help='Plot results at end')
parser.set_defaults(plot=False)

parser.add_argument('--mod_pkg', action='store',\
        help='package for the test model, {sklearn, tf}',\
        default='sklearn')
parser.add_argument('--nepochs',action='store',default=200,type=int,\
        help='Number of epochs (for SGD method only)')
parser.add_argument('--batch_size',action='store',default=100,type=int,\
        help='batch size (for SGD method only). 0=use batch_size = ny')
parser.add_argument('--beta_test',action='store',nargs='+',\
    default=np.linspace(0.1,3,20),type=float,\
    help='beta values to test test')
    
  
    
args = parser.parse_args()
test_type = args.test_type
dist_type = args.dist_type
logn_std = args.logn_std
logn_corr = args.logn_corr
save_dat = args.save_dat
save_dir = args.save_dir
ntest = args.ntest
reg_scale = args.reg_scale
plot_res = args.plot
batch_num = args.batch_num
mod_pkg = args.mod_pkg
nepochs = args.nepochs
batch_size = args.batch_size
beta_test = args.beta_test


test_type = 'nonlinear'
ntest = 1
reg_scale = 0.01
mod_pkg = 'tf'
plot_res = True


if not (test_type in ['logistic', 'linear', 'nonlinear']):
    raise ValueError('Unknown test type %s' % str(test_type) )
if not (mod_pkg in ['sklearn', 'tf']):
    raise ValueError('Unknown package %s' % str(mod_pkg))    
if not (dist_type in ['const', 'lognormal']):
    raise ValueError('Unknown distribution type %s' % dist_type)
if (test_type == 'nonlinear') and (mod_pkg == 'sklearn'):
    raise ValueError('sklearn has no method for non-linear optimization.\n' +\
                     'select --mod_pkg tf')
        
    
"""
Other parameters
"""    
nw  = 1000   # Number features
wvar = 1     # variance of w[i]
if test_type == 'logistic':
    ns = 1
    nit = 100
elif test_type == 'linear':
    ns = 10
    nit = 10
    snr = 10
elif test_type == 'nonlinear':
    ns = 10
    nit = 100
    snr = 20
    
test_grad = True
yerr_tgt = 0.05  # Minimum error (for logistic model only)    
pvar = 1.0   # Variance of input to output layer for linear model


# beta values to test
#beta_test = np.linspace(0.1,3,20)
#beta_test = [1.2]
nparam = len(beta_test)

# Intialize data
werr_sim = np.zeros((ntest, nparam))
werr_se  = np.zeros((ntest, nparam))
yerr_tr_sim = np.zeros((ntest, nparam))
yerr_ts_sim = np.zeros((ntest, nparam))
yerr_ts_cov = np.zeros((ntest, nparam))
yerr_ts_se = np.zeros((ntest, nparam))
grad_mse = np.zeros((ntest, nparam))
    

if test_type == 'linear':
    dvar0 = pvar*10**(-0.1*snr)
    out_true  = lambda p : p + np.random.normal(0,np.sqrt(dvar0),p.shape)
    out_model = lambda p : p
    score = lambda ytrue, yhat : 10*np.log10(np.mean(np.abs(ytrue-yhat)**2))
elif test_type == 'logistic':
    out_true  = lambda p : logistic_out(p, hard=False)
    out_model = lambda p : logistic_out(p, hard=True)
    score = lambda ytrue, yhat : np.mean(ytrue != yhat)
elif test_type == 'nonlinear':
    dvar0 = pvar*10**(-0.1*snr)
    out_true  = lambda p :\
        TestModel.fnl(p) + np.random.normal(0,np.sqrt(dvar0),p.shape)
    out_model = lambda p : TestModel.fnl(p)
    score = lambda ytrue, yhat : 10*np.log10(np.mean(np.abs(ytrue-yhat)**2))
        
for iparam, beta in enumerate(beta_test):
    for it in range(ntest):
 
        """
        Generate random instance of the model
        """
        ny = int(np.round(nw*beta))
        wshape = (nw,ns)
        yshape = (ny,ns)            
        Xshape = (ny,nw)
        wtrue = np.random.normal(0,1,wshape)
        
        # Generate data matrix
        gen = RandDatGen(Xshape, dist_type=dist_type,ssq_mean=pvar,\
                         logn_std=logn_std, logn_corr=logn_corr)
        if test_type == 'logistic':
            gen.set_ssq_mean_err(yerr_tgt=yerr_tgt)
        Xtr, Xts, s_tr, s_ts, s_mp, V0 = gen.sample()
        
        ptr = Xtr.dot(wtrue)        
        ytr = out_true(ptr)
                
        """
        Fit training data
        """
        if test_type == 'logistic':
            lam = reg_scale/wvar
        else:
            lam = reg_scale*dvar0/wvar
            
        # Create the model to test
        test_mod = TestModel(ny=ny,nw=nw,nout=ns,test_type=test_type,\
                             mod_pkg=mod_pkg, lam=lam, batch_size=0,\
                             verbose=0)
            
        # Fit the model and test on test data
        test_mod.fit(Xtr,ytr)
        if mod_pkg == 'tf':
            loss = test_mod.hist.history['loss']
        
        # Measure the parameter error
        what = test_mod.what
        werr_sim[it,iparam] = 10*np.log10(np.mean(np.abs(what-wtrue)**2))
        
        # Test the gradients to ensure that the method is converging
        # to the true minimla 
        grad, grad_in, grad_out = test_mod.test_grad(Xtr,ytr)
        grad_mse[it,iparam] = np.mean(grad**2)
        #print( 'Final MSE grad = %12.4e' % np.mean(grad**2))
        
        
        """
        Compute the test error via simulation
        """
        pts = Xts.dot(wtrue)
        yts = out_true(pts) 
        yts_hat = test_mod.predict(Xts) 
        yerr_ts_sim[it,iparam] = score(yts, yts_hat)
        
        
        """
        Compute the test error analytically from the covariance
        """
        ptrue = V0.dot(wtrue)
        phat  = V0.dot(what)    
        Pvar = compute_pvar_test(ptrue, phat, s_ts)        
        yerr_ts_cov[it, iparam] = compute_test_err(\
            Pvar, out_true, out_model, score)
                    
                        
        """
        Create the layers for the ML-VAMP solver
        """
        if test_type == 'logistic':
            lam_se = lam
        else:
            lam_se = lam/dvar0
        layer_in  = SEGaussInLayer(wshape,wvar,lam=lam_se,name='Input')
        layer_cov = SELinLayer([wshape, wshape], s=s_tr, name='Linear Cov')
        layer_mp  = SELinLayer([wshape, yshape], s=s_mp, name='Linear MP')
        if test_type == 'linear':
            layer_out = SEGaussOutLayer(yshape,dvar0=dvar0, dvar=dvar0, name='Output')
        elif test_type == 'logistic':
            layer_out = SELogisticOutLayer(yshape, damp_neg=0.5, use_prev=False)
        elif test_type == 'nonlinear':
            layer_out = SENLGaussOutLayer(\
                yshape, fnl=TestModel.fnl, fnl_grad=TestModel.fnl_grad,\
                dvar0=dvar0, damp_neg=0.5, use_prev=False)
        layers = [layer_in, layer_cov, layer_mp, layer_out]
    
        """
        Run the solver and compute the parameter and generalization error
        """
        solver = MLVampSE(layers,nit=nit,\
                          hist_list=['gam_neg', 'K_pos', 'gam_pos', 'tau_neg'] )
        solver.solve()
        gam_neg = np.array(solver.hist_dict['gam_neg'])
        gam_pos = np.array(solver.hist_dict['gam_pos'])
        K_pos   = np.array(solver.hist_dict['K_pos']) 
        tau_neg   = np.array(solver.hist_dict['tau_neg']) 
        K_pos = K_pos[:,2,:,:]
        
        werr_se[it,iparam] = 10*np.log10(solver.zerr[0])
        
        ptrue_se = layer_cov.ptrue
        phat_se  = layer_cov.phat
        Pvar_se = compute_pvar_test(ptrue_se, phat_se, s_ts) 
        yerr_ts_se[it, iparam] = compute_test_err(Pvar_se, out_true,\
                                                  out_model, score)
                  
        
        #print('beta=%7.2f werr_sim: %7.2e, werr_avg: %7.2e' % \
        #      (beta, werr_sim[it,iparam], werr_se[it,iparam]))
        print('beta=%7.2f it=%d yerr_cov: %7.2e, yerr_se: %7.2e grad=%12.4e' % \
              (beta, it, yerr_ts_cov[it,iparam], yerr_ts_se[it,iparam],\
               grad_mse[it,iparam]))
   
if 1:       
    # Take median values
    werr_sim_avg = np.median(werr_sim, axis=0)
    werr_se_avg = np.median(werr_se, axis=0)
    yerr_sim_avg = np.median(yerr_ts_sim, axis=0)
    yerr_cov_avg = np.median(yerr_ts_cov, axis=0)
    yerr_se_avg = np.median(yerr_ts_se, axis=0)
    
    if plot_res:
        plt.plot(beta_test, yerr_cov_avg, 'o', linewidth=3)
        plt.plot(beta_test, yerr_se_avg, linewidth=3)
            
    """
    Save results
    """
    if save_dat:
        # Create directory if necessary
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            print('Created directory %s' % save_dir)
            
        # Distribution string
        if dist_type == 'lognormal':
            dist_str = 'logn_std%02d_corr%03d' % (int(logn_std), int(logn_corr*100))
        else:
            dist_str = 'const'
        
        if reg_scale < 0.1:
            reg_str = 'reg_lo'
        else:
            reg_str = 'reg_hi'
        
        fn = ('%s_%02d_%s_%s.pkl'\
                        % (test_type, batch_num, dist_str, reg_str) )
        path = os.path.join(save_dir,fn)
        with open(path, 'wb') as fp:
            pickle.dump([beta_test, yerr_ts_sim, yerr_ts_cov, yerr_ts_se,\
                         werr_sim, werr_se, test_type, dist_type, logn_std, logn_corr ], fp)
        
        









