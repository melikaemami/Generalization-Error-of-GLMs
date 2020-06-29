"""
Plots results from a directory
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import argparse

matplotlib.rcParams.update({'font.size': 16})

"""
Get arguments
"""
# Define the parser
parser = argparse.ArgumentParser(description='Generalization error test')
parser.add_argument(\
    '--test_type', action='store',\
    help='test type: {logistic, linear, nonlinear}',\
    default='linear')
parser.add_argument(\
    '--data_dir', action='store',\
    help='directory to find files', default='')
parser.add_argument(\
    '--output', action='store',\
    help='output path', default='')
    
# Get arguments
args = parser.parse_args()
test_type = args.test_type
data_dir = args.data_dir
output = args.output

# Set defaults
if data_dir == '':
    data_dir = 'data_%s' % test_type
if output == '':
    output = '%s_sim_v1.png' % test_type
    


def load_data(data_dir, fn_fmt):
    """
    Loads data from multiple files from different batches

    Parameters
    ----------
    data_dir : str
        Name of the directory 
    fn_fmt : str
        File name format with %s for the batch number

    Returns
    -------
    beta_test : ndarray 
        List of beta values tested 
    yerr_cov_avg : ndarray
        True median generalization error over trials from all files
    yerr_se_avg : ndarray
        SE median generalization error over trials from all files
    """
    i = 0
    done = False
    while not done:
        # Get the file name
        fn = fn_fmt % (i)
        path = os.path.join(data_dir,fn)
        
        # Check if file name exists
        if os.path.exists(path):
        
            with open(path, 'rb') as fp:
                    beta_test, yerr_ts_simi, yerr_ts_covi, yerr_ts_sei,\
                    werr_sim, werr_se, test_type, dist_type, logn_std, logn_corr = pickle.load(fp)
            if (i==0):
                yerr_ts_cov = yerr_ts_covi
                yerr_ts_se = yerr_ts_sei
            else:
                yerr_ts_cov = np.vstack((yerr_ts_cov, yerr_ts_covi))
                yerr_ts_se = np.vstack((yerr_ts_cov, yerr_ts_sei))
            i += 1
        else:
            done = True
    
    if (i==0):
        raise ValueError('No files found %s' % fn)
    
    print('%d files found' % i)
    yerr_cov_avg = np.median(yerr_ts_cov, axis=0)
    yerr_se_avg = np.median(yerr_ts_se, axis=0)
    
    return beta_test, yerr_cov_avg, yerr_se_avg

if test_type == 'linear':
    fn_fmts = ['linear_%02d_const_reg_lo.pkl',\
                'linear_%02d_logn_std05_corr100_reg_lo.pkl',\
                'linear_%02d_logn_std05_corr050_reg_lo.pkl']
    data_dir = 'data_linear'    
    ylabel = 'Test MSE (dB)'
elif test_type == 'logistic':
    fn_fmts = ['logistic_%02d_const_reg_hi.pkl',\
               'logistic_%02d_logn_std05_corr100_reg_hi.pkl', \
               'logistic_%02d_logn_std05_corr050_reg_hi.pkl']
    data_dir = 'data_logistic'
    ylabel = 'Test error'
elif test_type == 'nonlinear':
    fn_fmts = ['nonlinear_%02d_const_reg_hi.pkl',\
               'nonlinear_%02d_logn_std05_corr100_reg_hi.pkl', \
               'nonlinear_%02d_logn_std05_corr050_reg_hi.pkl']
    data_dir = 'data_nonlin'
    ylabel = 'Test MSE (dB)'


plt.figure(figsize=(8,6))
for i, fn_fmt in enumerate(fn_fmts):
    beta_test, yerr_cov_avg, yerr_se_avg = load_data(data_dir, fn_fmt)
    
    col = 'C%d' % i
    plt.plot(beta_test, yerr_cov_avg,'o', color=col, linewidth=3)
    plt.plot(beta_test, yerr_se_avg, color=col, linewidth=2)
    
plt.grid()
plt.legend(['i.i.d. (sim)', 'i.i.d. (SE)', 'corr (sim)',\
            'corr (SE)', 'corr+mismatch (sim)', 'corr+mismatch (SE)'])
plt.xlabel('Sample ratio N/p')
plt.ylabel(ylabel)
plt.show()
plt.savefig(output, dpi=600)
plt.xlim([np.min(beta_test), np.max(beta_test)])
        