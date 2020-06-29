"""
mlvamp_se.py:  Solver for the ML-VAMP SE
"""
import numpy as np
import matplotlib.pyplot as plt
import solver


class MLVampSE(solver.Solver):
    def __init__(self,layers,nit=1,hist_list=[]):
        """
        ML-VAMP SE solver
        
        Parameters
        ----------
        layers : list of `SELayer`
            list of the layers in the sequential model
        nit : int
            number of iterations 
        hist_list:  list of strings
            variables to save on each iteration
        """
        
        # Save estimators and message handlers
        self.layers = layers
        self.nlayers = len(layers)
        
        # Check layer types and shapes
        if self.layers[0].layer_type != 'input':
            raise ValueError('layers[0] must be input')
        if self.layers[0].shape != self.layers[1].shape[0]:
            err_str = 'z[0].shape=%s does not match p[1].shape=%s' \
                (str(self.layers[0].shape), str(self.layers[1].shape[0]))
            raise ValueError(err_str)
        for i in range(1,self.nlayers-1):
            if self.layers[i].layer_type != 'middle':
                raise ValueError('layers[%d] must be middle' % i)
            if self.layers[i].shape[1] != self.layers[i+1].shape[0]:
                err_str = 'z[%d].shape=%s does not match p[%d].shape=%s' \
                    (i, str(self.layers[i].shape), i+1,\
                     str(self.layers[1].shape[0]))
                raise ValueError(err_str)
        if self.layers[self.nlayers-1].layer_type != 'output':
            raise ValueError('final layer must be output')
                    
                
        # Other parameters
        self.nit = nit        
        solver.Solver.__init__(self, hist_list)
        
    def summary(self):
        """
        Prints summary of the layers
        """
        fmt = '%-4s | %-18s | %-18s | %-12s | %-12s'
        print(fmt % ('', 'Layer', 'Type', 'p[i]', 'z[i]'))
        for i, layer in enumerate(self.layers):
            if layer.layer_type == 'input':
                print( fmt % (str(i), layer.name, layer.layer_class_name,\
                              '-', str(layer.shape)))
            else:
                print( fmt % (str(i), layer.name, layer.layer_class_name,\
                              str(layer.shape[0]), str(layer.shape[1])))
         
        
    def fwd_true(self):
        """
        Runs through the 'true' system in the forward pass.

        Returns
        -------
        None.

        """        
        taui = self.layers[0].compSETrue()
        self.tau_true = [taui]
        for i in range(1,self.nlayers):
            taui = self.layers[i].compSETrue(taui)
            self.tau_true.append(taui)
 
        # Initialize SE variables
        self.K_pos = []
        self.tau_neg = np.zeros(self.nlayers-1)
        self.gam_neg = np.tile(1e-6, (self.nlayers-1,))
        self.gam_pos = np.tile(1e-6, (self.nlayers-1,))
        
            
            
    def fwd_est(self):
        """
        Runs one iteration in the forward pass

        """
        self.K_pos = []        
        self.zerr = np.zeros(self.nlayers-1)
        K1, gam1_pos, zerri = self.layers[0].compSEEst(
            tau1=self.tau_neg[0], gam1_neg=self.gam_neg[0], dir='fwd')
        self.gam_pos[0] = gam1_pos
        self.K_pos.append(K1)
        self.zerr[0] = zerri
        
        for i in range(1,self.nlayers-1):
            layer = self.layers[i]
            K1, gam1_pos, zerri = layer.compSEEst(\
                K0=self.K_pos[i-1], gam0_pos=self.gam_pos[i-1], 
                tau1=self.tau_neg[i], gam1_neg=self.gam_neg[i], dir='fwd')
            self.gam_pos[i] = gam1_pos
            self.K_pos.append( K1 )
            self.zerr[i] = zerri
            
    def rev_est(self):
        """
        Run one iteration in the reverse pass
        """
        
        self.perr = np.zeros(self.nlayers-1)
        i = self.nlayers-1
        layer = self.layers[i]
        tau0, gam0_neg, perri = layer.compSEEst(K0=self.K_pos[i-1],\
            gam0_pos=self.gam_pos[i-1], dir='rev')
        self.tau_neg[i-1] = tau0
        self.gam_neg[i-1] = gam0_neg
        self.perr[i-1] = perri
        
        for i in range(self.nlayers-2,0,-1):
            layer = self.layers[i]
            tau0, gam0_neg, perri = layer.compSEEst(\
                K0=self.K_pos[i-1],   gam0_pos=self.gam_pos[i-1],\
                tau1=self.tau_neg[i], gam1_neg=self.gam_neg[i],dir='rev')
            self.tau_neg[i-1] = tau0
            self.gam_neg[i-1] = gam0_neg
            self.perr[i-1] = perri
                        
        
    def solve(self):
        """
        Runs the iterations of the SE equations
        """
        
        # Initial estimate
        self.fwd_true()

        # Loop over iterations
        for it in range(self.nit):
            
            # Forward pass
            self.fwd_est()
            
            # Reverse pass
            self.rev_est()
                        
            # Save history
            self.save_hist()
