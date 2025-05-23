'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This module defines the mapping of joint-angles to pose-blendshapes. 

Modules included:
- posemap:
  computes the joint-to-pose blend shape mapping given a mapping type as input

'''

import numpy as np
import cv2

# For backward compatibility, try to import chumpy if available
try:
    import chumpy as ch
    HAS_CHUMPY = True
    
    class Rodrigues(ch.Ch):
        dterms = 'rt'
        
        def compute_r(self):
            return cv2.Rodrigues(self.rt.r)[0]
        
        def compute_dr_wrt(self, wrt):
            if wrt is self.rt:
                return cv2.Rodrigues(self.rt.r)[1].T
                
except ImportError:
    HAS_CHUMPY = False
    # Create a minimal replacement for Rodrigues when chumpy is not available
    class Rodrigues:
        def __init__(self, rt):
            self.rt = rt
            
        def __array__(self):
            return cv2.Rodrigues(self.rt)[0]


def lrotmin(p): 
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0]-np.eye(3)).ravel() for pp in p.reshape((-1,3))]).ravel()
    
    if HAS_CHUMPY:
        if p.ndim != 2 or p.shape[1] != 3:
            p = p.reshape((-1,3))
        p = p[1:]
        return ch.concatenate([(Rodrigues(pp)-ch.eye(3)).ravel() for pp in p]).ravel()
    else:
        raise NotImplementedError("lrotmin for non-numpy arrays requires chumpy")

def posemap(s):
    if s == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s' % (str(s),)) 