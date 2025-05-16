'''
Copyright 2015 Matthew Loper, Naureen Mahmood and the Max Planck Gesellschaft.  All rights reserved.
This software is provided for research purposes only.
By using this software you agree to the terms of the SMPL Model license here http://smpl.is.tue.mpg.de/license

More information about SMPL is available here http://smpl.is.tue.mpg.
For comments or questions, please email us at: smpl@tuebingen.mpg.de


About this file:
================
This file defines the basic skinning modules for the SMPL loader which 
defines the effect of bones and blendshapes on the vertices of the template mesh.

Modules included:
- verts_decorated: 
  creates an instance of the SMPL model which inherits model attributes from another 
  SMPL model.
- verts_core: [overloaded function inherited by lbs.verts_core]
  computes the blending of joint-influences for each vertex based on type of skinning

'''

import numpy as np
from . import lbs
from .posemapper import posemap
import scipy.sparse as sp

# For backward compatibility, try to import chumpy if available
try:
    import chumpy
    HAS_CHUMPY = True
except ImportError:
    HAS_CHUMPY = False
    # Create a minimal chumpy replacement class for basic functionality
    class MinimalChumpy:
        def __init__(self):
            pass
            
        def zeros(self, shape):
            return np.zeros(shape)
            
        def vstack(self, arrays):
            return np.vstack(arrays)
    
    chumpy = MinimalChumpy()

def ischumpy(x): 
    if HAS_CHUMPY:
        return hasattr(x, 'dterms')
    return False

class MatVecMult:
    """Replacement for chumpy's MatVecMult when chumpy is not available"""
    def __init__(self, matrix, vector):
        self.matrix = matrix
        self.vector = vector
        
    def __array__(self):
        return self.matrix.dot(self.vector)

def verts_decorated(trans, pose, 
    v_template, J, weights, kintree_table, bs_style, f,
    bs_type=None, posedirs=None, betas=None, shapedirs=None, want_Jtr=False):

    if HAS_CHUMPY:
        for which in [trans, pose, v_template, weights, posedirs, betas, shapedirs]:
            if which is not None:
                assert ischumpy(which)

    v = v_template

    if shapedirs is not None:
        if betas is None:
            if HAS_CHUMPY:
                betas = chumpy.zeros(shapedirs.shape[-1])
            else:
                betas = np.zeros(shapedirs.shape[-1])
        v_shaped = v + shapedirs.dot(betas)
    else:
        v_shaped = v
        
    if posedirs is not None:
        v_posed = v_shaped + posedirs.dot(posemap(bs_type)(pose))
    else:
        v_posed = v_shaped
        
    v = v_posed
        
    if sp.issparse(J):
        regressor = J
        if HAS_CHUMPY:
            J_tmpx = MatVecMult(regressor, v_shaped[:,0])        
            J_tmpy = MatVecMult(regressor, v_shaped[:,1])        
            J_tmpz = MatVecMult(regressor, v_shaped[:,2])        
            J = chumpy.vstack((J_tmpx, J_tmpy, J_tmpz)).T
        else:
            J_tmpx = regressor.dot(v_shaped[:,0])
            J_tmpy = regressor.dot(v_shaped[:,1])
            J_tmpz = regressor.dot(v_shaped[:,2])
            J = np.vstack((J_tmpx, J_tmpy, J_tmpz)).T
    else:    
        if HAS_CHUMPY:
            assert(ischumpy(J))
        
    assert(bs_style=='lbs')
    result, Jtr = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr=True, 
                                xp=chumpy if HAS_CHUMPY else np)
     
    tr = trans.reshape((1,3))
    result = result + tr
    Jtr = Jtr + tr

    result.trans = trans
    result.f = f
    result.pose = pose
    result.v_template = v_template
    result.J = J
    result.weights = weights
    result.kintree_table = kintree_table
    result.bs_style = bs_style
    result.bs_type = bs_type
    if posedirs is not None:
        result.posedirs = posedirs
        result.v_posed = v_posed
    if shapedirs is not None:
        result.shapedirs = shapedirs
        result.betas = betas
        result.v_shaped = v_shaped
    if want_Jtr:
        result.J_transformed = Jtr
    return result

def verts_core(pose, v, J, weights, kintree_table, bs_style, want_Jtr=False, xp=None):
    
    if xp is None:
        xp = np if not HAS_CHUMPY else chumpy
        
    if HAS_CHUMPY and xp == chumpy:
        assert(hasattr(pose, 'dterms'))
        assert(hasattr(v, 'dterms'))
        assert(hasattr(J, 'dterms'))
        assert(hasattr(weights, 'dterms'))
     
    assert(bs_style=='lbs')
    result = lbs.verts_core(pose, v, J, weights, kintree_table, want_Jtr, xp)

    return result 