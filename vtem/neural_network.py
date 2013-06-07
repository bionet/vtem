#!/usr/bin/env python

import time

import pycuda.driver as cuda
from pycuda.tools import dtype_to_ctype
import numpy as np
from scikits.cuda.cublas import cublasDnrm2, cublasSnrm2, cublasCreate, cublasDestroy

import utils.parray as parray
import utils.linalg as la
from utils.kernel_utils import *


def rnn3(G, q, dt = 1e-6, alpha = 5000, steps = 4000, XOUTPUTSTEPS = None):
    """
    Solving the decoding problem using a recurrent neural network.
    
    Parameters
    ----------
    G: PitchArray
       Must be real and positive semidefinite.
    q: PitchArray
       The measurements from spikes
    dt: float (optional)
        the time step in simulating the continuous network
    alpha: float (optional)
           scaling factor
    steps: int (optional)
           the number of steps to run the network
    XOUTPUTSTEPS: int (optional)
           The number of steps that are returned.
           If using default None, only return the final result.
    
    Return
    ------
    c: PitchArray
       The approximate solution to the decoding problem
    output: PitchArray (optional)
            If XOUTPUTSTEPS is not None, the full output specified
    """
    if G.dtype != q.dtype:
        raise TypeError("matrix multiplication must have same dtype")

    if np.iscomplexobj(G):
        raise TypeError("RNN currently only solves real types")

    if (len(G.shape) != 2) | (len(q.shape) != 2):
        raise TypeError("G, q must both be matrices")

    if XOUTPUTSTEPS is None:
        XOUTPUTSTEPS = min(20, steps)
        x_steps = steps / XOUTPUTSTEPS
        fullout = False
    else:
        fullout = True
        x_steps = steps / int(XOUTPUTSTEPS)
        output = parray.empty((XOUTPUTSTEPS, q.size), q.dtype)

    c = parray.zeros_like(q)
    update_func = get_rnn3_update_func(G.dtype)

    dt = float(dt)
    alpha = float(alpha)

    y = parray.empty_like(q)

    if y.dtype == np.float64:
        normfunc = cublasDnrm2
    else:
        normfunc = cublasSnrm2

    grid = (6 * cuda.Context.get_device().MULTIPROCESSOR_COUNT, 1)
    
    handle = la.cublashandle()
    
    start = time.time()
    for i in range(0,steps+1):
        Gc = la.dot(G, c, handle = handle)
        launch_kernel(update_func, (256,1,1), grid, 
                      [c, dt*alpha, q, Gc, y, c.size, 1],
                      prepared = True)
        
        if i%x_steps == 0:
            ynorm = normfunc(handle.handle, y.size, y.gpudata, 1)
            print "%d, norm = %.10f, time=%f(ms)" % (i / x_steps, ynorm,
                                                     (time.time()-start)*1000);
            if fullout:
                cuda.memcpy_dtod(
                    int(output.gpudata) + 
                    output.dtype.itemsize*output.ld*int(i/x_steps-1), 
                    c.gpudata, c.dtype.itemsize * c.size)

    #cuda.memcpy_dtod(q.gpudata, c.gpudata, c.dtype.itemsize*c.size)

    if fullout:
        return c,output
    else:
		return c


def get_rnn3_update_func(dtype):
    rnn3_template = """
__global__ void rnn3_update(%(type)s* d_c, double dt_alpha, 
                            %(type)s* d_q, %(type)s* d_Gc,
                            %(type)s* d_ynorm, int size, int groupsize)
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int total_threads = blockDim.x * gridDim.x;
        %(type)s dy;
        for(int i = tid; i < size; i+=total_threads)
	{
        %(type)s tmp = d_Gc[i];
        for(int j = 1; j < groupsize; ++j)
        {
            tmp += d_Gc[i + j * size];
        }
        dy = ( d_q[i] - tmp );
        d_c[i] += dt_alpha * dy;
        d_ynorm[i] = dy;
	}
}
    """
    func = func_compile("rnn3_update", 
                        rnn3_template % {"type": dtype_to_ctype(dtype)})
    func.prepare([np.intp, np.float64, np.intp, np.intp,
                  np.intp, np.int32, np.int32])
    return func





