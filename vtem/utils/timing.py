#!/usr/bin/env python

import time
try:
    import pycuda.driver as cuda
    cudaflag = True
except ImportError:
    cudaflag = False
    
    
    
def func_timer(f):
    """Time the execution of function f. If arguments are specified,
    they are passed to the function being timed."""

    """example: if originally the function is C = np.dot(A,B), then call
    C = func_timer(np.dot)(A,B)  """

    def wrapper(*args, **kwargs):
        start = time.time()
        res = f(*args, **kwargs)
        if cudaflag:
            cuda.Context.synchronize()
        stop = time.time()
        print 'execution time = %.3f ms' % ((stop-start)*1000)
        return res
    return wrapper