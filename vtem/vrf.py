#!/usr/bin/env python

import pycuda.driver as cuda
import numpy as np
from pycuda.tools import dtype_to_ctype
from scikits.cuda.cublas import cublasDgemm, cublasZgemm, cublasCgemm

import utils.parray as parray
from utils.kernel_utils import launch_kernel, func_compile
import pycuda.gpuarray as garray
import utils.linalg as la

class vrf(object):
    """
    Visual receptive field
    """
    def __init__(self, shape, dtype = np.float64, dx = 1.0/16, dy = 1.0/16, scale = 1, domain = None ):
        """
        Parameters
        ----------
        shape : list 
            2-tuple indicating the shape (number of pixels in (height,width) / (y,x) ) of RF
        dtype : type, optional
            numpy dtype to use for RF calculations.
            If not specified , will default to np.float64
        dx : float 
            spatial resolution in x direction, distance between two pixels
        dy: float
            spatial resolution in y direction, distance between two pixels
        scale : float
             scale up the shape by scale and scale down dx,dy by scale
             effectively increase the resoltuion of the RFs
        domain : list
             list of 2, [a,b], specifying the domain to cover RFs
             in default parameter settings
             a in x direction, b in y direction
        
        Notes
        -----
        The coordinate system is given by the following:
        
        ::
        
             row (width / X) major
             
             -----------------------------------------> width
             |   Y
             |   ^-------------------->
             |   |-------------------->
             |   |-------------------->
             |   |--------------------> X
             | 
             |
             v
             height
        
        
        """
        self.shape = shape
        self.dtype = np.dtype(dtype)
        
        self.Pyall = shape[-2] * scale
        self.Pxall = shape[-1] * scale

        #if (self.Pxall/16*16 != self.Pxall) | (self.Pyall/16*16 != self.Pyall):
        #    raise ValueError("please use video with dimension multiples of 16")
        
        self.dx = self.dtype.type(dx).real / scale
        self.dy = self.dtype.type(dy).real / scale
                                
        self.Sxall = self.Pxall * self.dx
        self.Syall = self.Pyall * self.dy
        if domain is None:
            self.Sx = self.Pxall * self.dx
            self.Sy = self.Pyall * self.dy
            
        else:
            self.Sx = float(domain[0])
            self.Sy = float(domain[1])
            
        self.Px = int(np.round(self.Sx / self.dx))
        self.Py = int(np.round(self.Sy / self.dy))
        
        
        
    
    
class vrf_gabor(vrf):
    """ Gabor RFs"""
    
    def __init__(self, shape, dtype = np.float64, dx = 1.0/16, dy = 1.0/16, scale = 1, domain = None):
        """
        Parameters
        ----------
        shape : list 
            2-tuple indicating the shape (number of pixels in (height,width) / (y,x) ) of RF
        dtype : type, optional
            numpy dtype to use for RF calculations.
            If not specified , will default to np.float64
        dx : float 
            spatial resolution in x direction, distance between two pixels
        dy: float
            spatial resolution in y direction, distance between two pixels
        scale : float
             scale up the shape by scale and scale down dx,dy by scale
             effectively increase the resoltuion of the RFs
        domain : list
             list of 2, [a,b], specifying the domain to cover RFs
             in default parameter settings
             a in x direction, b in y direction
        
        Notes
        -----
        The coordinate system is given by the following:
        
        ::
        
             row (width / X) major
             
             -----------------------------------------> width
             |   Y
             |   ^-------------------->
             |   |-------------------->
             |   |-------------------->
             |   |--------------------> X
             | 
             |
             v
             height
        
        
        """
        vrf.__init__(self, shape, dtype = dtype, dx = dx, dy = dy, scale = scale, domain = domain)
        self.func = get_gabor_kernel(dtype)
        
    def get_gabor_parameters(self, dilations = [1, 0, -1, -2, -3], num_rotations = 8, alpha0 = np.float64(2), b0 = np.float64(1.5), b0scalebase=np.sqrt(2), KAPPA = 2.5):
        """
        Create a set of parameters for a family of gabor filters

        Parameters
        ----------
        dilation : list
             a list containing the dilation parameter m
        num_rotations : integer 
             number of rotations of each (dilation, translation_x, translation_y) pair
        alpha0 : np.float64 
             base factor of dilation
        b0 : np.float64 
             base factor of translation
        b0scalebase : np.float64 
             effective translation in each dilation will be
             b0 * (alpha0**dilations[i]) * b0scalebase**(-dilations[i])
        KAPPA : float 
             spatial bandwidth
        """
        
        b0 = np.float64(b0)
        b0scalebase=np.float64(b0scalebase)
        alpha0 = np.float64(alpha0)
        
        
        self.KAPPA = float(KAPPA)
        
        
        
        
        l0 = (num_rotations-1)*np.pi / num_rotations
        
        bi = np.empty(len(dilations),self.dtype)
        trans_x = np.empty(len(dilations),np.int32)
        trans_y = np.empty(len(dilations),np.int32)

        for i in xrange(bi.size):
            bi[i] = b0 * np.power(alpha0,dilations[i]) * np.power(b0scalebase,-dilations[i])
            trans_x[i] = int(np.floor(self.Sx/2/bi[i]))
            trans_y[i] = int(np.floor(self.Sy/2/bi[i]))
            print "(trans_x, trans_y) = (%d, %d)" % (trans_x[i], trans_y[i])

        self.num_neurons = 0
        for i in xrange(len(dilations)):
            self.num_neurons += (trans_x[i]*2 + 1)*(trans_y[i]*2 + 1) * num_rotations * 2

        self.num_neurons = int(self.num_neurons)

        self.h_alpha = np.empty(self.num_neurons, self.dtype)
        self.h_l = np.empty(self.num_neurons, self.dtype)
        self.h_y0 = np.empty(self.num_neurons, self.dtype)
        self.h_x0 = np.empty(self.num_neurons, self.dtype)
        self.h_ab = np.empty(self.num_neurons, np.int32)

        a = 0
        i = 0
        for m in dilations:
            for l in range(num_rotations):
                ll = l * l0
                for k in np.arange(-trans_y[i], trans_y[i]+1, 1):
                    for n in np.arange(-trans_x[i], trans_x[i]+1, 1):
                        self.h_alpha[a] = self.h_alpha[a+1] = alpha0**(m)
                        self.h_l[a] = self.h_l[a+1] = ll
                        self.h_y0[a] = self.h_y0[a+1] = k * bi[i]
                        self.h_x0[a] = self.h_x0[a+1] = n * bi[i]
                        self.h_ab[a] = 0
                        self.h_ab[a+1] = 1
                        a += 2
            i += 1
            
        
        
    def load_parameters(self, num_neurons = None, h_alpha = None, h_l = None, h_x0 = None, h_y0 = None, h_ab = None, KAPPA = 2.5, set = 0):
        """
        Load gabor parameters to GPU
        
        num_neurons, h_alpha, h_l, h_x0, h_y0, h_ab must be specified together
        or not specified at all
        
        Parameters
        ----------
        num_neurons : integer 
              total number of neurons
        h_alpha : ndarray of float64 
              containing dilation parameters alpha = alpha0**(m)
        h_l : ndarray of float64 
              containing rotation parameters l (angles)
        h_x0 : ndarray of float64 
              containing translation parameters x0
        h_y0 : ndarray of float64 
              containing translation parameters y0
        h_ab : ndarray of int32 
              containing 0 or 1, 0 for real part, 1 for imaginary part
              jth gabor filter will be generated according to h_alpha[j], h_l[j], h_n[j], h_k[j] and h_ab[j]

        KAPPA : float 
              spatial bandwidth
        
        set : integer
              0 if self parameters has not been set, 
              1 if they have been set by get_gabor_parameters or other manner)
        
        """
        
        if num_neurons is not None:
            self.num_neurons = num_neurons
            
            if h_alpha is None or h_l is None or h_x0 is None or h_y0 is None or h_ab is None:
                raise ValueError("must specify the gabor parameters")
            else:
                self.h_alpha = h_alpha
                self.h_l = h_l
                self.h_x0 = h_x0
                self.h_y0 = h_y0
                self.h_ab = h_ab
                self.KAPPA = KAPPA
    
        else:
            if not set:
                self.get_gabor_parameters(KAPPA=KAPPA)
        
        self.d_alpha = parray.to_gpu(1.0 / self.h_alpha)
        self.d_l = parray.to_gpu(self.h_l)
        self.d_x0 = parray.to_gpu(self.h_x0)
        self.d_y0 = parray.to_gpu(self.h_y0)
        self.d_ab = parray.to_gpu(self.h_ab)
        
    def filter(self, V):
        """
        Filter a video V
        Must set up parameters of gabor first
        
        Parameters
        ----------
        V : 3D ndarray, with shape (num_frames, Px, Py)
           
        Returns
        -------
        The filtered output by the gabor filters specified in self
        output is a PitchArray with shape (num_neurons, num_frames)
        jth row of which is the output of jth gabor filter

        """
        d_output = parray.empty((self.num_neurons, V.shape[0]), self.dtype)
        d_video = parray.to_gpu(V.reshape(V.shape[0], V.shape[1]*V.shape[2]))
    
        free,total = cuda.mem_get_info()
        self.ONE_TIME_FILTERS = (free / self.dtype.itemsize) * 3/4 / self.Pxall / self.Pyall
        
        handle = la.cublashandle()
        
        for i in np.arange(0,self.num_neurons,self.ONE_TIME_FILTERS):
            Nfilters =  min(self.ONE_TIME_FILTERS, self.num_neurons - i)
            self.generate_visual_receptive_fields(startbias = i, N_filters = Nfilters)
            cublasDgemm(handle.handle, 't','n', V.shape[0], int(Nfilters), self.Pxall*self.Pyall, self.dx*self.dy, d_video.gpudata, d_video.ld, self.gabors.gpudata, self.gabors.ld, 0, int(int(d_output.gpudata)+int(d_output.ld*i*d_output.dtype.itemsize)) , d_output.ld)
        return d_output.T()
    
    
        
    def generate_visual_receptive_fields(self, startbias = 0, N_filters = None, x_start = None, y_start = None):
        """
        Generate a batch of gabor filters from parameters set in self
        
        Parameters
        ----------
        start_bias : integer, optional 
             start the the (start_bias)th filter
        N_filters : integer, optional 
             number of filters to generate
        x_start : float, optional
             indicating the starting degree in x direction
        y_start : float, optional
             indicating the starting degree in y direction
        """
        if N_filters is None:
            N_filters = self.num_neurons - startbias
        
        
        try:
            if N_filters > self.gabors.shape[0]:
                del self.gabors
                self.gabors = parray.empty((N_filters, self.Pyall, self.Pxall), self.dtype)
        except:
            self.gabors = parray.empty((N_filters, self.Pyall, self.Pxall), self.dtype)
        

        if x_start is None:
            x_start = - self.Sxall/ 2

        if y_start is None:
            y_start = -self.Syall/2
        
        BLOCK_SIZE = 16
        
        launch_kernel(self.func, (BLOCK_SIZE, BLOCK_SIZE, 1), (((self.Pxall-1)/BLOCK_SIZE+1) * ((self.Pyall-1)/BLOCK_SIZE+1), int(N_filters)), [self.gabors, self.gabors.ld, [self.d_alpha, startbias], [self.d_l, startbias], [self.d_x0, startbias], [self.d_y0, startbias], [self.d_ab, startbias],  self.Pxall, self.Pyall, self.Sxall, self.Syall, x_start, y_start, self.KAPPA])

    def compute_Ds(self, Mx, My):
        """
        Returns the dirichlet coefficients of all gabor filters with order Mx, My
        in the format of PitchArray with shape (num_neurons, 2*Mx+1, 2*My+1)
        """
        
        import scikits.cuda.cufft as cufft
        d_Ds = parray.empty((self.num_neurons, 2*My+1, 2*Mx+1), self.dtype)
        ONE_TIME_FILTER = min(1024**3 / (self.Pxall * self.Pyall * d_Ds.dtype.itemsize) / 2, self.num_neurons)
        
        n = np.asarray((self.Pyall, self.Pxall) ,np.int32)
        
        if self.dtype == np.complex128:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecZ2Z
        else:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecC2C
        
        fft2Dsfun = get_fft2Ds_kernel(dtype = self.dtype)
        
        
        for i in range(0, self.num_neurons, ONE_TIME_FILTER):
            N_filters = min(ONE_TIME_FILTER, self.num_neurons - i)
            self.generate_visual_receptive_fields(startbias = i, N_filters = N_filters)
            
            
            if N_filters < ONE_TIME_FILTER:
                cufft.cufftDestroy(plan)
                if self.dtype == np.complex128:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, N_filters)
                    
                else:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, N_filters)
                
            
            
            #be careful with the side-by-side constraint
            fftfunc(plan, int(self.gabors.gpudata), int(self.gabors.gpudata), cufft.CUFFT_FORWARD)
            
            launch_kernel(fft2Dsfun, (256, 1, 1), (Mx*2+1, My * 2+1), [[d_Ds, i * d_Ds.ld], self.gabors, Mx, My, self.Pxall, self.Pyall, N_filters, d_Ds.ld, self.dx*self.dy]);
        
        cufft.cufftDestroy(plan)
        
        return d_Ds
        
        
    def compute_dirich_space(self, d_Ds, Mx, My, Px, Py, Sx, Sy, Wx, Wy, x_start = None, y_start = None):
        """
        Compute the spatial reconstruction functions
        
        Parameters
        ----------

        d_Ds : PitchArray 
             containing dirichlet coefficient most possibly created by compute_Ds
        Mx : integer
             Order in the x dimension
        My : integer
             Order in the y dimension
        Px : integer
             number of pixels in reconstruction functions in the x dimension 
        Py : integer
             number of pixels in reconstruction functions in the y dimension
        Sx : float
             spatial domain in degree of reconstruction functions
             in x direction
        Sy : float 
             spatial domain in degree of reconstruction functions
             in y direction
        Wx : float
             spatial bandwidth in x direction 
        Wy : float
             spatial bandwidth in y direction
        x_start : float 
             indicating the starting degree in x direction
        y_start : float
             indicating the starting degree in y direction

        output: PitchArray with shape (num_neurons, Px, Py)
        """
        
        if self.dtype == np.complex128:
            typef = np.dtype(np.float64)
        else:
            typef = np.dtype(np.float32)
                
        dirich_fun = get_dirich_space_kernel(self.dtype, typef)
                
        d_dirich = parray.empty((self.num_neurons, Py, Px),typef)
            
        if x_start is None:
            x_start = - Sx/ 2

        if y_start is None:
            y_start = - Sy/2
		
        BLOCKSIZE = 16
        launch_kernel(dirich_fun,(BLOCKSIZE, BLOCKSIZE, 1), (((Px-1) / BLOCKSIZE+1) * ((Py-1) / BLOCKSIZE+1), self.num_neurons), [d_dirich, d_dirich.ld, d_Ds, d_Ds.ld, Px, Py, Mx, My, Sx, Sy, x_start, y_start, Wx / Mx, Wy / My], shared = d_Ds.dtype.itemsize * (2*Mx+1), timed = "dirich")
		
        return d_dirich

            
            
    def compute_dirich_space_fft(self, d_Ds, Mx, My, Px, Py, Sx, Sy, Wx, Wy):
        import scikits.cuda.cufft as cufft
        
        dx = Sx / Px
        dy = Sy / Py
        
        Px1 = int(np.round(self.Sx / dx))
        Py1 = int(np.round(self.Sy / dy))
        
        
        if self.dtype == np.complex128:
            typef = np.dtype(np.float64)
        else:
            typef = np.dtype(np.float32)
        
        d_dirich = parray.empty((self.num_neurons, Py, Px),typef)
        
        freemem,totalmem = cuda.mem_get_info()
        
        ONE_TIME_FILTER = int(min(freemem / (Px1 * Py1 * d_Ds.dtype.itemsize) / 4, self.num_neurons))
        
        
        n = np.asarray((Py1, Px1) ,np.int32)
        
        
        if self.dtype == np.complex128:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecZ2Z
        else:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecC2C
        
        
        Ds2fftfun = get_Ds2fft_kernel(self.dtype)
        d_filter_complex = parray.empty((ONE_TIME_FILTER, Px1*Py1), self.dtype)
        filter2recfun = get_filter2rec_kernel(self.dtype)
        
        for i in range(0, self.num_neurons, ONE_TIME_FILTER):
            N_filters = min(ONE_TIME_FILTER, self.num_neurons - i)
            d_filter_complex.fill(0)
            
            launch_kernel(Ds2fftfun, (256,1,1), (Mx*2+1, My*2+1), [[d_Ds,i * d_Ds.ld], d_Ds.ld, d_filter_complex, d_filter_complex.ld, Mx, My, Px1, Py1, N_filters])
            
            if N_filters < ONE_TIME_FILTER:
                cufft.cufftDestroy(plan)
                if self.dtype == np.complex128:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, N_filters)
                
                else:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, N_filters)
            
            
            
            #be careful with the side-by-side constraint
            fftfunc(plan, int(d_filter_complex.gpudata), int(d_filter_complex.gpudata), cufft.CUFFT_INVERSE)
            
            
            BLOCK_SIZE = 16
            launch_kernel(filter2recfun, (BLOCK_SIZE,BLOCK_SIZE,1), (((Px-1) / BLOCK_SIZE + 1)* ((Py-1) / BLOCK_SIZE+1), N_filters), [[d_dirich,i * d_dirich.ld],d_dirich.ld, d_filter_complex, d_filter_complex.ld, N_filters, Px, Py, Px1, Py1])
        
        cufft.cufftDestroy(plan)
        
        return d_dirich
    
    
    def compute_Dsw(self, d_Ds, Mx, My, h_norm):
        """
        Compute the weighting matrix of the "correlation" between each two RFs
        
        Parameters
        ----------
        d_Ds : PitchArray 
            containing dirichlet coefficient most possibly created by compute_Ds
        Mx : integer
            order in the x dimension
        My : integer
            order in the y dimension
        
        Returns
        -------
        PitchArray with shape (num_neurons, num_neurons)
        """
        
        if self.dtype == np.complex128:
            gemm = cublasZgemm
        else:
            gemm = cublasCgemm
        
        d_weight = parray.empty((self.num_neurons, self.num_neurons), self.dtype)
        
        handle = la.cublashandle()
        
        gemm(handle.handle, 'c', 'n', self.num_neurons, self.num_neurons, (2*Mx+1)*(2*My+1), 1.0, d_Ds.gpudata, d_Ds.ld, d_Ds.gpudata, d_Ds.ld, 0, d_weight.gpudata, d_weight.ld);
        d_Dsw = d_weight.real()
        
        norm_func = get_put_norm_kernel(d_Dsw.dtype)
        launch_kernel(norm_func, (256, 1, 1), (d_Dsw.shape[0],1), [d_Dsw, parray.to_gpu(h_norm.astype(np.float64)), d_Dsw.ld])
        
        
        return d_Dsw

        

def get_put_norm_kernel(dtype):
    template = """
        __global__ void put_norm(%(type)s* d_SWeight, double* d_norm, int ld)
        {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        int NUM_NEURONS = gridDim.x;
        
        double norm1 = d_norm[bid];
        double norm2;
        
        for(int i = tid; i < NUM_NEURONS; i += blockDim.x)
        {
		norm2 = d_norm[i];
		d_SWeight[i + bid * ld] *= norm1 * norm2;
        }
        }
        """
    func = func_compile("put_norm", template  % {"type": dtype_to_ctype(dtype)})
    return func

def get_put_norm_unaligned_kernel(dtype):
    template = """
        __global__ void
        put_norm_unaligned(%(type)s* d_SWeight, int ld, double* d_norm, int* neuron_ind1, int* neuron_ind2, int num_neurons1)
        {
        int tid = threadIdx.x;
        int bid = blockIdx.x;
        
        int ind1 = neuron_ind1[bid];
        double norm1 = d_norm[ind1];
        int ind2;
        double norm2;
        
        for(int i = tid; i < num_neurons1; i += blockDim.x)
        {
        ind2 = neuron_ind2[i];
		norm2 = d_norm[ind2];
		d_SWeight[i + bid * ld] *= norm1 * norm2;
        }
        }
        """
    func = func_compile("put_norm_unaligned", template  % {"type": dtype_to_ctype(dtype)})
    return func

def get_Ds2fft_kernel(dtype):
    template = """
        #include <pycuda/pycuda-complex.hpp>
        __global__ void
        Ds2fft_kernel(%(type)s* d_Ds, int Ds_ld, %(type)s* filter, int filter_ld, int Mx, int My, int Px, int Py, int N_filters)
        {
        const int tid = threadIdx.x;
        const int mx = blockIdx.x - Mx;
        const int my = blockIdx.y - My;
        
        int fftind;
        double sgndxdy;
        
        if(mx < 0 && my <0)
        {
        fftind = Px * (Py + my) + (Px + mx);
        }else if(my < 0 && mx >= 0)
        {
        fftind = mx + (Px * (Py + my));
        }else if(mx >= 0 && my >=0)
        {
        fftind = my*Px + mx;
        }else//(my > 0 && mx < 0)
        {
        fftind = my * Px + (Px + mx);
        }  
        
        if( (mx+my)%%2 ==0)
        {
        sgndxdy = 1;
        }else
        {
        sgndxdy = -1;
        }
        
        %(type)s tmp;
        
        
        for(int i = tid; i < N_filters; i+=blockDim.x)
        {
        tmp = d_Ds[i * Ds_ld + (My-my)*(2*Mx+1) + (mx+Mx)];
        filter[i * filter_ld + fftind] = %(type)s(sgndxdy * pycuda::real(tmp),sgndxdy * pycuda::imag(tmp));
        }
        }
        
        """
    func = func_compile("Ds2fft_kernel", template % {"type": dtype_to_ctype(dtype)})
    return func


def get_filter2rec_kernel(dtype):
    if dtype == np.complex128:
        dtypef = np.float64
    else:
        dtypef = np.float32
    
    template = """
        #include <pycuda/pycuda-complex.hpp>
        #define BLOCK_SIZE 16    
        
        __global__ void
        filter2rec_kernel(%(typef)s* d_dirich_space, int dirich_ld, %(type)s* d_filter_complex, int filter_ld, int N_filters, int Px, int Py, int Px1, int Py1)
        {
        unsigned int tid_x; tid_x = threadIdx.x;
        unsigned int tid_y; tid_y = threadIdx.y;
        
        int xld = (Px-1)/BLOCK_SIZE + 1;
        unsigned int bid_x = blockIdx.x %% xld; 
        unsigned int bid_y = blockIdx.x / xld; 
        
        unsigned int filter_id = blockIdx.y;
        
        unsigned int dim_x = blockDim.x;
        unsigned int dim_y = blockDim.y;
        
        int pix_x = dim_x * bid_x + tid_x;
        int pix_y = dim_y * bid_y + tid_y;
        
        int xdrift = (Px1 - Px)/2;
        int ydrift = (Py1 - Py)/2;
        
        int input_ind = (pix_y + ydrift) * Px1 + pix_x + xdrift + filter_id * filter_ld;
        int output_ind = (pix_y) * Px + (pix_x) + filter_id * dirich_ld;
        
        if(pix_x < Px && pix_y < Py)
        {
        d_dirich_space[output_ind] = pycuda::real(d_filter_complex[input_ind]);
        }
        
        }
        """
    func = func_compile("filter2rec_kernel", template % {"type": dtype_to_ctype(dtype), "typef": dtype_to_ctype(dtypef)})
    return func



class vrf_cs(vrf):
    """ Centre Surround RFs"""
    def __init__(self, shape, dtype = np.float64, dx = 1.0/16, dy = 1.0/16, scale = 1, domain = None):
        """
        Parameters
        ----------
        shape : list 
            2-tuple indicating the shape (number of pixels in (height,width) / (y,x) ) of RF
        dtype : type, optional
            numpy dtype to use for RF calculations.
            If not specified , will default to np.float64
        dx : float 
            spatial resolution in x direction, distance between two pixels
        dy: float
            spatial resolution in y direction, distance between two pixels
        scale : float
             scale up the shape by scale and scale down dx,dy by scale
             effectively increase the resoltuion of the RFs
        domain : list
             list of 2, [a,b], specifying the domain to cover RFs
             in default parameter settings
             a in x direction, b in y direction
        
        Notes
        -----
        The coordinate system is given by the following:
        
        ::
        
             row (width / X) major
             
             -----------------------------------------> width
             |   Y
             |   ^-------------------->
             |   |-------------------->
             |   |-------------------->
             |   |--------------------> X
             | 
             |
             v
             height
        """
        vrf.__init__(self, shape, dtype = dtype, dx = dx, dy = dy, scale = scale, domain = domain)
        self.func = get_cs_kernel(dtype)
    
    def get_cs_parameters(self, dilations = [1, 0, -1, -2, -3], alpha0 = np.float64(2), b0 = np.float64(1.0), b0scalebase= np.float(1.0), sigma_center = 0.5, sigma_surround = 0.8):
        """
        Create a set of parameters for a family of Centre Surround filters

        Parameters
        -----------
        dilation : list, optional
              a list containing the dilation parameter m
        num_rotations : list, optional
              number of rotations of each (dilation, translation_x, translation_y) pair
        alpha0 : np.float64, optional 
              base factor of dilation
        b0 : np.float64, optional
              base factor of translation
        b0scalebase : np.float64, optional 
              effective translation in each dilation will be
              b0 * (alpha0**dilations[i]) * b0scalebase**(-dilations[i])
        sigma_center : float, optional
              standard deviation of the center
        sigma_surround : float, optional
              standard deviation of the surround
              
        """
        
        b0 = np.float64(b0)
        b0scalebase=np.float64(b0scalebase)
        alpha0 = np.float64(alpha0)
        
        sigma_center = float(sigma_center)
        sigma_surround = float(sigma_surround)
        
        self.sigma_center = sigma_center
        self.sigma_surround = sigma_surround
        
        
        bi = np.empty(len(dilations),self.dtype)
        trans_x = np.empty(len(dilations),np.int32)
        trans_y = np.empty(len(dilations),np.int32)

        for i in xrange(bi.size):
            bi[i] = b0 * np.power(alpha0,dilations[i]) * np.power(b0scalebase,-dilations[i])
            trans_x[i] = int(np.floor(self.Sx/2/bi[i]))
            trans_y[i] = int(np.floor(self.Sy/2/bi[i]))
            print "(trans_x, trans_y) = (%d, %d)" % (trans_x[i], trans_y[i])

        self.num_neurons = 0
        for i in xrange(len(dilations)):
            self.num_neurons += (trans_x[i]*2 + 1)*(trans_y[i]*2 + 1)

        self.num_neurons = int(self.num_neurons)

        self.h_alpha = np.empty(self.num_neurons, self.dtype)
        self.h_y0 = np.empty(self.num_neurons, self.dtype)
        self.h_x0 = np.empty(self.num_neurons, self.dtype)

        a = 0
        i = 0
        for m in dilations:
            for k in np.arange(-trans_y[i], trans_y[i]+1, 1):
                for n in np.arange(-trans_x[i], trans_x[i]+1, 1):
                    self.h_alpha[a] = alpha0**(m)
                    self.h_y0[a] = k * bi[i]
                    self.h_x0[a] = n * bi[i]
                    a += 1
            i += 1
            
        
        
    def load_parameters(self, num_neurons = None, h_alpha = None, h_x0 = None, h_y0 = None, set = 0, sigma_center = 0.5, sigma_surround = 0.8):
        """
        Load Centre Surround parameters to GPU
        
        num_neurons, h_alpha, h_l, h_x0, h_y0, h_ab must be specified together
        or unspecified together
        
        Parameters
        ----------
        num_neurons : integer 
              total number of neurons
        h_alpha : ndarray of float64 
              containing dilation parameters alpha = alpha0**(m)
        h_l : ndarray of float64 
              containing rotation parameters l (angles)
        h_x0 : ndarray of float64 
              containing translation parameters x0
        h_y0 : ndarray of float64 
              containing translation parameters y0
        h_ab : ndarray of int32 
              containing 0 or 1, 0 for real part, 1 for imaginary part
              jth gabor filter will be generated according to h_alpha[j], h_l[j], h_n[j], h_k[j] and h_ab[j]
        set : integer 
              0 if self parameters has not been set, 1 if they have been set by get_gabor_parameters or other manner)
        sigma_center : float, optional
              standard deviation of the center
        sigma_surround : float, optional
              standard deviation of the surround
        
        """
        
        if num_neurons is not None:
            self.num_neurons = num_neurons
            
            if h_alpha is None or h_x0 is None or h_y0 is None:
                raise ValueError("must specify the gabor parameters")
            else:
                self.h_alpha = h_alpha
                self.h_x0 = h_x0
                self.h_y0 = h_y0
                self.sigma_center = float(sigma_center)
                self.sigma_surround = float(sigma_surround)
    
        else:
            if not set:
                self.get_cs_parameters()
        
        self.d_alpha = parray.to_gpu(1.0 / self.h_alpha)
        self.d_x0 = parray.to_gpu(self.h_x0)
        self.d_y0 = parray.to_gpu(self.h_y0)
        
    def filter(self, V):
        """
        Filter a video V
        Must set up parameters of CS RF first
        
        Parameters
        ----------
        V : 3D ndarray, with shape (num_frames, Px, Py)
           
        Returns
        -------
        the filtered output by the gabor filters specified in self
        output is a PitchArray with shape (num_neurons, num_frames),
        jth row of which is the output of jth gabor filter

        """
        d_output = parray.empty((self.num_neurons, V.shape[0]), self.dtype)
        d_video = parray.to_gpu(V.reshape(V.shape[0], V.shape[1]*V.shape[2]))
    
        free,total = cuda.mem_get_info()
        self.ONE_TIME_FILTERS = (free / self.dtype.itemsize) * 3/4 / self.Pxall / self.Pyall
        
        handle = la.cublashandle()
        for i in np.arange(0,self.num_neurons,self.ONE_TIME_FILTERS):
            Nfilters =  min(self.ONE_TIME_FILTERS, self.num_neurons - i)
            self.generate_visual_receptive_fields(startbias = i, N_filters = Nfilters)
            cublasDgemm(handle.handle, 't','n', V.shape[0], int(Nfilters), self.Pxall*self.Pyall, self.dx*self.dy, d_video.gpudata, d_video.ld, self.filters.gpudata, self.filters.ld, 0, int(int(d_output.gpudata)+int(d_output.ld*i*d_output.dtype.itemsize)) , d_output.ld)
        return d_output.T()
    
    
        
    def generate_visual_receptive_fields(self, startbias = 0, N_filters = None, x_start = None, y_start = None):
        """
        Generate a batch of centre surround filters from parameters set in self
        
        Parameters
        ----------
        start_bias : integer, optional
            start the the (start_bias)th filter
        N_filters : integer, optional 
            generate N_filters filters
        x_start : float
            indicating the starting degree in x direction
        y_start : float 
            indicating the starting degree in y direction

        
        """
        if N_filters is None:
            N_filters = self.num_neurons - startbias
        
        
        try:
            if N_filters > self.filters.shape[0]:
                del self.filters
                self.filters = parray.empty((N_filters, self.Pyall, self.Pxall), self.dtype)
        except:
            self.filters = parray.empty((N_filters, self.Pyall, self.Pxall), self.dtype)
        

        if x_start is None:
            x_start = - self.Sxall/ 2

        if y_start is None:
            y_start = -self.Syall/2
        
        BLOCK_SIZE = 16
        
        launch_kernel(self.func, (BLOCK_SIZE, BLOCK_SIZE, 1), (((self.Pxall-1)/BLOCK_SIZE+1) * ((self.Pyall-1)/BLOCK_SIZE+1), int(N_filters)), [self.filters, self.filters.ld, [self.d_alpha, startbias], [self.d_x0, startbias], [self.d_y0, startbias], self.Pxall, self.Pyall, self.Sxall, self.Syall, x_start, y_start, self.sigma_center**2, self.sigma_surround**2])

    def compute_Ds(self, Mx, My):
        """
        Parameters
        ----------
        Mx : integer
            Order in the x dimension
        My : integer
            Order in the y dimension
        
        Returns
        -------
        The dirichlet coefficients of all gabor filters with order Mx, My
        in the format of PitchArray with shape (num_neurons, 2*Mx+1, 2*My+1)
        """
        
        import scikits.cuda.cufft as cufft
        d_Ds = parray.empty((self.num_neurons, 2*My+1, 2*Mx+1), self.dtype)
        ONE_TIME_FILTER = min(1024**3 / (self.Px * self.Py * d_Ds.dtype.itemsize) / 2, self.num_neurons)
        
        n = np.asarray((self.Py, self.Px) ,np.int32)
        
        if self.dtype == np.complex128:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecZ2Z
        else:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecC2C
        
        fft2Dsfun = get_fft2Ds_kernel(dtype = self.dtype)
        
        
        for i in range(0, self.num_neurons, ONE_TIME_FILTER):
            N_filters = min(ONE_TIME_FILTER, self.num_neurons - i)
            self.generate_visual_receptive_fields(startbias = i, N_filters = N_filters)
            
            
            if N_filters < ONE_TIME_FILTER:
                cufft.cufftDestroy(plan)
                if self.dtype == np.complex128:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, N_filters)
                    
                else:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, N_filters)
                
            
            
            #be careful with the side-by-side constraint
            fftfunc(plan, int(self.filters.gpudata), int(self.filters.gpudata), cufft.CUFFT_FORWARD)
            
            launch_kernel(fft2Dsfun, (256, 1, 1), (Mx*2+1, My * 2+1), [[d_Ds, i * d_Ds.ld], self.filters, Mx, My, self.Px, self.Py, N_filters, d_Ds.ld, self.dx*self.dy]);
        
        cufft.cufftDestroy(plan)
        
        return d_Ds
        
        
    def compute_dirich_space(self, d_Ds, Mx, My, Px, Py, Sx, Sy, Wx, Wy, x_start = None, y_start = None):
        """
        Compute the spatial reconstruction functions
        
        Parameters
        ----------
        d_Ds : PitchArray 
             containing dirichlet coefficient most possibly created by compute_Ds
        Mx : integer
             Order in the x direction
        My : integer
             Order in the y direction
        Px : integer
             number of pixels in reconstruction functions in x direction
        Py : integer
             number of pixels in reconstruction functions in y direction
        Sx : float
             spatial domain in degree of reconstruction functions in x direction
        Sy : float
             spatial domain in degree of reconstruction functions in y direction
        Wx : float
             spatial bandwidth in x direction
        Wy : float
             spatial bandwidth in y direction
        x_start : float, optional
             indicating the starting degree in x direction
        y_start : float, optional
             indicating the starting degree in y direction

        output: PitchArray with shape (num_neurons, Px, Py)
        """
        
        if self.dtype == np.complex128:
            typef = np.dtype(np.float64)
        else:
            typef = np.dtype(np.float32)
                
        dirich_fun = get_dirich_space_kernel(self.dtype, typef)
                
        d_dirich = parray.empty((self.num_neurons, Py, Px),typef)
            
        if x_start is None:
            x_start = - Sx/ 2

        if y_start is None:
            y_start = - Sy/2
		
        BLOCKSIZE = 16
        launch_kernel(dirich_fun,(BLOCKSIZE, BLOCKSIZE, 1), (((Px-1) / BLOCKSIZE+1) * ((Py-1) / BLOCKSIZE+1), self.num_neurons), [d_dirich, d_dirich.ld, d_Ds, d_Ds.ld, Px, Py, Mx, My, Sx, Sy, x_start, y_start, Wx / Mx, Wy / My], shared = d_Ds.dtype.itemsize * (2*Mx+1), timed = "dirich")
		
        return d_dirich

    def compute_dirich_space_fft(self, d_Ds, Mx, My, Px, Py, Sx, Sy, Wx, Wy):
        import scikits.cuda.cufft as cufft
        
        dx = Sx / Px
        dy = Sy / Py
        
        Px1 = int(np.round(self.Sx / dx))
        Py1 = int(np.round(self.Sy / dy))
        
        
        if self.dtype == np.complex128:
            typef = np.dtype(np.float64)
        else:
            typef = np.dtype(np.float32)
        
        d_dirich = parray.empty((self.num_neurons, Py, Px),typef)
        
        freemem,totalmem = cuda.mem_get_info()
        
        ONE_TIME_FILTER = int(min(freemem / (Px1 * Py1 * d_Ds.dtype.itemsize) / 4, self.num_neurons))
        
        
        n = np.asarray((Py1, Px1) ,np.int32)
        
        
        if self.dtype == np.complex128:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecZ2Z
        else:
            plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, ONE_TIME_FILTER)
            fftfunc = cufft.cufftExecC2C
        
        
        Ds2fftfun = get_Ds2fft_kernel(self.dtype)
        d_filter_complex = parray.empty((ONE_TIME_FILTER, Px1*Py1), self.dtype)
        filter2recfun = get_filter2rec_kernel(self.dtype)
        
        for i in range(0, self.num_neurons, ONE_TIME_FILTER):
            N_filters = min(ONE_TIME_FILTER, self.num_neurons - i)
            d_filter_complex.fill(0)
            
            launch_kernel(Ds2fftfun, (256,1,1), (Mx*2+1, My*2+1), [[d_Ds,i * d_Ds.ld], d_Ds.ld, d_filter_complex, d_filter_complex.ld, Mx, My, Px1, Py1, N_filters])
            
            if N_filters < ONE_TIME_FILTER:
                cufft.cufftDestroy(plan)
                if self.dtype == np.complex128:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_Z2Z, N_filters)
                
                else:
                    plan = cufft.cufftPlanMany(2, n.ctypes.data, None, 1, 0, None, 1, 0, cufft.CUFFT_C2C, N_filters)
            
            
            
            #be careful with the side-by-side constraint
            fftfunc(plan, int(d_filter_complex.gpudata), int(d_filter_complex.gpudata), cufft.CUFFT_INVERSE)
            
            
            BLOCK_SIZE = 16
            launch_kernel(filter2recfun, (BLOCK_SIZE,BLOCK_SIZE,1), (((Px-1) / BLOCK_SIZE + 1)* ((Py-1) / BLOCK_SIZE+1), N_filters), [[d_dirich,i * d_dirich.ld],d_dirich.ld, d_filter_complex, d_filter_complex.ld, N_filters, Px, Py, Px1, Py1])
        
        cufft.cufftDestroy(plan)
        
        return d_dirich
    
    def compute_Dsw(self, d_Ds, Mx, My, h_norm):
        """
        Compute the weighting matrix of the "correlation" between each two RFs
        
        Parameters
        ----------
        d_Ds : PitchArray
             containing dirichlet coefficient most possibly created by compute_Ds
        Mx : integer
             order in the x direction
        My : integer
             order in the y direction
 
        Returns
        -------
        PitchArray with shape (num_neurons, num_neurons)
        """
        
        if self.dtype == np.complex128:
            gemm = cublasZgemm
        else:
            gemm = cublasCgemm

        d_weight = parray.empty((self.num_neurons, self.num_neurons), self.dtype)
        handle = la.cublashandle()
        gemm(handle.handle, 'c', 'n', self.num_neurons, self.num_neurons, (2*Mx+1)*(2*My+1), 1.0, d_Ds.gpudata, d_Ds.ld, d_Ds.gpudata, d_Ds.ld, 0, d_weight.gpudata, d_weight.ld);
        d_Dsw = d_weight.real()
        
        norm_func = get_put_norm_kernel(d_Dsw.dtype)
        launch_kernel(norm_func, (256, 1, 1), (d_Dsw.shape[0],1), [d_Dsw, parray.to_gpu(h_norm.astype(np.float64)), d_Dsw.ld])
        return d_Dsw
 

def get_cs_kernel(dtype=np.dtype(np.float64)):
    gabor_template = """
    #include <pycuda/pycuda-complex.hpp>
    #define PI 3.141592653589793238462643383279
    #define BLOCK_SIZE 16
    
    __global__ void
cs_Kernel(%(type)s* g_filter, int filter_ld, double* g_m, double* g_x0, double* g_y0, int Px, int Py, double Sx, double Sy, double x_start, double y_start, double sigma_c_square, double sigma_s_square) 
{
	unsigned int tid_x = threadIdx.x;
	unsigned int tid_y = threadIdx.y;	

        int xld = (Px-1)/BLOCK_SIZE + 1;
	unsigned int bid_x = blockIdx.x %% xld; 
	unsigned int bid_y = blockIdx.x / xld; 

	unsigned int filter_id = blockIdx.y; //actually y index;

	unsigned int dim_x = blockDim.x;
	unsigned int dim_y = blockDim.y;


	__shared__ double alpha;
	__shared__ double x0;
	__shared__ double y0;
	
    if(tid_y == 0)
    {
        if(tid_x == 0)
        {
            alpha = g_m[filter_id];
        }else if(tid_x == 1)
        {
            x0 = g_x0[filter_id];
        }else if(tid_x == 2)
        {
            y0 = g_y0[filter_id];
        }
	}
	__syncthreads();
	
	double x, y;
	int pix_x = dim_x * bid_x + tid_x;
	int pix_y = dim_y * bid_y + tid_y;
	
	
	// degree per pixel
	double dxdy;
	
	dxdy = (double)(Sx / (Px));
	x =  (double)(pix_x) * dxdy + x_start;
	dxdy = (double)(Sy / (Py));
	y =  - ((double)(pix_y) * dxdy + y_start);
	

	x = alpha * (x - x0);
	y = alpha * (y - y0);

	double XY = -(x*x + y*y);
    
	double gb = (alpha) * ( exp ( XY / (2 * sigma_c_square)) / (sigma_c_square) - 0.9 * exp ( XY / (2 * sigma_s_square)) / (sigma_s_square));
	
    

	if(pix_x < Px && pix_y < Py)
        {
            int output_ind = (pix_y) * Px + (pix_x) + filter_id * filter_ld;
            g_filter[output_ind] = gb / (2*PI);
	}
}
    """
    
    func = func_compile("cs_Kernel", gabor_template % {"type": dtype_to_ctype(dtype)}, options=["--ptxas-options=-v --maxrregcount=32"])
    return func



def get_gabor_kernel(dtype=np.dtype(np.float64)):
    gabor_template = """
    #include <pycuda/pycuda-complex.hpp>
    #define PI 3.141592653589793238462643383279
    #define BLOCK_SIZE 16
    
    __global__ void
gabor_Kernel(%(type)s* g_filter, int filter_ld, double* g_m, double* g_l, double* g_x0, double* g_y0, int* g_ab, int Px, int Py, double Sx, double Sy, double x_start, double y_start, double KAPPA) 
{
	unsigned int tid_x = threadIdx.x;
	unsigned int tid_y = threadIdx.y;	

        int xld = (Px-1)/BLOCK_SIZE + 1;
	unsigned int bid_x = blockIdx.x %% xld; 
	unsigned int bid_y = blockIdx.x / xld; 

	unsigned int filter_id = blockIdx.y; //actually y index;

	unsigned int dim_x = blockDim.x;
	unsigned int dim_y = blockDim.y;


	__shared__ double alpha;
	__shared__ double theta;
	__shared__ double x0;
	__shared__ double y0;
	__shared__ int sc;
	
    if(tid_y == 0)
    {
        if(tid_x == 0)
        {
            alpha = g_m[filter_id];
        }else if(tid_x == 1)
        {
            theta = g_l[filter_id];
        }else if(tid_x == 2)
        {
            x0 = g_x0[filter_id];
        }else if(tid_x == 3)
        {
            y0 = g_y0[filter_id];
        }else if(tid_x == 4)
        {
            sc = g_ab[filter_id];
        }
	}
	__syncthreads();
	
	double x, y;
	int pix_x = dim_x * bid_x + tid_x;
	int pix_y = dim_y * bid_y + tid_y;
	
	
	// degree per pixel
	double dxdy;
	
	dxdy = (double)(Sx / (Px));
	x =  (double)(pix_x) * dxdy + x_start;
	dxdy = (double)(Sy / (Py));
	y =  - ((double)(pix_y) * dxdy + y_start);
	

	x = alpha * (x - x0);
	y = alpha * (y - y0);

	double sint, cost;

	sincos(theta, &sint, &cost);

	double X = x * cost + y * sint;
	double Y = -x * sint + y * cost;
	double first_part = alpha * (1/sqrt(2 * PI)) * exp (- ( (4 * X * X)  + (Y * Y)) / 8);
	
		
	sincos(KAPPA * X, &sint, &cost);
	double gb;
	if(sc == 0)
	{
		gb = first_part * cost;
	}else
	{
		gb = first_part * sint;
	}

	if(pix_x < Px && pix_y < Py)
        {
            int output_ind = (pix_y) * Px + (pix_x) + filter_id * filter_ld;
            g_filter[output_ind] = gb;
	}
}
    """
    
    func = func_compile("gabor_Kernel", gabor_template % {"type": dtype_to_ctype(dtype)}, options=["--ptxas-options=-v --maxrregcount=32"])
    return func

def get_fft2Ds_kernel(dtype=np.dtype(np.complex128)):
    fft2Ds_template = """
    #include <pycuda/pycuda-complex.hpp>
    
    __global__ void
fft2Ds_kernel(%(type)s* d_Ds, %(type)s* d_gabor_fft, int Mx, int My, int LPx, int LPy, int N_filters, int pitch, double dxdy)
{
	const int tid = threadIdx.x;
	const int mx = blockIdx.x - Mx;
	const int my = blockIdx.y - My;

	int fftind;
    double sgndxdy;
	
	if(mx < 0 && my <0)
	{
		fftind = LPx * (LPy + my) + (LPx + mx);
	}else if(my < 0 && mx >= 0)
	{
		fftind = mx + (LPx * (LPy + my));
	}else if(mx >= 0 && my >=0)
	{
		fftind = my*LPx + mx;
	}else//(my >= 0 && mx < 0)
	{
		fftind = my * LPx + (LPx + mx);
	}

	if( (mx+my)%%2 ==0)
	{
		sgndxdy = dxdy;
	}else
	{
		sgndxdy = -dxdy;
	}

	%(type)s tmp;
    
    if(mx*mx * My * My + my*my * Mx * Mx <= Mx*Mx*My*My )
    {
        for(int i = tid; i < N_filters; i+=blockDim.x)
        {
            tmp = d_gabor_fft[i * LPx * LPy + fftind];

            d_Ds[i * pitch + (My-my)*(2*Mx+1) + (mx+Mx)] = %(type)s(sgndxdy * pycuda::real(tmp), sgndxdy * pycuda::imag(tmp));
        }
    }else
    {
        for(int i = tid; i < N_filters; i+=blockDim.x)
        {
            d_Ds[i * pitch + (My - my)*(2*Mx+1) + (mx+Mx)] = 0;
        }
    }

}
    """
    func = func_compile("fft2Ds_kernel", fft2Ds_template % {"type": dtype_to_ctype(dtype)})
    return func


def get_dirich_space_kernel(dtype = np.dtype(np.complex128), typef = np.dtype(np.float64)):
    dirich_template = """
    #include <pycuda/pycuda-complex.hpp>
    #define BLOCK_SIZE 16

    
    __device__ pycuda::complex<float> Iexpf(const float x)
{
	float s,c;
	sincosf(x, &s, &c);
	return pycuda::complex<float>( c, s);
}

__device__ pycuda::complex<double> Iexp(const double x)
{
	double s,c;
	sincos(x, &s, &c);
	return pycuda::complex<double>( c, s);
}
	
	__global__ void
psi_dirich_space_Kernel(%(typef)s* dirich_space, int dirich_ld, %(type)s* Ds, int Ds_ld, int Px, int Py, int Mx, int My, double Sx, double Sy, double x_start, double y_start, double WMx, double WMy)
{
	unsigned int filter_id = blockIdx.y; //actually y index;


	extern __shared__ %(type)s s_Ds[];

	unsigned int tid_x = threadIdx.x;
	unsigned int tid_y = threadIdx.y;

	int xld = (Px-1)/BLOCK_SIZE+1;
	
	unsigned int bid_x = blockIdx.x %% xld; 
	unsigned int bid_y = blockIdx.x / xld; 

	unsigned int dim_x = blockDim.x;
	unsigned int dim_y = blockDim.y;


	double x, y;
	int pix_x = dim_x * bid_x + tid_x;
	int pix_y = dim_y * bid_y + tid_y;
	
	// degree per pixel
	double dxdy;
	
	dxdy = (double)(Sx / (Px ));
	x =  (double)(pix_x) * dxdy + x_start;
	dxdy = (double)(Sy / (Py ));
	y =  -((double)(pix_y) * dxdy + y_start);

	%(type)s sum = %(type)s(0,0);
	int a = 0;

	for(int my = -My; my <= My; ++my)
	{
		for(int i = threadIdx.x + threadIdx.y * BLOCK_SIZE; i < (2*Mx+1); i += BLOCK_SIZE*BLOCK_SIZE)
		{
			s_Ds[i] = Ds[filter_id * Ds_ld + i + (my+My) * (2*Mx+1)];
		}
		__syncthreads();
		
		a = 0;
		for(int mx = -Mx; mx <= Mx; ++mx)
		{
			if( mx*mx * My * My + my*my * Mx * Mx <= Mx*Mx*My*My)// && mx*mx + my*my > 0)
			{
				sum += s_Ds[a] * %(exp)s(mx * WMx * x + my * WMy * y);
			}
			++a;
		}
		__syncthreads();
	}

        if(pix_x < Px && pix_y < Py)
        {
            int output_ind = (pix_y) * Px + (pix_x) + filter_id * dirich_ld;
            dirich_space[output_ind] = pycuda::real(sum);
        }
}

"""
	
    if typef == np.float64:
        Iexp = "Iexp"
    else:
        Iexp = "Iexpf"
    
    func = func_compile("psi_dirich_space_Kernel", dirich_template % {"type": dtype_to_ctype(dtype), "typef": dtype_to_ctype(typef), "exp": Iexp})
    return func
    
    


