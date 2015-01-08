#!/usr/bin/env python

import tables
import pycuda.driver as cuda
import numpy as np

import vrf
import utils.spike_store as ss
import ensemble_encode as ee
import utils.parray
import utils.videoio as vio
from utils.simpleio import *


def VTEM_IAF(videofile, output_filename, Wt, rf, Wx=2*np.pi*4, Wy=2*np.pi*4,
             start_frame=0, end_frame=None, dx=1.0/16, dy=1.0/16, 
             fps=100, domain=None, h5input=True, npinput=False):
    """
    Encode a video with IAF neurons and the specified Receptive field
    
    Parameters
    -----------
    videofile : string
         
         ::
         
              Either
              1. Filename of a file containing the input video
              must be stored using write_memory_to_file in simpleio.py
              or using h5write in matlab
              Video array with shape (a,b,c)
              a: total number of frames
              b: number of pixels in y direction
              c: number of pixels in x direction
              c is the leading dimension
              in matlab should be (c,b,a)
              2. Or filename of a video file if h5input is set to False. 
              Will throw an error if OpenCV and the required codec are
              not installed         

    output_filename : string
         output filename that will contain the spike info
    Wt : float
         bandwidth in t variable
         if not specified, will use the info in spikefile
    rf : vrf 
         Receptive field object.
    Wx : float, optional
         bandwidth in x variable
         if not specified, will use the info in spikefile
    Wy : float, optional 
         bandwidth in y variable
         if not specified, will use the info in spikefile
    start_frame : integer, optional 
         starting frame to be encoded in the video  
    end_frame : integer, optional 
         ending frame to be encoded
         if not specified, will encoding to the end of the video
    dx : integer, optional 
         spatial resolution in x direction, distance between two pixels
    dy : integer, optional
         spatial resolution in y direction, distance between two pixels
    fps : integer, optional 
         frames per second of the video
    domain : list, optional 
         list of 2, [a,b], specifying the domain to encode
         a in x direction, b in y direction
         will only encode the center of the video with size [a,b]
         if not specified, the whole video screen will be encoded.       
    h5input : bool, optional
         True if the file specified is an h5 file.
         False if the file specified is a video file.
         If not specified, is set to True
    npinput : bool, optional 
         True if videofile is a numpy array.
         If not specified, will be set to False.

    Notes
    -----
    The coordinate system is given by the following
    
    ::    
    
            Row (width / X) major
            
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
        
    Examples
    --------

    >>> import atexit
    >>> import pycuda.driver as cuda
    >>> import numpy as np
    >>> from vtem import vtem,vtdm
    >>> cuda.init()
    >>> context1 = cuda.Device(0).make_context()
    >>> atexit.register(cuda.Context.pop)
    >>> gb = vrf.vrf_gabor((Py,Px), domain=domain, dx=dx, dy=dy)
    >>> gb.load_parameters()
    >>> gb.get_gabor_parameters(dilations = [1, 0, -1, -2, -3], num_rotations = 8, 
                                alpha0 = np.float64(2), b0 = np.float64(1.5), 
                                b0scalebase=np.sqrt(2), KAPPA = 3)
    >>> gb.load_parameters(set=1)
    >>> vtem.VTEM_Gabor_IAF('video.h5', 'spikes.h5', 2*np.pi*10, gb)

    """
    if npinput:
        videoarray = videofile
        h5input = False
    else:
        if h5input:
            h5file = tables.openFile(videofile, 'r')
            videoarray = h5file.root.real
        else:
            videoarray = vio.read_video(videofile)

    TotalFrames, Py, Px = videoarray.shape
    
    if end_frame is None:
        end_frame = TotalFrames
        
    if end_frame > TotalFrames:
        end_frame = TotalFrames
        print "warning: the end frame is changed to %d, which " \
        	  "is the maximum frame in the video file\n" % end_frame
    
    dt = 1.0 / fps
    T = (end_frame - start_frame)*dt

    dx = float(dx)
    dy = float(dy)

    print "number of neurons: %d" % (rf.num_neurons)
    
    if rf.__class__.__name__ == 'vrf_gabor':
        rftype = 'gabor'
    elif rf.__class__.__name__ == 'vrf_cs':
        rftype = 'cs'
    else:
        raise TypeError('Receptive field type not recognized\n.')
    
    spikefile = ss.vtem_storage(output_filename, rf.num_neurons, rftype)
    spikefile.write_video_attributes(Wt, Wx, Wy, rf.Px, rf.Py, dx, dy)
    
    if rftype == 'gabor':
        spikefile.write_gabor_parameters(rf)
    elif rftype == 'cs':
        spikefile.write_cs_parameters(rf)

    iaf = ee.IAF_encode(rf.num_neurons,dt)
    
    iaf.load_parameters()

    spikefile.write_neuron_parameters(iaf)
    
    freemem, totalmem = cuda.mem_get_info()
    
    ONE_TIME_FRAMES = min(101, np.int(min(end_frame - start_frame + 1, \
    	 2**(np.floor(np.log2((freemem/5) / videoarray.dtype.itemsize / \
         Px / Py))))))
    
    spikefile.write_refresh_interval((ONE_TIME_FRAMES-1)*dt)
    
    print "encoding..."
    total_spikes = 0
    block_video = 0
    for i in range(start_frame, end_frame, ONE_TIME_FRAMES-1):
        if(h5input):
            block_video = videoarray.read(i, i+ONE_TIME_FRAMES)
        else:
            block_video = videoarray[i:i+ONE_TIME_FRAMES,:,:]
        output = rf.filter(block_video)
        
        spikes,spike_count=iaf.encode(output, avg_rate=0.1)
        
        del output
        
        cum_spike_count = np.concatenate((np.zeros(1,np.int32),
                                          np.cumsum(spike_count)))
        
        all_spikes = np.empty(cum_spike_count[-1])
        for j in range(rf.num_neurons):
            all_spikes[cum_spike_count[j]:cum_spike_count[j+1]] = spikes[j]
        
        spikefile.write_spike_in_timebin(spike_count, all_spikes)
        
        del all_spikes
        iaf.reset_timer()
        total_spikes += cum_spike_count[-1]
        print "%.0f%%" % (float((i + ONE_TIME_FRAMES-1 - start_frame)) / \
        	(end_frame - start_frame)  *100)
    
    spikefile.close()
    if h5input:
        h5file.close()

    print "done, total spikes %d" % (total_spikes)
    

def VTEM_Gabor_IAF(videofile, output_filename, Wt, Wx=2*np.pi*4, Wy=2*np.pi*4,
                   start_frame=0, end_frame=None, dx=1.0/16, dy=1.0/16, 
                   fps=100, domain=None, h5input=True):
    """
    Encode a video with IAF neurons and Gabor receptive field
    with default parameters

    Parameters
    -----------
    videofile : string
         
         ::
         
              Either
              1. Filename of a file containing the input video
              must be stored using write_memory_to_file in simpleio.py
              or using h5write in matlab
              Video array with shape (a,b,c)
              a: total number of frames
              b: number of pixels in y direction
              c: number of pixels in x direction
              c is the leading dimension
              in matlab should be (c,b,a)
              2. Or filename of a video file if h5input is set to False. 
              Will throw an error if OpenCV and the required codec are
              not installed         

    output_filename : string
         output filename that will contain the spike info
    Wt : float
         bandwidth in t variable
         if not specified, will use the info in spikefile
    Wx : float, optional
         bandwidth in x variable
         if not specified, will use the info in spikefile
    Wy : float, optional 
         bandwidth in y variable
         if not specified, will use the info in spikefile
    start_frame : integer, optional 
         starting frame to be encoded in the video  
    end_frame : integer, optional 
         ending frame to be encoded
         if not specified, will encoding to the end of the video
    dx : integer, optional 
         spatial resolution in x direction, distance between two pixels
    dy: integer, optional
         spatial resolution in y direction, distance between two pixels
    fps : integer, optional 
         frames per second of the video
    domain : list, optional 
         list of 2, [a,b], specifying the domain to encode
         a in x direction, b in y direction
         will only encode the center of the video with size [a,b]
         if not specified, the whole video screen will be encoded.       
    h5input : bool, optional
         True if the file specified is an h5 file.
         False if the file specified is a video file.
         If not specified, is set to True

    Notes
    -----
    The coordinate system is given by the following
    
    ::    
    
            Row (width / X) major
            
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
      
    To specify receptive field parameters, create an object using
    vrf.vrf_gabor() with desired parameters and use function
    VTEM_IAF
    
    Examples
    --------
    >>> import atexit
    >>> import pycuda.driver as cuda
    >>> import numpy as np
    >>> from vtem import vtem,vtdm
    >>> cuda.init()
    >>> context1 = cuda.Device(0).make_context()
    >>> atexit.register(cuda.Context.pop)
    >>> vtem.VTEM_Gabor_IAF('video.h5', 'spikes.h5', 2*np.pi*10)
    
    """
    if h5input:
        h5file = tables.openFile(videofile, 'r')
        videoarray = h5file.root.real
    else:
        videoarray = vio.read_video(videofile)

    _ , Py, Px = videoarray.shape
    gb = vrf.vrf_gabor((Py,Px), domain=domain, dx=dx, dy=dy)
    gb.load_parameters()
    
    VTEM_IAF(videoarray, output_filename, Wt, gb, Wx, Wy,start_frame, 
             end_frame, dx, dy, fps, domain, h5input, npinput=True)
    if h5input:
        h5file.close()
    
    
def VTEM_CS_IAF(videofile, output_filename, Wt, Wx=2*np.pi*4, Wy=2*np.pi*4,
                start_frame=0, end_frame=None, dx=1.0/16, dy=1.0/16,
                fps=100, domain=None, h5input=True):
    """
    Encode a video with IAF neurons and Centre Surround receptive field
    with default parameters

    Parameters
    -----------
    videofile : string
         
         ::
         
              Either
              1. Filename of a file containing the input video
              must be stored using write_memory_to_file in simpleio.py
              or using h5write in matlab
              Video array with shape (a,b,c)
              a: total number of frames
              b: number of pixels in y direction
              c: number of pixels in x direction
              c is the leading dimension
              in matlab should be (c,b,a)
              2. Or filename of a video file if h5input is set to False. 
              Will throw an error if OpenCV and the required codec are
              not installed         

    output_filename : string
         output filename that will contain the spike info
    Wt : float
         bandwidth in t variable
         if not specified, will use the info in spikefile
    Wx : float, optional
         bandwidth in x variable
         if not specified, will use the info in spikefile
    Wy : float, optional 
         bandwidth in y variable
         if not specified, will use the info in spikefile
    start_frame : integer, optional 
         starting frame to be encoded in the video  
    end_frame : integer, optional 
         ending frame to be encoded
         if not specified, will encoding to the end of the video
    dx : integer, optional 
         spatial resolution in x direction, distance between two pixels
    dy: integer, optional
         spatial resolution in y direction, distance between two pixels
    fps : integer, optional 
         frames per second of the video
    domain : list, optional 
         list of 2, [a,b], specifying the domain to encode
         a in x direction, b in y direction
         will only encode the center of the video with size [a,b]
         if not specified, the whole video screen will be encoded.       
    h5input : bool, optional
         True if the file specified is an h5 file.
         False if the file specified is a video file.
         If not specified, is set to True

    Notes
    -----
    The coordinate system is given by the following
    
    ::    
    
            Row (width / X) major
            
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
      
    To specify receptive field parameters, create an object using
    vrf.vrf_cs() with desired parameters and use function
    VTEM_IAF

    Examples
    --------
    >>> import atexit
    >>> import pycuda.driver as cuda
    >>> import numpy as np
    >>> from vtem import vtem,vtdm
    >>> cuda.init()
    >>> context1 = cuda.Device(0).make_context()
    >>> atexit.register(cuda.Context.pop)
    >>> vtem.VTEM_CS_IAF('video.h5', 'spikes.h5', 2*np.pi*10)
    
    """
    if h5input:
        h5file = tables.openFile(videofile, 'r')
        videoarray = h5file.root.real
    else:
        videoarray = vio.read_video(videofile)

    _ , Py, Px = videoarray.shape
    cs = vrf.vrf_cs((Py,Px), domain=domain, dx=dx, dy=dy)
    cs.load_parameters()
    
    VTEM_IAF(videoarray, output_filename, Wt, cs, Wx, Wy,start_frame, 
             end_frame, dx, dy, fps, domain, h5input, npinput=True)
    if h5input:
        h5file.close()
   

