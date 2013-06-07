try:
  import cv
  CV_INSTALLED = True
except ImportError:
  CV_INSTALLED = False

try:
  import pycuda.gpuarray as garray
  PYCUDA = True
except ImportError:
  PYCUDA = False

import numpy as np
import simpleio as io


def read_video(filename,write_to_file = False,color=False):
  """
  Read video from the specified file
  
  parameters:
  filename: The name of the video file to be read.
  write_to_file: True if the video should be written to an h5file.
                 False otherwise.
                 If not specified, will be set to False.
  
  color: True if the ouput video should have 3 channels - RGB
         False if the output video should be grayscale.
         If not specified, will be set to False

  returns: a numpy array representing the video
  """

  if(not(CV_INSTALLED)):
    raise ImportError("Failure to load OpenCV.\n \t " \
                      "read_video requires: OpenCV\n")
  
  vidFile = cv.CaptureFromFile(filename)
  nFrames = int(cv.GetCaptureProperty(vidFile, cv.CV_CAP_PROP_FRAME_COUNT))
  fps = cv.GetCaptureProperty( vidFile, cv.CV_CAP_PROP_FPS)
  
  if nFrames==0:
    raise TypeError("Could not read from %s.\n\t"
                    "Please make sure that the required codec is installed.\n"\
                    % filename)
  
  frameImg = cv.QueryFrame(vidFile)
  
  if color:
    size = (nFrames, frameImg.height, frameImg.width, 3)
  else:
    size = (nFrames, frameImg.height, frameImg.width)
  
  vid_arr = np.zeros(size)
  
  rgb2k = np.array([0.114, 0.587, 0.299])
  for f in xrange(nFrames):
    arr=cv2array(frameImg)
    if color:
      vid_arr[f,:,:,:] = arr
    else:
      vid_arr[f,:,:] = np.sum(arr*rgb2k, axis=-1)/255 
    frameImg = cv.QueryFrame(vidFile)
  
  if write_to_file:
    filename=filename.rpartition('.')[0]+".h5"
    io.write_memory_to_file(vid_arr,filename)    
  
  return vid_arr



def write_video(vid_arr, filename, fps=10, h5input=False,fourcc=None):
  """
  Writes video to the specified filename

  parameters:
  vid_arr: A numpy array, GPUArray or PitchArray representing the video
  filename: The output filename
  fps: The frame rate of the output video
       If not specified, will be set to 10.
  h5input: True if vid_arr is a filename of an h5 file containg the video.
            If not specified, will be set to False.
  fourcc: An integer representing the codec to be used for 
          the video file. Can be specified using cv.CV_FOURCC.
          If not specified, will default to DIVX.
  
  """

  if(not(CV_INSTALLED)):
    raise ImportError("Failure to load OpenCV.\n \t " \
                        "write_video requires: OpenCV\n")
  if(h5input):
    vid_arr=io.read_file(vid_arr)
  else:
    if PYCUDA:
      if vid_arr.__class__.__name__ in ["GPUArray" , "PitchArray"]:
        vid_arr=vid_arr.get()
      elif vid_arr.__class__.__name__ != "ndarray":
        raise TypeError("Write video error: Unknown input type")
    elif vid_arr.__class__.__name__ != "ndarray":
      raise TypeError("Write video error: Unknown input type")
  
  height = vid_arr.shape[1]
  width = vid_arr.shape[2]
  if vid_arr.min() < 0: 
    vid_arr = vid_arr - vid_arr.min()
  if vid_arr.max() > 1:
    vid_arr = vid_arr/vid_arr.max()
  if len(vid_arr.shape)==3:
   temp=vid_arr
   vid_arr=np.zeros((temp.shape[0], height, width, 3))
   for i in range(3):
     vid_arr[:,:,:,i] = 255* temp[:,:,:]
   del temp

  vid_arr = vid_arr.astype('uint8')
  
  if fourcc is None:
    fourcc = cv.CV_FOURCC('D','I','V','X')
  writer = cv.CreateVideoWriter(filename, fourcc, fps, (width, height), 1)

  for i in xrange(vid_arr.shape[0]):
    cv.WriteFrame(writer, array2cv(vid_arr[i,:,:,:]))
  



def cv2array(im):
  """ 
  Converts IplImage to numpy array
  """
  depth2dtype = {
        cv.IPL_DEPTH_8U: 'uint8',
        cv.IPL_DEPTH_8S: 'int8',
        cv.IPL_DEPTH_16U: 'uint16',
        cv.IPL_DEPTH_16S: 'int16',
        cv.IPL_DEPTH_32S: 'int32',
        cv.IPL_DEPTH_32F: 'float32',
        cv.IPL_DEPTH_64F: 'float64',
    }

  arrdtype=im.depth
  a = np.fromstring(
         im.tostring(),
         dtype=depth2dtype[im.depth],
         count=im.width*im.height*im.nChannels)
  a.shape = (im.height,im.width,im.nChannels)
  return a



def array2cv(a):
  """
  Converts numpy array to IplImage
  """
  dtype2depth = {
        'uint8':   cv.IPL_DEPTH_8U,
        'int8':    cv.IPL_DEPTH_8S,
        'uint16':  cv.IPL_DEPTH_16U,
        'int16':   cv.IPL_DEPTH_16S,
        'int32':   cv.IPL_DEPTH_32S,
        'float32': cv.IPL_DEPTH_32F,
        'float64': cv.IPL_DEPTH_64F,
    }
  try:
    nChannels = a.shape[2]
  except:
    nChannels = 1
  cv_im = cv.CreateImageHeader((a.shape[1],a.shape[0]),
          dtype2depth[str(a.dtype)],
          nChannels)
  cv.SetData(cv_im, a.tostring(),
             a.dtype.itemsize*nChannels*a.shape[1])
  return cv_im
