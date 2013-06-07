#!/usr/bin/env python

import argparse
import atexit
import pycuda.driver as cuda
import numpy as np
from vtem import vtem,vtdm

parser = argparse.ArgumentParser()
parser.add_argument("-d","--device",help="Choose CUDA device number[default:0]",
                    type=int)
parser.add_argument("-i","--input_filename",help=("Input filename[default: "
                    + "'test_files/fly_large.h5']"))
parser.add_argument("-o","--output_filename",help=("Output filename "
                    + "without extension[default: 'rec']"))

args = parser.parse_args()

if args.device:
    device_no = args.device
else:
    device_no =0

if args.input_filename:
    input_filename = args.input_filename
else:
    input_filename = 'test_files/fly_large.h5'

if args.output_filename:
    output_filename = args.output_filename
else:
    output_filename = "rec"

cuda.init()
context1 = cuda.Device(device_no).make_context()
atexit.register(cuda.Context.pop)

Mx = 40

vtem.VTEM_Gabor_IAF(input_filename, 'spikes.h5', 
                    2*np.pi*10, h5input=True)
vtdm.decode_video('spikes.h5', 'dsw.h5', 'dirich.h5'
          , 0, 1, 0.01, Mx, rnn=True, alpha=5000, steps=4000, 
          dtype = np.float32, stitching=True, stitch_interval =0.2,
          spatial_stitching = True, spatial_interval = [80,80],
          output=output_filename, output_format=1, write_blocks=True)

