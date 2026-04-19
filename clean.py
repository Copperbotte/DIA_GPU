#this program will apply the bias subtraction,
#flat fielding and subtract the gradient background,
#it will also align the images to the first image in the list

#if you use this code, please cite Oelkers et al. 2015, AJ, 149, 50

#import the relevant libraries for basic tools
import numpy
import scipy
from scipy import stats
import scipy.ndimage as ndimage
import astropy
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
from astropy.wcs import WCS
from astropy.stats import sigma_clip as astropy_sigma_clip
import math
import time
import argparse
import sys

#libraries for image registration
import FITS_tools
from FITS_tools.hcongrid import hcongrid

#import relevant libraries for a list
import glob, os
from os import listdir
from os.path import isfile, join

#import relevant spline libraries
from scipy.interpolate import Rbf
import scipy.interpolate as scipy_interp

# Modern tools
import numpy as np
import numba

#####UPDATE INFORMATION HERE####
#DO YOU WANT TO FLAT FIELD AND BIAS SUBTRACT?
biassub = 0 # yes = 1 no = 0 to bias subtract
flatdiv = 0 # yes = 1 no = 0 to flat field
align = 1# yes = 1 no = 0 to align based on coordinates

#useful directories
#rawdir = '.../cal/' #directory with the raw images
#cdedir = '.../code/clean/' #directory where the code 'lives'
caldir = 'N/A' #directory with the calibration images such as bias & flat
#clndir = '../clean/'#directory for the cleaned images to be output

# Use forward slashes so the path works on Windows without escaping
from pathlib import Path
ROOT = Path('C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/')
cdedir = ROOT / "DIA" / "routines" / "Python"
#rawdir = ROOT / "DIA_TEMP" / "raw"
rawdir = ROOT / "TESS_sector_4"
#clndir = ROOT / "DIA_TEMP" / "clean"
clndir = ROOT / "DIA_TEMP" / "clean3"

# ensure the output directories exist
ROOT = Path(ROOT)#.mkdir(parents=True, exist_ok=True)
cdedir = Path(cdedir)#.mkdir(parents=True, exist_ok=True)
rawdir = Path(rawdir)#.mkdir(parents=True, exist_ok=True)
clndir = Path(clndir)#.mkdir(parents=True, exist_ok=True)

if False:
    file = 'tess2018292095939-s0004-1-4-0124'


    def print_header_if_matching(folder, file):
        files_ = [f for f in folder.glob("*.fits") if isfile(join(folder, f))]
        ref_, head_ = fits.getdata(os.path.join(folder, files_[0]), header = True)
        print(f"Header for {folder / files_[0]}:")
        print(head_.tostring('\n'))
        print('#'*80)
        print('#'*80)
        print('#'*80)
        globals().update(locals())




    #with fits.open('C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/TESS_sector_4/tess2018292095939-s0004-1-4-0124-s_ffic.fits') as hdul:
    #    print('raw:', hdul[0].header['DQUALITY'])

    print_header_if_matching(rawdir, file)
    print_header_if_matching(ROOT / "DIA_TEMP" / "clean2", file)
    print_header_if_matching(ROOT / "DIA_TEMP" / "clean3", file)


    raise Exception


#sample every how many pixels? usually 32x32 is OK but it can be larger or smaller
pix = 32 # UPDATE HERE FOR BACKGROUND SPACING
axs = 2048 # UPDATE HERE FOR IMAGE AXIS SIZE
###END UPDATE INFORMATION###

#get the image list and the number of files which need reduction
#os.chdir(rawdir) #changes to the raw image direcotory
files = [f for f in rawdir.glob("*.fits") if isfile(join(rawdir, f))] #gets the relevant files with the proper extension

# Cull files that don't match the specified camera and CCD
camera, ccd = '1', '4'
def get_camera_and_ccd(f):
    # with fits.open(f, memmap=True) as hdul:
    #     camera = hdul[1].header['CAMERA']
    #     ccd = hdul[1].header['CCD']
    # Grab these from the filename instead due to perf loss
    # tess2018297215939-s0004-1-4-0124-s_ffic.fits
    filename = f.stem  # gets filename without extension
    parts = filename.split('-')
    camera = parts[2]
    ccd = parts[3]
    return camera, ccd

def filter_file(f):
    #img_data, img_header = fits.getdata(f, header=True)
    #img_header = fits.getheader(f)
    # Get these from the data's header manually
    camera_val, ccd_val = get_camera_and_ccd(f)
    return (camera_val == camera) and (ccd_val == ccd)
files = list(filter(filter_file, files))

files.sort()
nfiles = len(files)
#os.chdir(cdedir) #changes back to the code directory

#get the zeroth image for registration
#read in the image
ref, rhead = fits.getdata(os.path.join(rawdir, files[0]), header = True)
rhead['CRPIX1'] = 1001.
rhead['NAXIS1'] = 2048
rhead['NAXIS2'] = 2048

#sample every how many pixels?
bxs = 512 #how big do you want to make the boxes for each image?
#bxs = axs # Do nothing #WARNING: O(n^3) 
#bxs = 1024
#bxs = 128 # Warning: O(n^3) 
lop = 2*pix
sze = int((bxs//pix)*(bxs//pix) + 2*(bxs//pix) + 1) #size holder for later

#read in the flat
if (flatdiv == 1):
    flist = fits.open(os.path.join(caldir, 'flat.fits'))
    fheader = flist[0].header #get the header info
    flat = flist[0].data #get the image info

#read in the bias
if (biassub == 1):
    blist = fits.open(os.path.join(caldir, 'bias.fits'))
    bheader = blist[0].header #get the header info
    bias = blist[0].data #get the image info


from time import time_ns
import humanize
# Usage: Create a TimerGroup with a set of keys.
# Then for each key, we call 'from timer_group with key'. Is that valid syntax?
# Can I instead do with timer_group['key'] and have it persist state?
class ProfileTimer_Inner:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key

    def __enter__(self):
        self.parent.start_times[self.key] = time_ns()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.parent.start_times[self.key] is not None:
            self.parent.elapsed_times[self.key].append(time_ns() - self.parent.start_times[self.key])
            #print(f"Timer for '{self.key}' recorded {self.parent.elapsed_times[self.key] / 1e9:.2f} seconds")
            self.parent.start_times[self.key] = None

class ProfileTimer:
    def __init__(self):
        self.keys = []
        self.start_times = {}
        self.elapsed_times = {}

    # If the key isn't present, spawn a new timer
    def __getitem__(self, key):
        if key not in self.keys:
            self.keys.append(key)
            self.start_times[key] = None
            self.elapsed_times[key] = []
            print(f"Created timer for '{key}'")
        return ProfileTimer_Inner(self, key)
    
    # def __enter__(self):
    #     for key in self.keys:
    #         self.start_times[key] = time_ns()
    #     return self
    # 
    # def __exit__(self, exc_type, exc_value, traceback):
    #     for key in self.keys:
    #         if self.start_times[key] is not None:
    #             self.elapsed_times[key].append(time_ns() - self.start_times[key])
    #             #print(f"Timer for '{key}' recorded {self.elapsed_times[key] / 1e9:.2f} seconds")
    #             self.start_times[key] = None

    def get_elapsed_times(self):
        secs = {key: np.array(elapsed) / 1e9 for key, elapsed in self.elapsed_times.items()}  # Convert nanoseconds to seconds
        return {key: [np.mean(elapsed[0:]), np.std(elapsed[0:]), len(elapsed)] for key, elapsed in secs.items()}  # Return average time for each key
        #return {key: [np.mean(elapsed[1:]), np.std(elapsed[1:]), len(elapsed)-1] for key, elapsed in secs.items()}  # Return average time for each key
    
    def __str__(self):
        elapsed_times = self.get_elapsed_times()
        # pre-stringify all parameters for better formatting
        keys = list(elapsed_times.keys())
        params = []
        for key in keys:
            val, std, count = elapsed_times[key]
            params.append([f'{key}', f'{val:.6f}', f'{std:.6f}', f'{count}', f'{val*count:.6f}', f'{humanize.metric(1/val, "Hz")}'])

        # align by specific directions for readability
        #param_max_lengths = list(map(lambda col: max(len(col)), zip(*params)))
        #param_lengths = [list(map(len, col)) for col in zip(*params)]
        params_T = list(zip(*params))
        param_lengths_T = [list(map(len, col)) for col in params_T]
        param_max_lengths = [max(lengths) for lengths in param_lengths_T]
        for n in range(len(params)):
            params[n][0] = params[n][0].rjust(param_max_lengths[0])
            params[n][1] = params[n][1].rjust(param_max_lengths[1])
            params[n][2] = params[n][2].ljust(param_max_lengths[2])
            params[n][3] = params[n][3].rjust(param_max_lengths[3])
            params[n][4] = params[n][4].rjust(param_max_lengths[4])
            params[n][5] = params[n][5].rjust(param_max_lengths[5])

        #return '\n'.join(f'{param[0]}: {param[1]} ± {param[2]} seconds for n={param[3]} in {param[4]} seconds' for param in params)
        return '\n'.join(f'{param[0]}: ({param[5]}) : {param[1]} ± {param[2]} seconds for n={param[3]} in {param[4]} seconds' for param in params)
        #return '\n'.join(f'{key}: {elapsed[0]:.6f} ± {elapsed[1]:.6f} seconds for n={elapsed[2]} in {elapsed[0]*elapsed[2]:.6f} seconds' for key, elapsed in elapsed_times.items())

#@numba.njit #(parallel=True) # Parallel seems to perform worse.
#@numba.njit()#parallel=True) # Parallel seems to perform worse.
def custom_rbf_eval_OLD(x, y, XI, YI, coeffs, shift, scale):
    # x: 1D array of x-coordinates of the INPUT sampled points (ex. 289)
    # y: 1D array of y-coordinates of the INPUT sampled points (ex. 289)
    # v: 1D array of values at the INPUT sampled points (ex. 289)
    ## rbf.nodes: 1D array of the fit nodes. Replaces v. (ex. 289)
    # rbf2._coeffs: "2D" array of the fit nodes and linear component. Replaces v. (ex. 289+3 x 1)
    # XI: 2D array of x-coordinates of the OUTPUT grid (ex. 512x512)
    # YI: 2D array of y-coordinates of the OUTPUT grid (ex. 512x512)
    # reshld: 2D array of the OUTPUT grid values (ex. 512x512)

    # with timers['rbf_creation']:
    #     rbf = Rbf(x, y, v, function = 'thin-plate', smooth = 0.0)
    # with timers['rbf_interpolation']:
    #     reshld = rbf(XI, YI)
    #     globals().update(locals())
    
    result = np.zeros(XI.shape, dtype=np.float64)
    dxy = np.zeros((4, x.shape[0]), dtype=np.float64)
    for idy in range(XI.shape[0]):
        for idx in range(XI.shape[1]):
            dxy[0] = XI[idy, idx] - x
            dxy[1] = YI[idy, idx] - y
            # dot product
            dxy[2] = dxy[0]**2 + dxy[1]**2
            dxy[3] = np.sqrt(dxy[2])          # r = euclidean distance
            dxy[3] = dxy[2]*np.log(dxy[3])     # r²·log(r)  (matches scipy exactly)
            dxy[3][dxy[2] == 0] = 0             # handle the singularity at zero distance
            #result[idy, idx] += np.sum(dxy[3] * rbf.nodes)
            #result[idy, idx] += np.sum(dxy[3] * rbf._coeffs[:-3, 0]) # use the fit nodes, not the original values
            result[idy, idx] += np.sum(dxy[3] * coeffs[:-3, 0]) # use the fit nodes, not the original values
    #result += rbf._coeffs[-3, 0] + rbf._coeffs[-2, 0]*(XI - rbf._shift[0])/rbf._scale[0] + rbf._coeffs[-1, 0]*(YI - rbf._shift[1])/rbf._scale[1] # add the linear component
    result += coeffs[-3, 0] + coeffs[-2, 0]*(XI - shift[0])/scale[0] + coeffs[-1, 0]*(YI - shift[1])/scale[1] # add the linear component
    return result

    #return result, np.max(np.abs(result - reshld))
    #return np.max(np.abs(result - reshld))

#@numba.njit#(parallel=True)
def custom_rbf_eval_CPU(x, y, XI, YI, coeffs, shift, scale):
    """
    Evaluate a TPS RBF on a regular grid, matching scipy bit-for-bit.

    Uses the same matrix-multiply approach as scipy internally:
    build the (M, N+3) evaluation matrix, then ``np.dot(vec, coeffs)``.
    BLAS DGEMV does pairwise/blocked accumulation, eliminating the
    O(N·eps) summation error of a naive per-pixel loop.

    NOT numba-compilable — the precision comes from BLAS internals.
    """
    N = len(x)
    H, W = XI.shape
    M = H * W

    # Build (N, 2) observation array and (M, 2) grid array
    obs = np.empty((N, 2), dtype=np.float64)
    obs[:, 0] = x
    obs[:, 1] = y

    grid = np.empty((M, 2), dtype=np.float64)
    grid[:, 0] = XI.ravel()
    grid[:, 1] = YI.ravel()

    shift_arr = np.asarray(shift, dtype=np.float64).ravel()
    scale_arr = np.asarray(scale, dtype=np.float64).ravel()
    if scale_arr.shape[0] == 1:
        scale_arr = np.array([scale_arr[0], scale_arr[0]], dtype=np.float64)

    # Normalised eval coords (polynomial part)
    xhat = np.empty_like(grid)
    xhat[:, 0] = (grid[:, 0] - shift_arr[0]) / scale_arr[0]
    xhat[:, 1] = (grid[:, 1] - shift_arr[1]) / scale_arr[1]

    # Process in chunks to bound memory at ~100 MB
    chunk_sz = max(1, 100_000_000 // (N * 8))
    out = np.empty(M, dtype=np.float64)

    for i0 in range(0, M, chunk_sz):
        i1 = min(i0 + chunk_sz, M)
        g  = grid[i0:i1]                                  # (C, 2)
        C  = i1 - i0

        # Kernel matrix: r²·log(r) with RAW coordinates
        diff = g[:, None, :] - obs[None, :, :]             # (C, N, 2)
        r2   = np.sum(diff ** 2, axis=2)                    # (C, N)
        K    = np.where(r2 > 0, r2 * np.log(np.sqrt(r2)), 0.0)

        # Evaluation matrix  (C, N+3)  =  [K | 1 | xhat_x | xhat_y]
        vec = np.empty((C, N + 3), dtype=np.float64)
        vec[:, :N]   = K
        vec[:, N]    = 1.0
        vec[:, N+1]  = xhat[i0:i1, 0]
        vec[:, N+2]  = xhat[i0:i1, 1]

        # BLAS matrix-vector multiply — same accumulation as scipy
        out[i0:i1] = np.dot(vec, coeffs).ravel()

    return out.reshape(H, W)


@numba.njit
def custom_rbf_eval_numba(x, y, XI, YI, coeffs, shift, scale):
    """
    Numba-compiled TPS RBF evaluator with Kahan compensated summation.

    Same formula as custom_rbf_eval, but expressed as a scalar loop so
    numba can JIT-compile it.  Kahan summation reduces accumulation
    error from O(N·eps) to O(eps²), giving ~1e-15 max error.
    """
    H = XI.shape[0]
    W = XI.shape[1]
    N = x.shape[0]
    result = np.zeros((H, W), dtype=np.float64)

    # Extract polynomial coefficients (last 3 rows of coeffs)
    c0 = coeffs[N, 0]      # constant
    c1 = coeffs[N + 1, 0]  # x coefficient
    c2 = coeffs[N + 2, 0]  # y coefficient
    shift_x = shift[0]
    shift_y = shift[1]
    scale_x = scale[0]
    scale_y = scale[-1]     # handles both scalar and 2-element scale

    for idy in range(H):
        for idx in range(W):
            u = XI[idy, idx]
            v = YI[idy, idx]

            # ── Kahan-compensated RBF kernel sum ──
            s    = 0.0   # running sum
            comp = 0.0   # compensation for lost low-order bits
            for k in range(N):
                dx = u - x[k]
                dy = v - y[k]
                r2 = dx * dx + dy * dy
                if r2 > 0.0:
                    rbf_val = r2 * np.log(np.sqrt(r2))
                else:
                    rbf_val = 0.0

                # ── Non-Kahan variant ──
                # s += rbf_val * coeffs[k, 0]

                # ── Kahan extra terms ──
                term = rbf_val * coeffs[k, 0] - comp
                t    = s + term
                comp = (t - s) - term
                s    = t

            # ── Polynomial (normalised coordinates) ──
            s += c0 + c1 * (u - shift_x) / scale_x + c2 * (v - shift_y) / scale_y

            result[idy, idx] = s

    return result

def custom_rbf_eval(x, y, XI, YI, coeffs, shift, scale):
    #return custom_rbf_eval_CPU(x, y, XI, YI, coeffs, shift, scale)
    return custom_rbf_eval_numba(x, y, XI, YI, coeffs, shift, scale)


@numba.njit#(parallel=True)
def sigma_clip_numba(data, clip_low, clip_high):
    """
    Sigma clip via in-place compaction.  Returns (count, low_bound, high_bound).

    `data` is modified in-place: data[:count] holds the surviving values.
    No sorting, no masks, no temporary arrays — just two passes per
    iteration (stats + compact) in tight machine code.
    """
    size = len(data)
    while True:
        # Pass 1: sum and sum-of-squares in one sweep
        s  = 0.0
        s2 = 0.0
        for i in range(size):
        #for i in numba.prange(size):
            v = data[i]
            s  += v
            s2 += v * v
        mean = s / size
        var  = s2 / size - mean * mean
        std  = var ** 0.5
        lo = mean - std * clip_low
        hi = mean + std * clip_high

        # Pass 2: compact survivors to the front
        j = 0
        for i in range(size):
        #for i in numba.prange(size):
            v = data[i]
            if v >= lo and v <= hi:
                data[j] = v
                j += 1
        if j == size:
            return size, lo, hi, std
        size = j


@numba.njit(parallel=True)
def sigma_clip_mask_numba(data, mask, clip_low, clip_high):
    """
    Sigma clip that produces a boolean mask instead of compacting.

    `data` is read-only.  `mask` is written: True = kept, False = clipped.
    Returns (count, low_bound, high_bound).

    This is the variant you'd translate to a CUDA kernel — each pixel
    reads its value + mask flag, writes updated mask.  No data movement.
    """
    n = len(data)
    for i in range(n):
        mask[i] = True
    size = n
    while True:
        s  = 0.0
        s2 = 0.0
        #for i in range(n):
        for i in numba.prange(n):
            if mask[i]:
                v = data[i]
                s  += v
                s2 += v * v
        mean = s / size
        var  = s2 / size - mean * mean
        std  = var ** 0.5
        lo = mean - std * clip_low
        hi = mean + std * clip_high
        changed = 0
        #for i in range(n):
        for i in numba.prange(n):
            if mask[i] and (data[i] < lo or data[i] > hi):
                mask[i] = False
                changed += 1
        if changed == 0:
            return size, lo, hi, std
        size -= changed

import subprocess as sp
# Prints a line ran through a subprocess how you expect it to work!
def Call(command, getlines=False, silence=False):
    pprint = "> %s"%command
    print(pprint)
    lines = []
    with sp.Popen(command,
        stdout=sp.PIPE, stderr=sp.STDOUT,
        shell=True, universal_newlines=True # universal newlines makes it not a binary output
    ) as proc:
        for line in iter(proc.stdout.readline, ""):
            if not silence:
                print(line[:-1])
            if getlines:
                lines.append(line[:-1])
            
    if getlines:
        return proc.returncode, lines
    return proc.returncode

# if Fa lse:
import os # https://stackoverflow.com/questions/40856535/pycuda-nvcc-fatal-cannot-find-compiler-cl-exe-in-path
if Call("cl.exe"):#(os.system("cl.exe")):
    pre, post = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC", r"bin\Hostx64\x64"
    os.environ['PATH'] += ';'+os.path.join(pre, sorted(os.listdir(pre))[-1], post)
if Call("cl.exe"):#(os.system("cl.exe")):
    raise RuntimeError("cl.exe still not found, path probably incorrect")
Call(r'"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vsvarsall.bat" x64')

import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as nvcc

cu_code = r"""
// Map 2D thread/block indices to a 1D index, return false if out of bounds
// Use at the start of a kernel with:
// int idx;
// if (mapidx(&idx, width, height)) return;
// idx is now bounded and has the right index!
// 
__device__ bool mapidx(int* idx, int width, int height) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    *idx = y * width + x;

    //bool bounded_below = (0 <= x) && (0 <= y);
    bool bounded_above = (x < width) && (y < height);
    //return !(bounded_below && bounded_above);

    // Is bounded below guaranteed?
    return !bounded_above; 
}
#define DEF_IDX(width, height) int idx; if(mapidx(&idx, width, height)) return;
"""

cu_code += r"""

__constant__ double c_shiftscale[4]; // shift_x, shift_y, scale_x, scale_y
__constant__ unsigned int c_width_height[2]; // height, width
__constant__ unsigned int c_npoints; // number of sampled points
__constant__ double c_linear[3]; // a + bx + cy linear component

// dst:    output grid (reshld), size width*height
// src_xy: input sample coords, interleaved {x,y}, length 2*npoints
// src_c:  RBF coefficients for each sample point, length npoints
//
// Output grid coordinates are derived from the thread index (integer
// meshgrid matching linspace(0, bxs-1, bxs)), so no dst_xy buffer is needed.
__global__
void rbf_interpolate_cuda_f32(float* dst, float* src_xy, float* src_c) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col >= (int)c_width_height[1] || row >= (int)c_width_height[0]) return;
    int idx = row * (int)c_width_height[1] + col;

    float u = (float)col;  // XI[row, col] = col
    float v = (float)row;  // YI[row, col] = row

    // Kahan (compensated) summation: reduces accumulation error
    // from O(N·eps) to O(eps²).  Cost: 3 extra FP ops per iteration,
    // fully pipelined — negligible vs the log/sqrt.
    float result = 0.0f;
    float comp   = 0.0f;  // running compensation
    for (unsigned int i = 0; i < c_npoints; i++) {
        float dx = u - src_xy[i*2 + 0];
        float dy = v - src_xy[i*2 + 1];
        float r2 = dx*dx + dy*dy;
        float rbf_val = (r2 > 0.0f) ? (r2 * logf(sqrtf(r2))) : 0.0f;
        //result += rbf_val * src_c[i];

        // Kahan summation
        float term = rbf_val * src_c[i] - comp;
        float t    = result + term;
        comp   = (t - result) - term;
        result = t;
    }
    result += (float)c_linear[0]
            + (float)c_linear[1] * (u - (float)c_shiftscale[0]) / (float)c_shiftscale[2]
            + (float)c_linear[2] * (v - (float)c_shiftscale[1]) / (float)c_shiftscale[3];

    dst[idx] = result;
}

// Let's update this one to actually use float32 internally for the for loop.
// Performance matters a LOT in this case.
__global__
void rbf_interpolate_cuda_f64_storage(double* dst, double* src_xy, double* src_c) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col >= (int)c_width_height[1] || row >= (int)c_width_height[0]) return;
    int idx = row * (int)c_width_height[1] + col;

    float u = (float)col;  // XI[row, col] = col
    float v = (float)row;  // YI[row, col] = row

    // Mixed-precision Kahan summation: f32 kernel math, f64 accumulation.
    // The individual rbf_val * coeff products are f32 (~7 digits), but
    // the Kahan bookkeeping (result, comp, term, t) is f64 so we capture
    // the low-order bits that f32 addition would lose.  This gives
    // roughly f64-quality summation from f32-quality terms.
    double result = 0.0;
    double comp   = 0.0;
    for (unsigned int i = 0; i < c_npoints; i++) {
        float dx = (float)u - (float)src_xy[i*2 + 0];
        float dy = (float)v - (float)src_xy[i*2 + 1];

        float r2 = dx*dx + dy*dy;
        float rbf_val = (r2 > 0.0f) ? (r2 * logf(r2)) * 0.5f : 0.0f;

        // Kahan: term and t MUST be double to capture the residual
        double term = (double)rbf_val * (double)src_c[i] - comp;
        double t    = result + term;
        comp   = (t - result) - term;
        result = t;
        // This works by tracking the low-order bits lost to rounding in 'comp', and subtracting it from the next term to add, so that it gets included in the sum instead of being lost. The new 'comp' is then the low-order bits lost from adding the term to the result, which will be corrected in the next iteration. This way, we effectively keep a running compensation for lost precision, allowing us to achieve much better accuracy than a naive summation.
        // Aha! So its kinda like an error diffusion technique, or a kind of dithering?
        // copilot: Exactly! It's a way to "diffuse" the rounding errors across the entire summation, rather than letting them accumulate in a way that can lead to significant loss of precision. By keeping track of the compensation, we can ensure that the final result is much closer to what we would get with higher precision arithmetic, even though we're using lower precision for the intermediate calculations.
    }
    result += c_linear[0]
            + c_linear[1] * ((double)u - c_shiftscale[0]) / c_shiftscale[2]
            + c_linear[2] * ((double)v - c_shiftscale[1]) / c_shiftscale[3];

    dst[idx] = result;
}

__global__
void rbf_interpolate_cuda_f64(double* dst, double* src_xy, double* src_c) {
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col >= (int)c_width_height[1] || row >= (int)c_width_height[0]) return;
    int idx = row * (int)c_width_height[1] + col;

    double u = (double)col;  // XI[row, col] = col
    double v = (double)row;  // YI[row, col] = row

    // Kahan (compensated) summation: reduces accumulation error
    // from O(N·eps) to O(eps²).  Cost: 3 extra FP ops per iteration,
    // fully pipelined — negligible vs the log/sqrt.
    double result = 0.0;
    double comp   = 0.0;  // running compensation
    for (unsigned int i = 0; i < c_npoints; i++) {
        double dx = u - src_xy[i*2 + 0];
        double dy = v - src_xy[i*2 + 1];

        double r2 = dx*dx + dy*dy;
        double rbf_val = (r2 > 0.0) ? (r2 * log(sqrt(r2))) : 0.0;
        double term = rbf_val * src_c[i] - comp;
        double t    = result + term;
        comp   = (t - result) - term;
        result = t;
    }
    result += c_linear[0]
            + c_linear[1] * (u - c_shiftscale[0]) / c_shiftscale[2]
            + c_linear[2] * (v - c_shiftscale[1]) / c_shiftscale[3];

    dst[idx] = result;
}

// ─── Sigma-clip kernels ──────────────────────────────────────────────
// Masked parallel reduction: one pass computes per-block partial
// (sum, sum_sq, count).  Grid-stride loop keeps block count small.
__global__
void sigma_clip_reduce(const double* data, const int* mask,
                       double* partial_sum, double* partial_sum_sq,
                       int* partial_count, int N) {
    extern __shared__ char _smem[];
    double* s_sum    = (double*)_smem;
    double* s_sum_sq = s_sum + blockDim.x;
    int*    s_count  = (int*)(s_sum_sq + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double lsum = 0.0, lsum2 = 0.0;
    int lcnt = 0;
    for (int i = gid; i < N; i += stride) {
        if (mask[i]) {
            double v = data[i];
            lsum  += v;
            lsum2 += v * v;
            lcnt++;
        }
    }

    s_sum[tid]    = lsum;
    s_sum_sq[tid] = lsum2;
    s_count[tid]  = lcnt;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]    += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
            s_count[tid]  += s_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sum[blockIdx.x]    = s_sum[0];
        partial_sum_sq[blockIdx.x] = s_sum_sq[0];
        partial_count[blockIdx.x]  = s_count[0];
    }
}

// Per-element mask update: clip values outside [lo, hi].
// Atomically increments *d_changed for each newly-masked pixel.
__global__
void sigma_clip_update_mask(const double* data, int* mask,
                            double lo, double hi,
                            int* d_changed, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    if (mask[gid]) {
        double v = data[gid];
        if (v < lo || v > hi) {
            mask[gid] = 0;
            atomicAdd(d_changed, 1);
        }
    }
}

// ─── f32 sigma-clip kernels ─────────────────────────────────────────
// Same algorithm as f64, but data is float32 → half the memory bandwidth.
// Reduction accumulates into f64 for accuracy; mask update reads f32.
__global__
void sigma_clip_reduce_f32(const float* data, const int* mask,
                           double* partial_sum, double* partial_sum_sq,
                           int* partial_count, int N) {
    extern __shared__ char _smem[];
    double* s_sum    = (double*)_smem;
    double* s_sum_sq = s_sum + blockDim.x;
    int*    s_count  = (int*)(s_sum_sq + blockDim.x);

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double lsum = 0.0, lsum2 = 0.0;
    int lcnt = 0;
    for (int i = gid; i < N; i += stride) {
        if (mask[i]) {
            double v = (double)data[i];
            lsum  += v;
            lsum2 += v * v;
            lcnt++;
        }
    }
    s_sum[tid]    = lsum;
    s_sum_sq[tid] = lsum2;
    s_count[tid]  = lcnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]    += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
            s_count[tid]  += s_count[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        partial_sum[blockIdx.x]    = s_sum[0];
        partial_sum_sq[blockIdx.x] = s_sum_sq[0];
        partial_count[blockIdx.x]  = s_count[0];
    }
}

__global__
void sigma_clip_update_mask_f32(const float* data, int* mask,
                                double lo, double hi,
                                int* d_changed, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    if (mask[gid]) {
        double v = (double)data[gid];
        if (v < lo || v > hi) {
            mask[gid] = 0;
            atomicAdd(d_changed, 1);
        }
    }
}

// ─── Batched sigma-clip kernels ─────────────────────────────────────
// Process n_chunks independent chunks in ONE kernel launch.
// Each chunk has chunk_size elements; chunk c starts at data[c*chunk_size].
// Reduction: each block handles ONE chunk (blockIdx.x = chunk index,
//            threads grid-stride within that chunk).
// We launch n_chunks blocks × tpb threads.
__global__
void sigma_clip_reduce_batched(const double* data, const int* mask,
                               double* chunk_sum, double* chunk_sum_sq,
                               int* chunk_count,
                               int chunk_size, int n_chunks) {
    extern __shared__ char _smem[];
    double* s_sum    = (double*)_smem;
    double* s_sum_sq = s_sum + blockDim.x;
    int*    s_count  = (int*)(s_sum_sq + blockDim.x);

    int chunk = blockIdx.x;  // one block per chunk
    if (chunk >= n_chunks) return;
    int tid = threadIdx.x;
    int base = chunk * chunk_size;

    double lsum = 0.0, lsum2 = 0.0;
    int lcnt = 0;
    for (int i = tid; i < chunk_size; i += blockDim.x) {
        int gi = base + i;
        if (mask[gi]) {
            double v = data[gi];
            lsum  += v;
            lsum2 += v * v;
            lcnt++;
        }
    }
    s_sum[tid]    = lsum;
    s_sum_sq[tid] = lsum2;
    s_count[tid]  = lcnt;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid]    += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
            s_count[tid]  += s_count[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        chunk_sum[chunk]    = s_sum[0];
        chunk_sum_sq[chunk] = s_sum_sq[0];
        chunk_count[chunk]  = s_count[0];
    }
}

// Mask update for batched: each chunk has its own [lo, hi] bounds.
// bounds[chunk*2] = lo, bounds[chunk*2+1] = hi.
// d_changed[chunk] is atomically incremented per chunk.
__global__
void sigma_clip_update_mask_batched(const double* data, int* mask,
                                    const double* bounds, int* d_changed,
                                    int chunk_size, int total_N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_N) return;
    if (mask[gid]) {
        int chunk = gid / chunk_size;
        double lo = bounds[chunk * 2];
        double hi = bounds[chunk * 2 + 1];
        double v = data[gid];
        if (v < lo || v > hi) {
            mask[gid] = 0;
            atomicAdd(&d_changed[chunk], 1);
        }
    }
}

"""
# Sanitize
import re
def strip_cuda_comments(src: str) -> str:
    # remove // comments to end of line
    src = re.sub(r"//.*?$", "", src, flags=re.M)
    # remove /* … */ block comments (multiline)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return src
cu_code = strip_cuda_comments(cu_code)

krnl = nvcc.SourceModule(cu_code)

class CUDARBFInterpolator:
    """
    GPU-accelerated thin-plate spline RBF interpolator.

    Wraps ``scipy.interpolate.RBFInterpolator`` for coefficient fitting (CPU)
    and a custom CUDA kernel for fast evaluation on a regular output grid.

    The output grid is an integer meshgrid ``(0..width-1, 0..height-1)``;
    coordinates are derived from the CUDA thread index, eliminating the
    need for a separate dst_xy buffer.

    All GPU memory is allocated once at construction time so that the
    per-frame cost is just a few small memcpy's + one kernel launch.
    """

    def __init__(self, krnl, width, height, max_n_points, mode=None):
        """
        Parameters
        ----------
        krnl : pycuda.compiler.SourceModule
            Compiled CUDA module containing ``rbf_interpolate_cuda``.
        width, height : int
            Dimensions of the output evaluation grid.
        max_n_points : int
            Upper bound on the number of RBF sample points.
        mode : str, optional
            Precision mode for the GPU kernel. Options are "f32", "f64_storage", and "f64".
        """
        mode = "f64" if mode is None else mode #handle python quirk where this can be changed
        # -------------------------------------------------------------------
        self.width  = width
        self.height = height
        self.max_n_points = max_n_points
        self.n_pixels = width * height

        # Float type selection for testing precision tradeoffs. 
        # The kernel uses float internally, so we need to convert to float32 for the GPU. This is a bit sad, but it seems that using float64 storage with float32 math is more accurate than using float64 math, which is likely due to the fact that the kernel's math is optimized for float32 and may not handle float64 as well. If we used float64 storage with float64 math, we might get even better precision, but it would likely be much slower. So we'll stick with float32 storage and math for the GPU, and just convert to float64 on the CPU side for better precision in the final output.
        
        # ---- Kernel handle + dtype config ---------------------------------
        LUT = {
            "f32":         (np.float32, krnl.get_function("rbf_interpolate_cuda_f32")),
            "f64_storage": (np.float64, krnl.get_function("rbf_interpolate_cuda_f64_storage")),
            # use the f64 storage kernel for better precision, even though it has f32 math internally (Test perf on this)
            "f64":         (np.float64, krnl.get_function("rbf_interpolate_cuda_f64"))
        }
        #self.float_t, self.kernel = LUT["f64_storage"] # UPDATE HERE FOR KERNEL/DTYPE SELECTION
        self.float_t, self.kernel = LUT[mode]

        # ---- GPU buffer allocations (once) --------------------------------
        sizeof_float = np.dtype(self.float_t).itemsize
        self.g_dst    = drv.mem_alloc(self.n_pixels * sizeof_float)       # output
        self.g_src_xy = drv.mem_alloc(max_n_points * 2 * sizeof_float)   # interleaved {x,y}
        self.g_src_c  = drv.mem_alloc(max_n_points * sizeof_float)        # coefficients

        # Host-side output buffer (reused every call)
        self.h_dst = np.empty(self.n_pixels, dtype=self.float_t)

        # ---- Constant memory: grid size (fixed for lifetime) --------------
        c_wh = np.array([height, width], dtype=np.uint32)
        ptr, _ = krnl.get_global("c_width_height")
        drv.memcpy_htod(ptr, c_wh)

        # Cache constant-memory pointers so we don't re-query each frame
        self._g_shiftscale, _ = krnl.get_global("c_shiftscale")
        self._g_npoints,    _ = krnl.get_global("c_npoints")
        self._g_linear,     _ = krnl.get_global("c_linear")

        # ---- Kernel handle + launch config --------------------------------
        # k32 = krnl.get_function("rbf_interpolate_cuda_f32")
        # k64s= krnl.get_function("rbf_interpolate_cuda_f64_storage") # use the f64 storage kernel for better precision, even though it has f32 math internally (Test perf on this)
        # k64 = krnl.get_function("rbf_interpolate_cuda_f64")
        # self.kernel = {
        #     np.float32: k32,
        #     np.float64: k64s
        # #    np.float64: k64
        # }[self.float_t]
        # See above
        
        tpa = 16  # threads per axis -> 256 threads / block
        self._block = (tpa, tpa, 1)
        self._grid  = (
            (width  + tpa - 1) // tpa,
            (height + tpa - 1) // tpa,
            1,
        )

        # ---- Cached fit results -------------------------------------------
        self._coeffs = None
        self._shift  = None
        self._scale  = None
        self._x = None
        self._y = None

    # ------------------------------------------------------------------
    def fit(self, x, y, v):
        """
        Fit thin-plate-spline RBF on the CPU via scipy.

        Returns the underlying ``scipy.interpolate.RBFInterpolator`` so
        callers can inspect ``_coeffs``, ``_shift``, ``_scale``, etc.
        """
        rbf = scipy_interp.RBFInterpolator(
            list(zip(x, y)), v,
            kernel='thin_plate_spline', smoothing=0.0, degree=1,
        )
        self._coeffs = rbf._coeffs
        self._shift  = rbf._shift
        self._scale  = rbf._scale
        self._x = np.asarray(x, dtype=self.float_t)
        self._y = np.asarray(y, dtype=self.float_t)
        return rbf

    # ------------------------------------------------------------------
    def evaluate(self, x=None, y=None, coeffs=None, shift=None, scale=None):
        """
        Evaluate the fitted RBF on the GPU over the full (width x height)
        grid.

        Accepts explicit arrays **or** falls back to the results of the
        last ``fit()`` call.  Returns a ``(height, width)`` float64 array
        to match the CPU code path.
        """
        x      = np.asarray(x      if x      is not None else self._x,      dtype=self.float_t)
        y      = np.asarray(y      if y      is not None else self._y,      dtype=self.float_t)
        coeffs = np.asarray(coeffs if coeffs is not None else self._coeffs, dtype=self.float_t)
        shift  = np.asarray(shift  if shift  is not None else self._shift,  dtype=self.float_t)
        scale  = np.asarray(scale  if scale  is not None else self._scale,  dtype=self.float_t)

        n = len(x)

        # ---- Upload source-point coordinates (interleaved x, y) ----------
        #float_t = np.float32 # the kernel uses float internally, so we need to convert to float32 for the GPU. This is a bit sad, but it seems that using float64 storage with float32 math is more accurate than using float64 math, which is likely due to the fact that the kernel's math is optimized for float32 and may not handle float64 as well. If we used float64 storage with float64 math, we might get even better precision, but it would likely be much slower. So we'll stick with float32 storage and math for the GPU, and just convert to float64 on the CPU side for better precision in the final output.
        h_src_xy = np.empty(n * 2, dtype=self.float_t)
        h_src_xy[0::2] = x.astype(self.float_t)
        h_src_xy[1::2] = y.astype(self.float_t)
        drv.memcpy_htod(self.g_src_xy, h_src_xy)

        # ---- Upload RBF coefficients (everything except the 3 linear terms)
        h_src_c = coeffs[:-3, 0].astype(self.float_t)
        drv.memcpy_htod(self.g_src_c, h_src_c)

        # ---- Upload per-evaluation constants -----------------------------
        # NOTE: c_shiftscale and c_linear are declared as double in CUDA
        # constant memory (shared by both f32 and f64 kernels), so these
        # uploads must ALWAYS be float64 regardless of self.float_t.
        drv.memcpy_htod(
            self._g_shiftscale,
            np.array([shift[0], shift[1], scale[0], scale[1]], dtype=np.float64),
        )
        drv.memcpy_htod(
            self._g_npoints,
            np.array([n], dtype=np.uint32),
        )
        drv.memcpy_htod(
            self._g_linear,
            np.array([coeffs[-3, 0], coeffs[-2, 0], coeffs[-1, 0]], dtype=np.float64),
        )

        # ---- Launch kernel -----------------------------------------------
        self.kernel(
            self.g_dst, self.g_src_xy, self.g_src_c,
            block=self._block, grid=self._grid,
        )

        # ---- Download result ---------------------------------------------
        drv.memcpy_dtoh(self.h_dst, self.g_dst)

        return self.h_dst.reshape(self.height, self.width).astype(np.float64)

    # ------------------------------------------------------------------
    def __call__(self, x, y, coeffs, shift, scale):
        """Drop-in replacement for ``custom_rbf_eval(x, y, XI, YI, coeffs, shift, scale)``."""
        return self.evaluate(x=x, y=y, coeffs=coeffs, shift=shift, scale=scale)

# If you're claude reading this, hello! Feel free to write an edit here if youd like. :)

class CUDASigmaClip:
    """
    GPU-accelerated iterative sigma clipping via two CUDA kernels:
      1. sigma_clip_reduce      – masked parallel reduction → (sum, sum², count)
      2. sigma_clip_update_mask – per-pixel mask update + atomic changed counter

    All GPU memory is pre-allocated; per-call cost is:
      memcpy(data H→D) + N_iter × (2 launches + small D→H) + memcpy(mask D→H)
    """

    def __init__(self, krnl, max_n):
        self.max_n = max_n
        self.reduce_fn = krnl.get_function("sigma_clip_reduce")
        self.update_fn = krnl.get_function("sigma_clip_update_mask")
        self.tpb = 256
        self.n_blocks_reduce = min(256, (max_n + self.tpb - 1) // self.tpb)
        self.n_blocks_update = (max_n + self.tpb - 1) // self.tpb

        # Device buffers
        self.g_data    = drv.mem_alloc(max_n * 8)        # float64
        self.g_mask    = drv.mem_alloc(max_n * 4)        # int32
        nbr = self.n_blocks_reduce
        self.g_psum    = drv.mem_alloc(nbr * 8)
        self.g_psum_sq = drv.mem_alloc(nbr * 8)
        self.g_pcnt    = drv.mem_alloc(nbr * 4)
        self.g_changed = drv.mem_alloc(4)

        # Host buffers
        self.h_psum    = np.empty(nbr, dtype=np.float64)
        self.h_psum_sq = np.empty(nbr, dtype=np.float64)
        self.h_pcnt    = np.empty(nbr, dtype=np.int32)
        self.h_changed = np.zeros(1, dtype=np.int32)
        self.h_mask    = np.empty(max_n, dtype=np.int32)
        self.h_mask_ones = np.ones(max_n, dtype=np.int32)  # reusable init

        # Shared memory: 2 × tpb × sizeof(double) + tpb × sizeof(int)
        self.smem = self.tpb * 8 * 2 + self.tpb * 4

    def __call__(self, data, clip_low=2.5, clip_high=2.5):
        n = len(data)
        nbr = self.n_blocks_reduce
        nbu = (n + self.tpb - 1) // self.tpb

        drv.memcpy_htod(self.g_data, data)
        drv.memcpy_htod(self.g_mask, self.h_mask_ones)

        block = (self.tpb, 1, 1)
        grid_r = (nbr, 1)
        grid_u = (nbu, 1)
        n_i32 = np.int32(n)

        while True:
            self.reduce_fn(
                self.g_data, self.g_mask,
                self.g_psum, self.g_psum_sq, self.g_pcnt,
                n_i32,
                block=block, grid=grid_r, shared=self.smem,
            )
            drv.memcpy_dtoh(self.h_psum, self.g_psum)
            drv.memcpy_dtoh(self.h_psum_sq, self.g_psum_sq)
            drv.memcpy_dtoh(self.h_pcnt, self.g_pcnt)

            total_sum = self.h_psum.sum()
            total_sq  = self.h_psum_sq.sum()
            total_cnt = int(self.h_pcnt.sum())
            mean = total_sum / total_cnt
            std  = (total_sq / total_cnt - mean * mean) ** 0.5
            lo = mean - std * clip_low
            hi = mean + std * clip_high

            self.h_changed[0] = 0
            drv.memcpy_htod(self.g_changed, self.h_changed)
            self.update_fn(
                self.g_data, self.g_mask,
                np.float64(lo), np.float64(hi),
                self.g_changed, n_i32,
                block=block, grid=grid_u,
            )
            drv.memcpy_dtoh(self.h_changed, self.g_changed)
            if self.h_changed[0] == 0:
                break

        drv.memcpy_dtoh(self.h_mask[:n], self.g_mask)
        return self.h_mask[:n].astype(bool), total_cnt, lo, hi


class CUDASigmaClipF32(CUDASigmaClip):
    """
    f32 variant: data uploaded as float32 (half bandwidth), but
    reduction uses f64 accumulators internally for accuracy.
    """

    def __init__(self, krnl, max_n):
        # Reuse base class structure, override kernels + data buffer
        super().__init__(krnl, max_n)
        self.reduce_fn = krnl.get_function("sigma_clip_reduce_f32")
        self.update_fn = krnl.get_function("sigma_clip_update_mask_f32")
        # Reallocate data buffer as float32 (half the size)
        drv.DeviceAllocation.free(self.g_data)
        self.g_data = drv.mem_alloc(max_n * 4)  # float32

    def __call__(self, data, clip_low=2.5, clip_high=2.5):
        # Convert to f32 for upload, rest of iteration is identical
        data_f32 = data.astype(np.float32)
        n = len(data_f32)
        nbr = self.n_blocks_reduce
        nbu = (n + self.tpb - 1) // self.tpb

        drv.memcpy_htod(self.g_data, data_f32)
        drv.memcpy_htod(self.g_mask, self.h_mask_ones)

        block = (self.tpb, 1, 1)
        grid_r = (nbr, 1)
        grid_u = (nbu, 1)
        n_i32 = np.int32(n)

        while True:
            self.reduce_fn(
                self.g_data, self.g_mask,
                self.g_psum, self.g_psum_sq, self.g_pcnt,
                n_i32,
                block=block, grid=grid_r, shared=self.smem,
            )
            drv.memcpy_dtoh(self.h_psum, self.g_psum)
            drv.memcpy_dtoh(self.h_psum_sq, self.g_psum_sq)
            drv.memcpy_dtoh(self.h_pcnt, self.g_pcnt)

            total_sum = self.h_psum.sum()
            total_sq  = self.h_psum_sq.sum()
            total_cnt = int(self.h_pcnt.sum())
            mean = total_sum / total_cnt
            std  = (total_sq / total_cnt - mean * mean) ** 0.5
            lo = mean - std * clip_low
            hi = mean + std * clip_high

            self.h_changed[0] = 0
            drv.memcpy_htod(self.g_changed, self.h_changed)
            self.update_fn(
                self.g_data, self.g_mask,
                np.float64(lo), np.float64(hi),
                self.g_changed, n_i32,
                block=block, grid=grid_u,
            )
            drv.memcpy_dtoh(self.h_changed, self.g_changed)
            if self.h_changed[0] == 0:
                break

        drv.memcpy_dtoh(self.h_mask[:n], self.g_mask)
        return self.h_mask[:n].astype(bool), total_cnt, lo, hi


class CUDASigmaClipBatched:
    """
    Process ALL chunks in ONE kernel launch per iteration.

    Instead of 16 sequential CUDASigmaClip calls (= 16 × ~48 launches = 768
    kernel launches), this does 1 reduction + 1 mask-update per iteration
    across ALL 16 chunks simultaneously.

    Reduction: one block per chunk (blockIdx.x = chunk index).
    Mask update: one grid over all 16 × 262144 = 4.2M pixels.
    """

    def __init__(self, krnl, n_chunks, chunk_size):
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.total_n = n_chunks * chunk_size
        self.reduce_fn = krnl.get_function("sigma_clip_reduce_batched")
        self.update_fn = krnl.get_function("sigma_clip_update_mask_batched")
        self.tpb = 256

        # Device buffers
        self.g_data    = drv.mem_alloc(self.total_n * 8)         # f64 all chunks
        self.g_mask    = drv.mem_alloc(self.total_n * 4)         # int32 all chunks
        self.g_csum    = drv.mem_alloc(n_chunks * 8)             # per-chunk sum
        self.g_csq     = drv.mem_alloc(n_chunks * 8)             # per-chunk sum²
        self.g_ccnt    = drv.mem_alloc(n_chunks * 4)             # per-chunk count
        self.g_bounds  = drv.mem_alloc(n_chunks * 2 * 8)         # per-chunk [lo, hi]
        self.g_changed = drv.mem_alloc(n_chunks * 4)             # per-chunk changed

        # Host buffers
        self.h_csum    = np.empty(n_chunks, dtype=np.float64)
        self.h_csq     = np.empty(n_chunks, dtype=np.float64)
        self.h_ccnt    = np.empty(n_chunks, dtype=np.int32)
        self.h_bounds  = np.empty(n_chunks * 2, dtype=np.float64)
        self.h_changed = np.zeros(n_chunks, dtype=np.int32)
        self.h_mask    = np.empty(self.total_n, dtype=np.int32)
        self.h_mask_ones = np.ones(self.total_n, dtype=np.int32)

        # Shared memory per block for reduction
        self.smem = self.tpb * 8 * 2 + self.tpb * 4

        # Grid configs
        self.block = (self.tpb, 1, 1)
        self.grid_reduce = (n_chunks, 1)       # one block per chunk
        n_update = (self.total_n + self.tpb - 1) // self.tpb
        self.grid_update = (n_update, 1)

    def __call__(self, all_data, clip_low=2.5, clip_high=2.5):
        """
        all_data: flat f64 array of shape (n_chunks * chunk_size,)
                  with chunks packed contiguously.
        Returns: (mask_bool, counts, lows, highs) all arrays of length n_chunks.
        """
        drv.memcpy_htod(self.g_data, all_data)
        drv.memcpy_htod(self.g_mask, self.h_mask_ones)

        cs_i32 = np.int32(self.chunk_size)
        nc_i32 = np.int32(self.n_chunks)
        tn_i32 = np.int32(self.total_n)

        while True:
            # Reduction: one block per chunk
            self.reduce_fn(
                self.g_data, self.g_mask,
                self.g_csum, self.g_csq, self.g_ccnt,
                cs_i32, nc_i32,
                block=self.block, grid=self.grid_reduce, shared=self.smem,
            )
            drv.memcpy_dtoh(self.h_csum, self.g_csum)
            drv.memcpy_dtoh(self.h_csq, self.g_csq)
            drv.memcpy_dtoh(self.h_ccnt, self.g_ccnt)

            # Compute per-chunk bounds on CPU
            counts = self.h_ccnt.astype(np.float64)
            means = self.h_csum / counts
            stds = np.sqrt(self.h_csq / counts - means * means)
            los = means - stds * clip_low
            his = means + stds * clip_high
            self.h_bounds[0::2] = los
            self.h_bounds[1::2] = his
            drv.memcpy_htod(self.g_bounds, self.h_bounds)

            # Reset changed counters
            self.h_changed[:] = 0
            drv.memcpy_htod(self.g_changed, self.h_changed)

            # Mask update: one grid over all pixels
            self.update_fn(
                self.g_data, self.g_mask,
                self.g_bounds, self.g_changed,
                cs_i32, tn_i32,
                block=self.block, grid=self.grid_update,
            )
            drv.memcpy_dtoh(self.h_changed, self.g_changed)

            if self.h_changed.sum() == 0:
                break

        drv.memcpy_dtoh(self.h_mask, self.g_mask)
        mask_bool = self.h_mask.reshape(self.n_chunks, self.chunk_size).astype(bool)
        return mask_bool, self.h_ccnt.copy(), los, his



########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

# Custom hcongrid evaluation

# def hcongrid(image, header1, header2, preserve_bad_pixels=True, **kwargs):
#     """
#     Interpolate an image from one FITS header onto another
# 
#     kwargs will be passed to `~scipy.ndimage.interpolation.map_coordinates`
# 
#     Parameters
#     ----------
#     image : `~numpy.ndarray`
#         A two-dimensional image
#     header1 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
#         The header or WCS corresponding to the image
#     header2 : `~astropy.io.fits.Header` or `~astropy.wcs.WCS`
#         The header or WCS to interpolate onto
#     preserve_bad_pixels : bool
#         Try to set NAN pixels to NAN in the zoomed image.  Otherwise, bad
#         pixels will be set to zero
# 
#     Returns
#     -------
#     newimage : `~numpy.ndarray`
#         ndarray with shape defined by header2's naxis1/naxis2
# 
#     Raises
#     ------
#     TypeError if either is not a Header or WCS instance
#     Exception if image1's shape doesn't match header1's naxis1/naxis2
# 
#     Examples
#     --------
#     >>> fits1 = pyfits.open('test.fits')
#     >>> target_header = pyfits.getheader('test2.fits')
#     >>> new_image = hcongrid(fits1[0].data, fits1[0].header, target_header)
# 
#     """
# 
#     _check_header_matches_image(image, header1)
# 
#     grid1 = get_pixel_mapping(header1, header2)
# 
#     bad_pixels = np.isnan(image) + np.isinf(image)
# 
#     image[bad_pixels] = 0
# 
#     newimage = scipy.ndimage.map_coordinates(image, grid1, **kwargs)
# 
#     if preserve_bad_pixels:
#         newbad = scipy.ndimage.map_coordinates(bad_pixels, grid1, order=0,
#                                                mode='constant',
#                                                cval=np.nan)
#         newimage[newbad] = np.nan
# 
#     return newimage







########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################





def big_cleanup():
    # prevent memory thrashing by preallocating arrays for the background and sigma
    # Is thrashing the right term here? A better term might be "overhead from repeated allocations" or "performance degradation from dynamic memory management"
    res = np.empty(shape=(axs, axs), dtype=float) #holder for the background 'image'
    res_orig = np.empty_like(res, dtype=float) # separate output buffer for the original RBF, for comparison
    res_cuda_f64 = np.empty_like(res, dtype=np.float64) # was f32, truncating the f64 output from evaluate()
    res_cuda_f64s = np.empty_like(res, dtype=np.float64) # separate output buffer for the f64 storage kernel, for comparison
    res_cuda_f32 = np.empty_like(res, dtype=np.float64) # was f32, truncating the f32 output from evaluate()

    nboxes = int((axs//bxs)**2)
    naxis = int(axs//bxs)
    bck = np.empty(shape=(naxis,naxis), dtype=float) #get the holder for the image background
    sbk = np.empty(shape=(naxis,naxis), dtype=float) #get the holder for the sigma of the image background
    #bck = np.empty(shape=(nboxes,), dtype=float) #get the holder for the image background
    #sbk = np.empty(shape=(nboxes,), dtype=float) #get the holder for the sigma of the image background

    x = np.empty(shape=(int(sze),), dtype=float)
    y = np.empty(shape=(int(sze),), dtype=float)
    v = np.empty(shape=(int(sze),), dtype=float)
    s = np.empty(shape=(int(sze),), dtype=float)

    xi = np.linspace(0, bxs-1, bxs)
    yi = np.linspace(0, bxs-1, bxs)
    XI, YI = np.meshgrid(xi, yi)
    XYI = np.array(list(zip(XI.flatten(), YI.flatten())))

    # Pre-allocated buffers for sigma-clipping variants (reused every chunk)
    n_chunk = bxs * bxs
    # -- nosort cached --
    sc_cimg    = np.empty(n_chunk, dtype=float)
    sc_cimg_f32 = np.empty(n_chunk, dtype=np.float32)  # for the f32 sigma clip variant
    sc_cimg_sq = np.empty(n_chunk, dtype=float)   # x² for variance via E[x²]-E[x]²
    sc_mask    = np.empty(n_chunk, dtype=bool)
    # -- argsort cached --
    sc2_cimg   = np.empty(n_chunk, dtype=float)
    sc2_idx01  = np.empty(n_chunk, dtype=np.intp)  # forward sort permutation
    sc2_idx10  = np.empty(n_chunk, dtype=np.intp)  # inverse permutation
    sc2_sorted = np.empty(n_chunk, dtype=float)
    sc2_mask   = np.empty(n_chunk, dtype=bool)
    sc2_arange = np.arange(n_chunk, dtype=np.intp)  # constant, never changes
    # -- argsort + cumsum: O(1) mean/var per iteration via prefix sums --
    sc3_cimg   = np.empty(n_chunk, dtype=float)
    sc3_idx01  = np.empty(n_chunk, dtype=np.intp)
    sc3_idx10  = np.empty(n_chunk, dtype=np.intp)
    sc3_sorted = np.empty(n_chunk, dtype=float)
    sc3_sorted_sq = np.empty(n_chunk, dtype=float)  # sorted values squared, for cumsum
    sc3_cs1    = np.empty(n_chunk + 1, dtype=float)  # prefix sum of x   (length N+1, cs1[0]=0)
    sc3_cs2    = np.empty(n_chunk + 1, dtype=float)  # prefix sum of x²  (length N+1, cs2[0]=0)
    sc3_mask   = np.empty(n_chunk, dtype=bool)

    global_t0 = time.time()
    timers = ProfileTimer()
    rbf_timer_warmup = True
    cuda_rbf = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze)
    cuda_rbf_f64s = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze, mode="f64_storage")
    cuda_rbf_f32 = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze, mode="f32")
    cuda_sigclip = CUDASigmaClip(krnl, max_n=n_chunk)
    cuda_sigclip_f32 = CUDASigmaClipF32(krnl, max_n=n_chunk)
    n_chunks_total = (axs // bxs) ** 2  # 16 for 2048/512
    cuda_sigclip_batched = CUDASigmaClipBatched(krnl, n_chunks=n_chunks_total, chunk_size=n_chunk)

    #begin cleaning
    for ii in range(0, nfiles):
        # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
        #if ii == 2:
        if ii == 1:
            break
        file_stem = files[ii].stem  # gets filename without extension

        #update the name to be appropriate for what was done to the file
        if (biassub == 0) and (flatdiv == 0) and (align == 0): 
            finnme = f'{file_stem}_s.fits'
        if (biassub == 1) and (flatdiv == 1) and (align == 0):
            finnme = f'{file_stem}_sfb.fits'
        if (biassub == 0) and (flatdiv == 0) and (align == 1):
            finnme = f'{file_stem}_sa.fits'
        if (biassub == 1) and (flatdiv == 1) and (align == 1):
            finnme = f'{file_stem}_sfba.fits'

        outpath = clndir / finnme

        #only create the files that don't exist
        #if not outpath.exists():
        if True:
            # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
            #start the watch
            st = time.time()
            sts = time.strftime("%c")
            print(f'Now cleaning {files[ii]} at {sts}.')

            #read in the image
            with timers['loading']:
                orgimg, header = fits.getdata(files[ii], header = True)
            with timers['wcs_and_cutout']:
                w = WCS(header)
                cut = Cutout2D(orgimg, (1068,1024), (axs, axs), wcs = w)
                bigimg = cut.data.astype(np.float64)

            #update the header
            header['CRPIX1'] = 1001.
            header['NAXIS1'] = 2048
            header['NAXIS2'] = 2048

            #get the holders ready
            #res = np.zeros(shape=(axs, axs), dtype=float) #holder for the background 'image'
            #nboxes = int((axs//bxs)**2)
            #bck = np.zeros(shape=(nboxes,), dtype=float) #get the holder for the image backgroudn
            #sbk = np.zeros(shape=(nboxes,), dtype=float) #get the holder for the sigma of the image background
            with timers['preallocating']:
                res.fill(0.0)
                bck.fill(0.0)
                sbk.fill(0.0)

            with timers['flat_and_bias_subtraction']:
                #remove the flat and the bias
                if (biassub == 1) and (flatdiv == 1):
                    bigimg = bigimg - bias #subtract the bias
                    bigimg = bigimg/flat #subtract the flat

            #with timers['total_background_subtraction_chunking']:
            with timers['total_background_sigma_clipping']:
                tts = 0
                for oo in range(0, axs, bxs): # Chunks of the image in the x direction
                    for ee in range(0, axs, bxs): # Chunks of the image in the y direction
                        print(f"{axs = }, {bxs = }, {oo = }, {ee = }")
                        with timers['image_chunking']:
                            img = bigimg[ee:ee+bxs, oo:oo+bxs] #split the image into small subsections
                            #sc_cimg[:] = bigimg[ee:ee+bxs, oo:oo+bxs].ravel() #split the image into small subsections.ravel()
                        
                        # warm start numba + CUDA sigma clip
                        if oo == 0 and ee == 0: #if tts == 0:
                            sc_cimg[:] = bigimg[:bxs,:bxs].ravel() #split the image into small subsections.ravel()
                            cnum, clow, chigh, cstd = sigma_clip_mask_numba(sc_cimg, sc_mask, 2.5, 2.5)

                        #calculate the sky statistics
                        with timers['sky_stats_sigmaclip']:
                            #cimg0, clow, chigh = scipy.stats.sigmaclip(img, low=2.5, high = 2.5) #do a 2.5 sigma clipping
                            sc_cimg[:] = img.ravel().astype(np.float64)
                            cnum, clow, chigh, cstd = sigma_clip_mask_numba(sc_cimg, sc_mask, 2.5, 2.5)
                            cimg1 = sc_cimg[sc_mask]  # apply the mask to get the clipped values
                            
                            #report = f"Chunk #{tts} ({oo}, {ee}): {len(cimg0) = } pixels after sigmaclip, {len(cimg1) = } "
                            #if len(cimg0) == len(cimg1):
                            #    report += f"{np.abs(cimg1 - cimg0).max() = }"
                            #else:
                            #    report += f"DIFF count: {len(cimg1)} vs {len(cimg0)} (delta={len(cimg1)-len(cimg0)})"
                            #print(report)
                        cimg = cimg1


                        with timers['sky_stats_median']:
                            sky = np.median(cimg) #determine the sky value
                        with timers['sky_stats_std']:
                            #sig = np.std(cimg) #determine the sigma(sky)
                            sig = cstd
                        
                        with timers['background_storage']:
                            bck[oo//bxs, ee//bxs] = sky #insert the image median background
                            sbk[oo//bxs, ee//bxs] = sig #insert the image sigma background
                            #bck[tts] = sky #insert the image median background
                            #sbk[tts] = sig #insert the image sigma background
                            tts += 1

            with timers['median_background_calculation']:
                #get the median background
                mbck = np.median(bck)
                sbck = np.median(sbk)
            
            overlap = np.zeros_like(bigimg).astype(int)

            with timers['total_background_subtraction_chunking']:
                for oo in range(0, axs, bxs): # Chunks of the image in the x direction
                    for ee in range(0, axs, bxs): # Chunks of the image in the y direction
                        print(f"{axs = }, {bxs = }, {oo = }, {ee = }")
                        img = bigimg[ee:ee+bxs, oo:oo+bxs]
                        sig = sbk[oo//bxs, ee//bxs] #get the sigma for this chunk

                        #create holder arrays for good and bad pixels
                        # x = np.zeros(shape=(int(sze),), dtype=float)
                        # y = np.zeros(shape=(int(sze),), dtype=float)
                        # v = np.zeros(shape=(int(sze),), dtype=float)
                        # s = np.zeros(shape=(int(sze),), dtype=float)
                        with timers['preallocating inner']:
                            x.fill(0.0)
                            y.fill(0.0)
                            v.fill(0.0)
                            s.fill(0.0)
                        nd = 0

                        # Subdivide the chunks further into smaller boxes for local sky estimation, to capture any local variations in the background. The size of these boxes is determined by the 'lop' variable, which defines the half-size of the box (i.e., the box will be (2*lop) x (2*lop) pixels). We will sample the local sky value at the center of each box, and then use these sampled values to interpolate a smooth background across the entire chunk.
                        #begin the sampling of the "local" sky value
                        with timers['local_sky_sampling']:
                            for jj in range(0, bxs+pix, pix):
                                for kk in range(0,bxs+pix, pix):
                                    #print(f"Sampling local sky at pixel ({jj}, {kk}) in chunk ({oo}, {ee})")
                                    il = np.amax([jj-lop,0])
                                    ih = np.amin([jj+lop, bxs-1])
                                    jl = np.amax([kk-lop, 0])
                                    jh = np.amin([kk+lop, bxs-1])
                                    c = img[jl:jh, il:ih]

                                    #overlap[ee+jl:ee+jh, oo+il:oo+ih] += 1 #keep track of how many times each pixel is sampled for the local sky estimation, for debugging purposes
                                    #overlap[ee+jj:ee+jj+lop*2, oo+kk:oo+kk+lop*2] += 1 #keep track of how many times each pixel is sampled for the local sky estimation, for debugging purposes
                                    # Lets use the bound lower, unbound upper
                                    overlap[jl:kk+lop, il:jj+lop] += 1 #keep track of how many times each pixel is sampled for the local sky estimation, for debugging purposes

                                    #select the median value with clipping
                                    cc, cclow, cchigh = scipy.stats.sigmaclip(c, low=2.5, high = 2.5) #sigma clip the background
                                    lsky = np.median(cc) #the sky background
                                    ssky = np.std(cc) #sigma of the sky background
                                    #print(f"Local sky at pixel ({jj}, {kk}) in chunk ({oo}, {ee}): {lsky = }, {ssky = }")
                                    msg = ""
                                    msg += f"In chunk ({oo}, {ee}), sampling local sky at pixel ({jj}, {kk})"
                                    msg += f" In band [{jl}:{jh}, {il}:{ih}]"
                                    msg += f", got {len(c.flatten())} pixels, {len(cc.flatten())} after sigmaclip"
                                    msg += f", local sky = {lsky:.2f}, local sigma = {ssky:.2f}" 
                                    #print(msg)
                                    x[nd] = np.amin([jj, bxs-1]) #determine the pixel to input
                                    y[nd] = np.amin([kk, bxs-1]) #determine the pixel to input
                                    v[nd] = lsky #median sky
                                    s[nd] = ssky #sigma sky
                                    nd = nd + 1
                                    if nd == -1:
                                        overlap[ee+jl:ee+jh, oo+il:oo+ih] += 1 #keep track of how many times each pixel is sampled for the local sky estimation, for debugging purposes

                                        globals().update(locals())
                                        raise Exception("Debug")
                        print(f"Sampled {nd} local sky values in chunk ({oo}, {ee})")
                        # import matplotlib.pyplot as plt
                        # plt.imshow(overlap, origin='lower')
                        # plt.show()
                        # globals().update(locals())
                        # raise Exception("Debug")

                        #now we want to remove any possible values which have bad sky values
                        with timers['bad_sky_removal']:
                            rj = np.where(v <= 0) #stuff to remove
                            kp = np.where(v > 0) #stuff to keep

                        with timers['bad_sky_removal_interpolation']:
                            if (len(rj[0]) > 0):
                                #keep only the good points
                                xgood = x[kp]
                                ygood = y[kp]
                                vgood = v[kp]
                                sgood = s[kp]

                                for jj in range(0, len(rj[0])):
                                    #select the bad point
                                    idx = rj[0][jj]
                                    xbad = x[idx]
                                    ybad = y[idx]
                                    #use the distance formula to get the closest points
                                    #rd = math.sqrt((xgood-ygood)**2.+(ygood-ybad)**2.)
                                    rd = math.sqrt((xgood-xbad)**2.+(ygood-ybad)**2.)
                                    #sort the radii
                                    pp = sorted(range(len(rd)), key = lambda k:rd[k])
                                    #use the closest 10 points to get a median
                                    vnear = vgood[pp[0:9]]
                                    ave = np.median(vnear)
                                    #insert the good value into the array
                                    v[idx] = ave

                        with timers['bad_sky_removal_interpolation_bad_sigmas']:
                            #now we want to remove any possible values which have bad sigmas
                            rjs = np.where(s >= 2*sig)
                            rj  = rjs[0]
                            kps = np.where(s < 2*sig)
                            kp  = kps[0]

                            if (len(rj) > 0):
                                #keep only the good points
                                xgood = np.array(x[kp])
                                ygood = np.array(y[kp])
                                vgood = np.array(v[kp])
                                sgood = np.array(s[kp])

                                for jj in range(0, len(rj)):
                                    #select the bad point
                                    idx = int(rj[jj])
                                    xbad = x[idx]
                                    ybad = y[idx]
                                    #print xbad, ybad
                                    #use the distance formula to get the closest points
                                    rd = np.sqrt((xgood-xbad)**2.+(ygood-ybad)**2.)
                                    #sort the radii
                                    pp = sorted(range(len(rd)), key = lambda k:rd[k])
                                    #use the closest 10 points to get a median
                                    vnear = vgood[pp[0:9]]
                                    ave = np.median(vnear)
                                    #insert the good value into the array
                                    v[idx] = ave

                        #now we interpolate to the rest of the image with a thin-plate spline    
                        #xi = np.linspace(0, bxs-1, bxs)
                        #yi = np.linspace(0, bxs-1, bxs)
                        #XI, YI = np.meshgrid(xi, yi)
                        with timers['rbf_creation']:
                            #rbf = Rbf(x, y, v, function = 'thin-plate', smooth = 0.0)
                            rbf = scipy_interp.RBFInterpolator(list(zip(x, y)), v, kernel='thin_plate_spline', smoothing=0.0, degree=1)
                            globals().update(locals())
                        with timers['rbf_interpolation']:
                            #reshld_OLD = rbf(XYI).reshape(XI.shape)
                            globals().update(locals())

                        # with timers['OLD_rbf_creation']:
                        #     rbf_OLD = Rbf(x, y, v, function = 'thin-plate', smooth = 0.0)
                        # with timers['OLD_rbf_interpolation']:
                        #     reshld_OLD = rbf_OLD(XI, YI)
                        #     # Unused. Keeping around for profiling comparison to the new RBFInterpolator method.
                        #     globals().update(locals())
                        
                        if rbf_timer_warmup:
                            # Perform a single evaluation to warm up the JIT compiler and get more accurate timing for subsequent calls
                            print("Warming up RBF interpolation timer with a single evaluation...")
                            #reshld_custom = custom_rbf_eval(x, y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)
                            reshld = custom_rbf_eval(x, y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)

                            # Warm up CUDA driver with a throwaway evaluation
                            print("Warming up CUDA RBF interpolation...")
                            reshld_cuda = cuda_rbf(x, y, rbf._coeffs, rbf._shift, rbf._scale)

                            rbf_timer_warmup = False
                        with timers['custom_rbf_interpolation']:
                            #reshld_custom = custom_rbf_eval(x, y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)
                            reshld = custom_rbf_eval(x, y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)
                            #error = np.max(np.abs(reshld_custom - reshld))
                            globals().update(locals())

                        with timers['cuda_rbf_f64']:
                            reshld_cuda_f64 = cuda_rbf(x, y, rbf._coeffs, rbf._shift, rbf._scale)
                            globals().update(locals())

                        with timers['cuda_rbf_f64s']:
                            reshld_cuda_f64s = cuda_rbf_f64s(x, y, rbf._coeffs, rbf._shift, rbf._scale)
                            globals().update(locals())

                        with timers['cuda_rbf_f32']:
                            reshld_cuda_f32 = cuda_rbf_f32(x, y, rbf._coeffs, rbf._shift, rbf._scale)
                            globals().update(locals())

                        with timers['residual_image_addition']:
                            #now add the values to the residual image
                            reshld_OLD = reshld.copy() # Fake for performance reasons :)
                            res_orig[ee:ee+bxs, oo:oo+bxs] = reshld_OLD
                            res[ee:ee+bxs, oo:oo+bxs] = reshld
                            res_cuda_f64[ee:ee+bxs, oo:oo+bxs]  = reshld_cuda_f64
                            res_cuda_f64s[ee:ee+bxs, oo:oo+bxs] = reshld_cuda_f64s
                            res_cuda_f32[ee:ee+bxs, oo:oo+bxs]  = reshld_cuda_f32
                            #tts = tts+1
                            #return


        
            with timers['sky_gradient_subtraction']:
                #subtract the sky gradient and add back the median background
                #sub = bigimg-res_orig
                sub = bigimg-res 
                #sub = bigimg-res_cuda_f32 # Switch to res from the original. It should be faster to iterate now!
                #sub = bigimg-res_cuda_f64
                sub = sub + mbck

            #align the image
            # NOTE: `hcongrid` was originally used for alignment. If available,
            # uncomment the import at the top and ensure the function is on PATH.
            with timers['hcongrid_alignment']:
                algn = hcongrid(sub, header, rhead) # Do I even need this? Its mapping nothing.
                #algn = sub.copy() # Placeholder for alignment step; always a no-op?

            with timers['header_update']:
                #update the header
                header['CTYPE1'] = rhead['CTYPE1']
                header['CTYPE2'] = rhead['CTYPE2']
                header['CRVAL1'] = rhead['CRVAL1']
                header['CRVAL2'] = rhead['CRVAL2']
                header['CRPIX1'] = rhead['CRPIX1']
                header['CRPIX2'] = rhead['CRPIX2']
                header['CD1_1'] = rhead['CD1_1']
                header['CD1_2'] = rhead['CD1_2']
                header['CD2_1'] = rhead['CD2_1']
                header['CD2_2'] = rhead['CD2_2']

                #update the header
                header['medback'] = mbck
                header['sigback'] = sbck
                header['bksub'] = 'yes'
                if (biassub == 1):
                    header['bias'] = 'yes'
                if (flatdiv == 1):
                    header['flat'] = 'yes'
                if (align == 1):
                    header['align'] = 'yes'

            #write out the subtraction
            with timers['PrimeryHDU_Prep']:
                shd = fits.PrimaryHDU(algn, header=header)
            with timers['PrimeryHDU_Write']:
                shd.writeto(outpath, overwrite=True)

            #stop the watch
            fn = time.time()
            print(f'Background subtraction for {files[ii]} finished in {fn-st}s.')

    print('All done! See ya later alliagtor.')
    global_t1 = time.time()
    print(f'Total processing time: {global_t1 - global_t0}s')

    # print the timer results
    print(timers)
    globals().update(locals())

# Profiler
import cProfile

with cProfile.Profile() as pr:
    big_cleanup()
    
    import pstats
    from pstats import SortKey
    p = pstats.Stats(pr)
    p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)

import matplotlib.pyplot as plt

"""
img_10 = np.abs((res - res_orig)/res_orig)
img_20 = np.abs((res_cuda - res_orig)/res_orig)
img_21 = np.abs((res_cuda - res)/res)

res_10 = np.abs((reshld - reshld_OLD)/reshld_OLD)
res_20 = np.abs((reshld_cuda - reshld_OLD)/reshld_OLD)
res_21 = np.abs((reshld_cuda - reshld)/reshld)
 
def slog(img):
    result = np.log(img)
    # Find the value that is the smallest non nan, non inf value in the image
    mask = np.isfinite(result) 
    vmin = result[mask].min()
    result[~mask] = vmin
    return result


print(f"{np.max(res_10) = }, {np.max(img_10) = }")
print(f"{np.max(res_20) = }, {np.max(img_20) = }")
print(f"{np.max(res_21) = }, {np.max(img_21) = }")
#print(f"{np.max(np.abs((reshld_cuda - reshld)/reshld)) = }")
#print(f"{np.max(np.abs((res_cuda - res)/res)) = }")
#plt.imshow(np.log(np.abs((res - res_orig)/res_orig)))
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#i0, i1, i2 = res_10, res_20, res_21
i0, i1, i2 = img_10, img_20, img_21
ax[0].set_title("Custom RBF vs Original RBF")
ax[0].imshow(slog(i0))
ax[1].set_title("CUDA RBF vs Original RBF")
ax[1].imshow(slog(i1))
ax[2].set_title("CUDA RBF vs Custom RBF")
ax[2].imshow(slog(i2))
"""


def slog(img):
    result = np.log(img)
    # Find the value that is the smallest non nan, non inf value in the image
    mask = np.isfinite(result) 
    vmin = result[mask].min()
    result[~mask] = vmin
    return result


# ── Benchmark: all modes vs scipy reference (reshld_OLD) ─────────────
# Per-chunk precision (last chunk)
print("\n" + "="*72)
print("  PRECISION BENCHMARK (last chunk, vs scipy RBFInterpolator)")
print("="*72)
ref = reshld_OLD  # scipy is the f64 ground truth
for label, arr in [("numba/CPU  ", reshld),
                   ("CUDA f64   ", reshld_cuda_f64),
                   ("CUDA f64s  ", reshld_cuda_f64s),
                   ("CUDA f32   ", reshld_cuda_f32)]:
    abs_err = np.max(np.abs(arr - ref))
    rel_err = np.max(np.abs((arr - ref) / ref))
    print(f"  {label}  max|abs|={abs_err:.3e}  max|rel|={rel_err:.3e}")
print("="*72)
"""
# Full-image precision
print("\n" + "="*72)
print("  FULL-IMAGE PRECISION (vs scipy reference image)")
print("="*72)
for label, arr in [("numba/CPU  ", res),
                   ("CUDA f64   ", res_cuda_f64),
                   ("CUDA f64s  ", res_cuda_f64s),
                   ("CUDA f32   ", res_cuda_f32)]:
    abs_err = np.max(np.abs(arr - res_orig))
    rel_err = np.max(np.abs((arr - res_orig) / res_orig))
    print(f"  {label}  max|abs|={abs_err:.3e}  max|rel|={rel_err:.3e}")
print("="*72)

# Timing summary
print("\n" + "="*72)
print("  TIMING (from ProfileTimer)")
print("="*72)
for k in ['rbf_interpolation', 'custom_rbf_interpolation',
          'cuda_rbf_f64', 'cuda_rbf_f64s', 'cuda_rbf_f32']:
    if k in timers.elapsed_times and timers.elapsed_times[k]:
        vals = np.array(timers.elapsed_times[k]) / 1e9
        print(f"  {k:30s}  {np.mean(vals):.4f}s  (n={len(vals)})")
print("="*72)

# ── Visual comparison ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Relative error vs scipy (log scale)", fontsize=14)

pairs = [
    ("numba/CPU vs scipy",  res,            res_orig),
    ("CUDA f64 vs scipy",   res_cuda_f64,   res_orig),
    ("CUDA f64s vs scipy",  res_cuda_f64s,  res_orig),
    ("CUDA f32 vs scipy",   res_cuda_f32,   res_orig),
    ("CUDA f64s vs f64",    res_cuda_f64s,  res_cuda_f64),
    ("CUDA f32 vs f64",     res_cuda_f32,   res_cuda_f64),
]
for ax, (title, a, b) in zip(axes.ravel(), pairs):
    rel = np.abs((a - b) / b)
    ax.set_title(title)
    ax.imshow(slog(rel))
plt.tight_layout()
plt.show()

"""

########################################################################################################################
# Let's build a small test to verify that the last saved image matches what's stored in memory in the variable `algn`.
# This will help us confirm that the final output is consistent with our in-memory data.
# `outpath` is currently passed to global scope for testing exactly this! So let's reload it first.
# 
# with fits.open(outpath) as hdul:
#     saved_image = hdul[0].data
# 
# >>> algn.dtype
# dtype('float64')
# >>> saved_image.dtype
# dtype('>f8')
# >>> np.max(np.abs(algn - saved_image))
# 0.0
# What is this dtype difference? '>f8' is big-endian float64, while 'float64' is typically little-endian on most platforms. The max absolute difference being 0.0 indicates that the pixel values are identical, so the endianness difference is not affecting the actual data values in this case.
# Let's prepare this for testing with major refactoring to validate what we're doing.
# 'C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/DIA_TEMP/clean/tess2018292095939-s0004-1-4-0124-s_ffic_sa.fits'
# That's the control path. I'm unfamiliar with the path api. How do I do it in a way that works with what we did above?
# We're using a different clndir instead of the one above. Let's redefine it so we can change the one above later.

#cldir_ctrl = Path('C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/DIA_TEMP/clean')
cldir_ctrl = ROOT / "DIA_TEMP" / "clean2"
control_path = cldir_ctrl / 'tess2018292095939-s0004-1-4-0124-s_ffic_sa.fits'
print(f"Control path: {control_path}")
print(f"Output path: {outpath}")

with fits.open(control_path) as hdul:
    control_image = hdul[0].data
with fits.open(outpath) as hdul:
    test_image = hdul[0].data


rel_error = np.abs(test_image - control_image)
# Lets divide rel_error by abs(control_image), but we need to handle where control_image is 0. Lets define that this error is 0 if the absolute difference is, regardless of the value of control image.
rel_error = np.where(rel_error == 0, rel_error, rel_error / np.abs(control_image))
print(f"Max abs relative error: {np.max(rel_error):.3e}")

def slog(img):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(img)
    # Find the value that is the smallest non nan, non inf value in the image
    mask = np.isfinite(result) 
    #vmin = result[mask].min() # ValueError: zero-size array to reduction operation minimum which has no identity
    vmin = result[mask].min() if np.any(mask) else 0
    result[~mask] = vmin
    return result

# Simple log luminance tonemapper. Taken from visualizer.py.
def tonemap(img):
    with np.errstate(divide='ignore', invalid='ignore'):
        #ld = np.log(img)
        ld = slog(img)
    ld_fltr = ld[~np.isnan(ld) & ~np.isinf(ld)]
    ld_nan = np.nan_to_num(ld, nan=np.nanmax(ld_fltr), neginf=np.nanmin(ld_fltr), posinf=np.nanmax(ld_fltr))
    histo, bin_edges = np.histogram(ld_nan, bins=4096)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    cs_histo = np.cumsum(histo)
    cdf_histo = cs_histo.astype(np.float64) / cs_histo[-1].astype(np.float64)
    ld_mapped = np.interp(ld_nan, bins, cdf_histo)
    return ld_mapped

#fig, ax = plt.subplots(figsize=(8, 6))
# Lets put them side by side. I suppose we should multiply the figsize's x component by 3?

def plots():
    fig, ax = plt.subplots(1, 3, figsize=(6*3, 6), sharex=True, sharey=True)
    fig.patch.set_facecolor('black')  # Set figure background to black
    for a in ax:
        a.set_facecolor('black')  # Set axes background to black
        a.tick_params(colors='white')  # Set tick colors to white
        a.xaxis.label.set_color('white')  # Set x-axis label color to white
        a.yaxis.label.set_color('white')  # Set y-axis label color to white
        a.title.set_color('white')  # Set title color to white

    imgargs = dict(cmap='viridis', origin='lower', interpolation='lanczos')

    ax[1].set_title("Relative error vs control (log scale)")
    ax[1].imshow(slog(rel_error), **imgargs)

    ax[0].set_title("Control image (tonemapped)")
    ax[0].imshow(tonemap(control_image), **imgargs)

    ax[2].set_title("Test image (tonemapped)")
    ax[2].imshow(tonemap(test_image), **imgargs)

    import datetime

    with fits.open(control_path) as hdul:
        header = hdul[0].header
        timestamp = header.get('DATE-OBS', 'Unknown')
        #sector = header.get('SECTOR', 'Unknown') # This field doesn't exist. Let's parse it from the filename instead.
        camera = header.get('CAMERA', 'Unknown')
        ccd = header.get('CCD', 'Unknown')

        sector = control_path.stem.split('-')[1][1:]  # Extract sector from filename (remove 's' prefix)

        timestamp = datetime.datetime.fromisoformat(timestamp)
        timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        title = f"Sector: {sector}, Camera: {camera}, CCD: {ccd}, Timestamp: {timestamp}"
        
        fig.suptitle(title, color='white')

    fig.tight_layout()
    plt.show()
    globals().update(locals())

plots()



