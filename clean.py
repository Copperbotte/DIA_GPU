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
import scipy.linalg #import lu_factor, lu_solve # to nuke scipy perf in rbf construction [I prefer using scipy.linalg.lu_factor instead of importing it "pure" :) ]

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
    def __init__(self, skipto=1):
        self.keys = []
        self.start_times = {}
        self.elapsed_times = {}
        self.skipto = skipto

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
        st = self.skipto
        return {key: [np.mean(elapsed[st:]), np.std(elapsed[st:]), len(elapsed)] for key, elapsed in secs.items()}  # Return average time for each key
        #return {key: [np.mean(elapsed[1:]), np.std(elapsed[1:]), len(elapsed)-1] for key, elapsed in secs.items()}  # Return average time for each key
    
    def __str__(self):
        elapsed_times = self.get_elapsed_times()
        # pre-stringify all parameters for better formatting
        keys = list(elapsed_times.keys())
        params = []
        for key in keys:
            val, std, count = elapsed_times[key]
            # make numpy shush warnings
            with np.errstate(divide='ignore', invalid='ignore'):
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

    # Exports the elapsed *samples* directly.
    # This is useful for plotting distributions of times, not just summary stats.
    def export_json(self, filename):
        import json
        with open(filename, 'w', newline='') as csvfile:
            json.dump({key: [elapsed / 1e9 for elapsed in elapsed_list] for key, elapsed_list in self.elapsed_times.items()}, csvfile, indent=4)
        

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

cu_code += r"""

__device__ // This variant will only ever be used as an interface for the function below, and the newton iterator.
void vec2_distort_sip_cuda_f64(
    double* __restrict__ distort,
    double* __restrict__ uv,
    double* __restrict__ AB
) {
    // Manually construct the power sets
    double up[5];
    double vp[5];
    up[0] = 1.0;
    vp[0] = 1.0;
    for(int i=1; i<5; i++)
    {
        up[i] = uv[0] * up[i - 1];
        vp[i] = uv[1] * vp[i - 1];
    }

    // Einsum
    double dist_u = 0.0;
    double dist_v = 0.0;
    for(int k=2; k<5; k++) // 2, 3, 4
    {
        for(int i=0; i<=k; i++) // 0..k
        {
            int j = k - i;
            double uvp = up[i]*vp[j];
            dist_u += AB[0 * 25 + i * 5 + j] * uvp;
            dist_v += AB[1 * 25 + i * 5 + j] * uvp;
        }
    }
    distort[0] = dist_u;
    distort[1] = dist_v;
}

__device__
void vec2_distort_sip_inv_newton_cuda_f64(
    double* __restrict__ result,
    double* __restrict__ out_i,
    double* __restrict__ AB,
    double* __restrict__ iAB
) {
    // # ── Warm start: AP/BP inverse polynomial ──    
    double distort[2];
    vec2_distort_sip_cuda_f64(distort, out_i, iAB);

    double vb[2] = {
        out_i[0] + distort[0],
        out_i[1] + distort[1]
    }; // This is the initial guess for vb, which is out + inverse_SIP(out). We compute inverse_SIP(out) using the iAB coefficients, which are the coefficients of the inverse polynomial fit.

    // iter<2 -> 2.27373675e-12? weird.
    // iter<3 -> 3e-12
    // iter<4 -> 2e-12 (Likely hit limitations of double precision arithmetic here, not convergence)
    for(int iter=0; iter<3; iter++) // Empirically, this has a max error of 3e-12. Machine precision! I'll take it.
    {
        // Build power arrays at current guess
        double up[5+2]; // Pad by 2 to avoid out-of-bounds when computing derivatives. We'll just ignore the last two elements.
        double vp[5+2];
        up[1+0] = 1.0;
        vp[1+0] = 1.0;
        for(int i=1; i<5; i++)
        {
            up[1+i] = vb[0] * up[1+i - 1]; // I'm using an explicit 1+ to keep the pad in mind. The 1+i-1 should constexpr away.
            vp[1+i] = vb[1] * vp[1+i - 1];
        }
        up[0] = 0.0; up[5+1] = 0.0;  // pad: ensures 0*up[0]=0 even if it were NaN
        vp[0] = 0.0; vp[5+1] = 0.0;
        
        // Evaluate SIP_fwd and its 2×2 Jacobian simultaneously
        double f_uv[2] = {0.0, 0.0}; // {SIP_fwd_u, SIP_fwd_v}
        // J = I + dSIP/dvb, so we accumulate only the dSIP part
        double J_ij[4] = {0.0, 0.0, 0.0, 0.0}; // {dSIP_u/du, dSIP_u/dv, dSIP_v/du, dSIP_v/dv} = {J[0,0], J[0,1], J[1,0], J[1,1]} = {J[0], J[1], J[2], J[3]} for convenience

        for(int k=2; k<5; k++) // 2, 3, 4
        {
            for(int i=0; i<=k; i++) // 0..k
            {
                int j = k - i; // because i+j=k. 
                double uv_p = up[1+i]*vp[1+j];
                double duv_p_du = i*up[1+i-1]*vp[1+j];
                double duv_p_dv = j*up[1+i]*vp[1+j-1];
                double ab[2] = {AB[0 * 25 + i * 5 + j], AB[1 * 25 + i * 5 + j]}; // {a_ij, b_ij}

                f_uv[0] += ab[0] * uv_p; // SIP_fwd_u
                f_uv[1] += ab[1] * uv_p; // SIP_fwd_v

                J_ij[0] += ab[0] * duv_p_du; // dSIP_u/du
                J_ij[1] += ab[0] * duv_p_dv; // dSIP_u/dv
                J_ij[2] += ab[1] * duv_p_du; // dSIP_v/du
                J_ij[3] += ab[1] * duv_p_dv; // dSIP_v/dv
            }
        }

        // Residual: f(vb) = vb + SIP_fwd(vb) - out
        double res_uv[2] = {
            vb[0] + f_uv[0] - out_i[0], // ru
            vb[1] + f_uv[1] - out_i[1]  // rv
        };

        // Full Jacobian: J = I + dSIP/dvb
        J_ij[0] += 1.0;
        J_ij[3] += 1.0;

        // Invert 2×2 Jacobian analytically: inv(J) = adj(J) / det(J)
        double det = J_ij[0] * J_ij[3] - J_ij[1] * J_ij[2];
        double inv_det = 1.0 / det;
        // Update: vb <- vb - inv(J) @ residual
        vb[0] -= inv_det * (J_ij[3] * res_uv[0] - J_ij[1] * res_uv[1]);
        vb[1] -= inv_det * (J_ij[0] * res_uv[1] - J_ij[2] * res_uv[0]);
    }

    result[0] = vb[0];
    result[1] = vb[1];
}

struct __align__(8) TESS_Mapping_Struct {
    int    img_shape[2];       // 8 bytes
    double ref_px_coord[2];    // 16 bytes
    double cd[4];              // 32 bytes
    double cd_inv[4];          // 32 bytes
    double Rotation[9];        // 72 bytes
    double fwd_AB[50];         // 400 bytes
    double inv_AB[50];         // 400 bytes
}; // Total: 8 + 16 + 32 + 32 + 72 + 400 + 400 = 960 bytes, which is less than the 48KB limit for constant memory on most GPUs.

__constant__ TESS_Mapping_Struct dst_data; // Reference image. We set this once!
__constant__ TESS_Mapping_Struct src_data; // This is set for each image we upload. We could get away with just one struct and copying it each time, but this way we can have both the src and dst parameters available to the kernels at the same time without needing to copy between them.
// It may be worth it to do this in batches of 50 for a later step, but I suspect that's not necessary due to the hardware compute bottleneck.

__device__
void ICRS_from_TESS_cuda_f64(
    double* __restrict__ icrs_xyz, // double[3]
    double* __restrict__ src_pixel, // double[2]
    TESS_Mapping_Struct* src
){
    // Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    double uv[2];
    for(int i=0; i<2; i++)
        uv[i] = src_pixel[i] - (src->ref_px_coord[i] - 1);
    
    double distort[2];
    vec2_distort_sip_cuda_f64(distort, uv, src->fwd_AB);

    double xyz[3];
    for(int i=0; i<2; i++)
        xyz[i] = 0.0;

    // It turns out hcongrid doesn't do this. Its a bug in their code, not mine.
    // What the fuck?
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            xyz[i] += src->cd[i*2 + j] * (uv[j] + distort[j]);
    //for(int i=0; i<2; i++)
    //    xyz[i] = uv[i] + distort[i];

    const double PI = 3.1415926535897932384626433832795;
    xyz[2] = 180.0 / PI;

    double tmp[3];
    for(int i=0; i<3; i++)
        tmp[i] = 0.0;

    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            tmp[i] += src->Rotation[i*3 + j] * xyz[j];

    for(int i=0; i<3; i++)
        icrs_xyz[i] = tmp[i];
}

__device__
void TESS_from_ICRS_cuda_f64(
    double* __restrict__ dst_pixel, // double[2]
    double* __restrict__ icrs_xyz, // double[3]
    TESS_Mapping_Struct* dst
){
    const double PI = 3.1415926535897932384626433832795;

    double xyz[3];
    for(int i=0; i<3; i++)
        xyz[i] = 0.0;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            xyz[i] += dst->Rotation[j*3 + i] * icrs_xyz[j]; // Transposed!

    // Projection!
    for(int i=0; i<2; i++)
        xyz[i] *= (180.0 / PI) / xyz[2];
    
    // It turns out hcongrid doesn't do this. Its a bug in their code, not mine.
    // What the fuck?
    double uv[2];
    for(int i=0; i<2; i++)
        uv[i] = 0.0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            uv[i] += dst->cd_inv[i*2 + j] * xyz[j];
    //double uv[2] = {xyz[0], xyz[1]};

    double distort[2];
    vec2_distort_sip_inv_newton_cuda_f64(distort, uv, dst->fwd_AB, dst->inv_AB);
    // 
    for(int i=0; i<2; i++)
        dst_pixel[i] = distort[i] + (dst->ref_px_coord[i] - 1);

    // double distort[2];
    // vec2_distort_sip_cuda_f64(distort, uv, dst->inv_AB);
    // //
    // for(int i=0; i<2; i++)
    //     dst_pixel[i] = uv[i] + distort[i] + (dst->ref_px_coord[i] - 1);


}

__device__ __forceinline__
double lerp(double a, double b, double p) {
    return (1.0 - p)*a + p*b;
}

__device__ __forceinline__
double ilerp(double a, double b, double x) {
    return (x - a) / (b - a);
}

__device__ __forceinline__ // Query the image at the specified xy coordinate.
double textel(
    const double* __restrict__ img,
    int ix, int iy,
    int w, int h
) {
    // Branch-free: clamp to valid range, then zero out if OOB
    int valid = (0 <= ix) & (ix < w) & (0 <= iy) & (iy < h);  // 1 or 0
    ix = max(0, min(ix, w - 1));  // clamp so the load is always safe
    iy = max(0, min(iy, h - 1));

    return valid ? img[iy * w + ix] : 0.0;  // SEL instruction, no branch
}

__device__ __forceinline__ // Query the image at the specified xy coordinate.
double texture(
    const double* __restrict__ img,
    double x, double y,
    int w, int h
) {
    int ix = (int)(x + 0.5);  // round to nearest
    int iy = (int)(y + 0.5);

    return textel(img, ix, iy, w, h);
}

// Generated by Claude Opus 4.6 on 2026 March 17.
// I was hoping to construct it from the chain above :(
__device__
double texture_catmull_rom(
    const double* __restrict__ img,
    double x, double y,
    int w, int h)
{
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    double tx = x - (double)ix;
    double ty = y - (double)iy;

    double wx[4], wy[4];
    wx[0] = tx * (-0.5 + tx * ( 1.0 - 0.5 * tx));
    wx[1] = 1.0 + tx * tx * (-2.5 + 1.5 * tx);
    wx[2] = tx * ( 0.5 + tx * ( 2.0 - 1.5 * tx));
    wx[3] = tx * tx * (-0.5 + 0.5 * tx);
    
    wy[0] = ty * (-0.5 + ty * ( 1.0 - 0.5 * ty));
    wy[1] = 1.0 + ty * ty * (-2.5 + 1.5 * ty);
    wy[2] = ty * ( 0.5 + ty * ( 2.0 - 1.5 * ty));
    wy[3] = ty * ty * (-0.5 + 0.5 * ty);

    double val = 0.0;
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            val += wx[n] * wy[m] * textel(img, ix+n-1, iy+m-1, w, h);

    return val;
}

// ═══════════════════════════════════════════════════════════════════
//  IIR B-spline prefilter (cubic, order=3)
//  Converts raw samples → B-spline coefficients IN-PLACE.
//  
//  Run twice: once for rows (stride=1), once for columns (stride=w).
//  Each thread handles one independent row or column.
//  
//  Matches scipy.ndimage._interpolation.spline_filter with
//  mode='constant' (implicit zeros outside boundary → mirror BC
//  for the IIR is what scipy actually does internally).
// ═══════════════════════════════════════════════════════════════════

__global__
void bspline3_prefilter_rows_f64(
    double* __restrict__ c,   // [h × w] image, modified in-place
    int w, int h)
{
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row >= h) return;

    const double z1 = -0.26794919243112270647;  // -2 + sqrt(3)
    const double lambda = 6.0;                   // normalization
    const int K = 24;  // mirror horizon: |z1|^24 ≈ 2.6e-14, below f64 eps

    int base = row * w;

    // ── Causal pass: mirror boundary ──
    // c⁺[0] = sum_{k=0..K} z1^k * c[mirror(k)]
    double c_plus = c[base + 0];
    double z1k = z1;
    for (int k = 1; k <= K && k < 2*w-2; k++) {
        int mk = (k < w) ? k : (2*w - 2 - k);  // mirror index
        c_plus += z1k * c[base + mk];
        z1k *= z1;
    }
    c[base + 0] = c_plus;

    for (int i = 1; i < w; i++)
        c[base + i] = c[base + i] + z1 * c[base + i - 1];

    // ── Anticausal pass ──
    // Initial condition: c⁻[N-1] = z1/(z1²-1) * (c⁺[N-1] + z1*c⁺[N-2])
    c[base + w-1] = z1 / (z1*z1 - 1.0) *
                    (c[base + w-1] + z1 * c[base + w-2]);

    for (int i = w - 2; i >= 0; i--)
        c[base + i] = z1 * (c[base + i + 1] - c[base + i]);

    // ── Scale ──
    for (int i = 0; i < w; i++)
        c[base + i] *= lambda;
}

__global__
void bspline3_prefilter_cols_f64(
    double* __restrict__ c,   // [h × w] image, modified in-place
    int w, int h)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    if (col >= w) return;

    const double z1 = -0.26794919243112270647;
    const double lambda = 6.0;
    const int K = 24;

    // stride = w (column-major access within row-major layout)

    // ── Causal pass: mirror boundary ──
    double c_plus = c[0 * w + col];
    double z1k = z1;
    for (int k = 1; k <= K && k < 2*h-2; k++) {
        int mk = (k < h) ? k : (2*h - 2 - k);
        c_plus += z1k * c[mk * w + col];
        z1k *= z1;
    }
    c[0 * w + col] = c_plus;

    for (int i = 1; i < h; i++)
        c[i * w + col] = c[i * w + col] + z1 * c[(i-1) * w + col];

    // ── Anticausal pass ──
    c[(h-1) * w + col] = z1 / (z1*z1 - 1.0) *
                         (c[(h-1) * w + col] + z1 * c[(h-2) * w + col]);

    for (int i = h - 2; i >= 0; i--)
        c[i * w + col] = z1 * (c[(i+1) * w + col] - c[i * w + col]);

    // ── Scale ──
    for (int i = 0; i < h; i++)
        c[i * w + col] *= lambda;
}

// Generated by Claude Opus 4.6 on 2026 March 17.
// I was hoping to construct it from the chain above :(
__device__
double texture_cubic_b(
    const double* __restrict__ img,
    double x, double y,
    int w, int h)
{
    int ix = (int)floor(x);
    int iy = (int)floor(y);
    double tx = x - floor(x);//(double)ix;
    double ty = y - floor(y);//(double)iy;

    double wx[4], wy[4];
    double omt = 1.0 - tx;
    wx[0] = omt * omt * omt * (1.0/6.0);
    wx[1] = (3.0*tx*tx*tx - 6.0*tx*tx + 4.0) * (1.0/6.0);
    wx[2] = (-3.0*tx*tx*tx + 3.0*tx*tx + 3.0*tx + 1.0) * (1.0/6.0);
    wx[3] = tx * tx * tx * (1.0/6.0);
    
    double omt_y = 1.0 - ty;
    wy[0] = omt_y * omt_y * omt_y * (1.0/6.0);
    wy[1] = (3.0*ty*ty*ty - 6.0*ty*ty + 4.0) * (1.0/6.0);
    wy[2] = (-3.0*ty*ty*ty + 3.0*ty*ty + 3.0*ty + 1.0) * (1.0/6.0);
    wy[3] = ty * ty * ty * (1.0/6.0);

    double val = 0.0;
    for (int m = 0; m < 4; m++)
        for (int n = 0; n < 4; n++)
            val += wx[n] * wy[m] * textel(img, ix+n-1, iy+m-1, w, h);

    return val;
}

// The big one >:)
__global__ // Remaps pixels from src onto dst. do NOT have dst's pixels loaded into memory, that's where this writes.
void cuda_hcongrid_f64(
    double* __restrict__ dst_img,
    double* __restrict__ src_img
    // TESS_Mapping_Struct* dst,
    // TESS_Mapping_Struct* src
){
    TESS_Mapping_Struct* dst = &dst_data;
    TESS_Mapping_Struct* src = &src_data;

    //int idx;
    //if(mapidx(&idx, dst->img_shape[1], dst->img_shape[0])) return; // Map 2D thread/block indices to a 1D index, return if out of bounds
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= dst->img_shape[0] * dst->img_shape[1]) return;
    // idx is now bounded and has the right index!

    // It looks like I dont need to allocate anything more than what's here. Yippee!

    // Let's sketch the entire shader.
    // 1. Generate a pixel coordinate for this thread. (dst's coordinates)
    // 2. Map dst -> ICRS
    // 3. Map ICRS -> src
    // 4. Sample src at the mapped coordinate.
    // 5. Write to dst.'
    // 6. Yippee!

    // 1. Generate pixel coordinate [0, N)
    int x = idx % dst->img_shape[1]; // These pictures are always 2048x2048. I wonder if we should hardcode this. eh
    int y = idx / dst->img_shape[1];

    // 2. Map dst -> ICRS
    double xy[2] = {(double)x, (double)y};
    double icrs[3];
    ICRS_from_TESS_cuda_f64(icrs, xy, dst);
    double mag = sqrt(icrs[0]*icrs[0] + icrs[1]*icrs[1] + icrs[2]*icrs[2]);
    for(int i=0; i<3; i++)
        icrs[i] /= mag; // Normalize to unit vector. This is important for numerical stability in the next step, which involves a matrix multiplication with the R_ij matrix that can have large values.
    
    // 3. Map ICRS -> src
    TESS_from_ICRS_cuda_f64(xy, icrs, src);

    // 4. Sample src at the mapped coordinate.
    //double lum = texture_catmull_rom(src_img, xy[0], xy[1], src->img_shape[1], src->img_shape[0]);
    double lum = texture_cubic_b(src_img, xy[0], xy[1], src->img_shape[1], src->img_shape[0]);

    // 5. Write to dst.
    dst_img[idx] = lum;
    //dst_img[idx] = xy[0]; //  Validate that data flows through properly

    //double dx = xy[0] - (double)x;
    //double dy = xy[1] - (double)y;
    //dst_img[idx] = sqrt(dx*dx + dy*dy);

    // 6. Yippee!
}

__global__
void cuda_resample_undistort_f64(
    double* __restrict__ dst_img,
    double* __restrict__ src_img,
    int chunk_w, int chunk_h,
    int chunk_x0, int chunk_y0
){
    // Removes SIP distortion only — no frame-to-frame alignment.
    // Uses src_data constant memory for WCS (same image's own WCS).
    // Output pixels are in ideal gnomonic coordinates (linear w.r.t. sky).
    TESS_Mapping_Struct* wcs = &src_data;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= chunk_w * chunk_h) return;

    int x = idx % chunk_w;
    int y = idx / chunk_w;

    // Output pixel position in full-frame CCD coordinates
    double full_x = (double)(x + chunk_x0);
    double full_y = (double)(y + chunk_y0);

    // Offset from CRPIX — these are "ideal" undistorted pixel offsets
    double uv_ideal[2] = {
        full_x - (wcs->ref_px_coord[0] - 1.0),
        full_y - (wcs->ref_px_coord[1] - 1.0)
    };

    // SIP inverse Newton: find distorted uv such that uv_dist + SIP_fwd(uv_dist) = uv_ideal
    double uv_dist[2];
    vec2_distort_sip_inv_newton_cuda_f64(uv_dist, uv_ideal, wcs->fwd_AB, wcs->inv_AB);

    // Convert back to pixel coords, relative to the chunk origin
    double src_x = uv_dist[0] + (wcs->ref_px_coord[0] - 1.0) - (double)chunk_x0;
    double src_y = uv_dist[1] + (wcs->ref_px_coord[1] - 1.0) - (double)chunk_y0;

    // B-spline sample the prefiltered source image
    double lum = texture_cubic_b(src_img, src_x, src_y, chunk_w, chunk_h);

    dst_img[idx] = lum;
}


"""
def cuda_compile(cu_code):
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
    return krnl

#krnl = nvcc.SourceModule(cu_code)
krnl = cuda_compile(cu_code)

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

    def __call__(self, all_data, clip_low=2.5, clip_high=2.5, init_mask=None):
        """
        all_data: flat f64 array of shape (n_chunks * chunk_size,)
                  with chunks packed contiguously.
        init_mask: optional int32 array same shape as all_data. If provided,
                   pixels with mask=0 are excluded from the start (e.g. padding).
        Returns: (mask_bool, counts, lows, highs, means, stds).
        """
        drv.memcpy_htod(self.g_data, all_data)
        if init_mask is not None:
            drv.memcpy_htod(self.g_mask, init_mask)
        else:
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
        return mask_bool, self.h_ccnt.copy(), los, his, means, stds


@numba.njit(parallel=True, nogil=True)
def extract_medians_from_masks(data_flat, mask_flat, n_windows, window_size):
    """
    Compute per-window median and std from flat data/mask arrays.
    data_flat and mask_flat are both 1D, length n_windows * window_size.
    mask_flat: 1 = valid/surviving, 0 = padding or clipped.
    Returns (medians, stds) each of length n_windows.
    """
    medians = np.empty(n_windows, dtype=np.float64)
    stds = np.empty(n_windows, dtype=np.float64)
    for w in numba.prange(n_windows):
        base = w * window_size
        # Count survivors
        cnt = 0
        for i in range(window_size):
            if mask_flat[base + i]:
                cnt += 1
        if cnt == 0:
            medians[w] = 0.0
            stds[w] = 0.0
            continue
        # Gather survivors + running sums for std
        survivors = np.empty(cnt, dtype=np.float64)
        j = 0
        s = 0.0
        s2 = 0.0
        for i in range(window_size):
            if mask_flat[base + i]:
                val = data_flat[base + i]
                survivors[j] = val
                s += val
                s2 += val * val
                j += 1
        # Median via sort
        survivors.sort()
        if cnt % 2 == 1:
            medians[w] = survivors[cnt // 2]
        else:
            medians[w] = (survivors[cnt // 2 - 1] + survivors[cnt // 2]) / 2.0
        # Std (population, ddof=0)
        mean = s / cnt
        stds[w] = ((s2 / cnt) - mean * mean) ** 0.5
    return medians, stds


def repair_bad_sky_values(x, y, v, s, sig):
    """
    Non-destructively repair bad sky sample values.
    Returns v_clean with bad values replaced by neighbor interpolation.
    Bad values: v <= 0, or s >= 2*sig.
    """
    v_clean = v.copy()
    # Pass 1: fix where v <= 0
    bad = np.where(v_clean <= 0)[0]
    good = np.where(v_clean > 0)[0]
    if len(bad) > 0 and len(good) > 0:
        xgood, ygood, vgood = x[good], y[good], v_clean[good]
        for idx in bad:
            rd = np.sqrt((xgood - x[idx])**2 + (ygood - y[idx])**2)
            nearest = np.argsort(rd)[:9]
            v_clean[idx] = np.median(vgood[nearest])
    # Pass 2: fix where s >= 2*sig
    bad = np.where(s >= 2 * sig)[0]
    good = np.where(s < 2 * sig)[0]
    if len(bad) > 0 and len(good) > 0:
        xgood, ygood, vgood = x[good], y[good], v_clean[good]
        for idx in bad:
            rd = np.sqrt((xgood - x[idx])**2 + (ygood - y[idx])**2)
            nearest = np.argsort(rd)[:9]
            v_clean[idx] = np.median(vgood[nearest])
    return v_clean











def wcs_variables_validate(header):
    """Extract all relevant WCS variables from the FITS header.
    This is a separate step from build_wcs_transform() because some of
    these variables are needed for both the per-pixel transform and
    the precomputation step."""

    keys = [
        # Image details
        "CTYPE1", # = 'RA---TAN-SIP'       / Gnomonic projection + SIP distortions
        "CTYPE2", # = 'DEC--TAN-SIP'       / Gnomonic projection + SIP distortions
        "NAXIS1", # =                 2136 / length of first array dimension      
        "NAXIS2", # =                 2078 / length of second array dimension     

        # UV Coordinates: ij = uv * (naxis - 1) - (crpix - 1)  # Step 1: [0,1] → pixel offset from CRPIX
        "CRPIX1", # =               1045.0 / X reference pixel  # Coordinate Reference Pixel. We use this to offset!
        "CRPIX2", # =               1001.0 / Y reference pixel  # 

        # Circular harmonics distortion polynomial:
        # Let [u,v] = r[cos(φ), sin(φ)]. Then, 
        # r²(A₂₀c²s⁰ + A₁₁c¹s¹ + A₀₂c⁰s²) + r³(A₃₀c³s⁰ + A₂₁c²s¹ + A₁₂c¹s² + A₀₃c⁰s³) + r⁴(A₄₀c⁴s⁰ + A₃₁c³s¹ + A₂₂c²s² + A₁₃c¹s³ + A₀₄c⁰s⁴)
        # This is a common graphics pitfall, but shows that this is a 2nd order and above correction *only.*
        "A_ORDER", # =                    4 / Polynomial order, axis 1  
        "B_ORDER", # =                    4 / Polynomial order, axis 2  
        "AP_ORDER", # =                   4 / Inv polynomial order, axis 1
        "BP_ORDER", # =                   4 / Inv polynomial order, axis 2
        # r² terms
        "A_2_0", "A_1_1", "A_0_2",                    "AP_2_0", "AP_1_1", "AP_0_2",
        "B_2_0", "B_1_1", "B_0_2",                    "BP_2_0", "BP_1_1", "BP_0_2",
        # r³ terms
        "A_3_0", "A_2_1", "A_1_2", "A_0_3",           "AP_3_0", "AP_2_1", "AP_1_2", "AP_0_3",
        "B_3_0", "B_2_1", "B_1_2", "B_0_3",           "BP_3_0", "BP_2_1", "BP_1_2", "BP_0_3",
        # r⁴ terms
        "A_4_0", "A_3_1", "A_2_2", "A_1_3", "A_0_4",  "AP_4_0", "AP_3_1", "AP_2_2", "AP_1_3", "AP_0_4",
        "B_4_0", "B_3_1", "B_2_2", "B_1_3", "B_0_4",  "BP_4_0", "BP_3_1", "BP_2_2", "BP_1_3", "BP_0_4",
        # NOTE: This HARDCODES the orders to 4 on both. We'll need to validate that all of TESS obeys this constraint.
        # 
        # Also, this polynomial applies to the *un-normalized* pixel coordinates, so they'll be quite small.
        # I'm surprised they aren't applied to normalized radii scaled by 1/Bin(N,k) or in general 1/a!b!c!... or more generally
        #     following a Cauchy-Hadamard 1/limsup criterion to avoid vanishingly tiny coefficients.
        # Gemini: "Without this normalization, the coefficients for higher-order terms (r^4) must counteract
        #     the O(N^k) growth of the pixel offsets, leading to precision loss and preventing the series 
        #     from naturally behaving like a convergent power series within the image's "radius of convergence."
        # sure. A pade approximant would work great here too and would avoid the next strange variables. Perhaps we should exclude them?
        #"A_DMAX", # =   46.929824173871566 / maximum distortion, axis 1  
        #"B_DMAX", # =    47.05180925988134 / maximum distortion, axis 2  

        # "Coordinate Description" CD matrix: Step 3: pixel → intermediate world coordinates (degrees)
        # its a view transform with no translation. Weird that its in polar, shouldn't this be nonlinear?
        "CD1_1", "CD1_2", # [u]   [CD1_1, CD1_2] [x]
        "CD2_1", "CD2_2", # [v] = [CD2_1, CD2_2].[y]

        # Gnomonic projection reference point: Step 4: tangent plane → native Cartesian
        # This assigns the z coordinate of the CD matrix, tangent to the celestial sphere scaled by some reference point.
        "CRVAL1", # = 110.8016271758663000 / RA at CRPIX1, CRPIX2  
        "CRVAL2", # = -64.2806653158681200 / DEC at CRPIX1, CRPIX2 
        # NOTE: The pixel labelled CRPix commutes with *all previous corrections* completely! r=0 here, so all distortions vanish, 
        #     and all matrix operations act on [0,0] to produce [0,0]. This is a critical insight that allows us to precompute the rotation matrix R from CRVAL alone, without worrying about SIP or CD.
        # Or... We can compute the axis-angle rotor from these values instead! 
    ]

    for key in keys:
        if key not in header:
            raise ValueError(f"Missing required WCS header keyword: {key}")
    
    return keys

def wcs_variables_load_nonan(header, keys):
    """
    Validate that all required WCS header keywords are present and not NaN.
    Returns a dict of all the relevant WCS variables extracted from the header.
    """
    STRING_KEYS = {"CTYPE1", "CTYPE2"}
    TESS_Mapping_Variables = dict()
    for key in keys:
        value = header[key]
        if key not in STRING_KEYS and np.isnan(value):
            raise ValueError(f"WCS header keyword {key} is NaN")
        TESS_Mapping_Variables[key] = value
    return TESS_Mapping_Variables

from collections import namedtuple
WCSMappingStruct = namedtuple('WCSMappingStruct', [
    'ctype_i',
    'naxis_int_i',
    'naxis_i',
    'ref_px_coord_i',
    'fwd_distortion_order_i',
    'inv_distortion_order_i',
    'fwd_distortion_coeffs_kij',
    'inv_distortion_coeffs_kij',
    'fwd_distortion_coeffs_r2_ki',
    'inv_distortion_coeffs_r2_ki',
    'fwd_distortion_coeffs_r3_ki',
    'inv_distortion_coeffs_r3_ki',
    'fwd_distortion_coeffs_r4_ki',
    'inv_distortion_coeffs_r4_ki',
    'cd_matrix_original_ij',
    'cd_matrix_ij',
    'cd_matrix_inv_ij',
    'R_ij',
    'Normal_i',
])

def wcs_assemble_struct(TESS_Mapping_Variables):
    """
    Precompute everything needed for the per-pixel transform.

    Returns a WCSMappingStruct instance — the "uniform buffer" that gets uploaded once
    to the GPU and read by every thread.
    """
    # Context for copilot reading this: I'm going to cull comments as we go to keep things organized. This paste below is not final!
    # The _ijk suffixes refer to the indices they "ought" to refer to. The number of them is the same as the number of array dimensions.
    Vars = TESS_Mapping_Variables
    INT_T = np.int32
    FLOAT_T = np.float64

    # Image details
    ctype_i = np.array([Vars["CTYPE1"], Vars["CTYPE2"]]) # (RA, Dec) axes
    naxis_int_i = np.array([Vars["NAXIS1"], Vars["NAXIS2"]], dtype=INT_T) # (nx, ny) indexing
    naxis_i = naxis_int_i.astype(FLOAT_T) # (nx, ny) indexing

    # UV Coordinates: ij = uv * (naxis - 1) - (crpix - 1)  # Step 1: [0,1] → pixel offset from CRPIX
    ref_px_coord_i = np.array([Vars["CRPIX1"], Vars["CRPIX2"]], dtype=FLOAT_T) # (x, y) indexing

    # Circular harmonics distortion polynomial:
    fwd_distortion_order_i = np.array([Vars["A_ORDER"], Vars["B_ORDER"]], dtype=INT_T) # [A/B] polynomial order
    inv_distortion_order_i = np.array([Vars["AP_ORDER"], Vars["BP_ORDER"]], dtype=INT_T) # [AP/BP] polynomial order

    # This variant requires another comment.
    # I'm going to seperate the coefficients to be loaded in a "lean" way as described below,
    #     and then pack them to the "fast" way. The indicies in this case refer to the kth dimension, and the ith *from x=order*.
    # This ordering may be clearer when packed into the matrices.
    # r² terms
    fwd_distortion_coeffs_r2_ki = np.array([
        [Vars["A_2_0"], Vars["A_1_1"], Vars["A_0_2"]],
        [Vars["B_2_0"], Vars["B_1_1"], Vars["B_0_2"]]
    ])
    inv_distortion_coeffs_r2_ki = np.array([
        [Vars["AP_2_0"], Vars["AP_1_1"], Vars["AP_0_2"]],
        [Vars["BP_2_0"], Vars["BP_1_1"], Vars["BP_0_2"]]
    ])

    # r³ terms
    fwd_distortion_coeffs_r3_ki = np.array([
        [Vars["A_3_0"], Vars["A_2_1"], Vars["A_1_2"], Vars["A_0_3"]],
        [Vars["B_3_0"], Vars["B_2_1"], Vars["B_1_2"], Vars["B_0_3"]]
    ])
    inv_distortion_coeffs_r3_ki = np.array([
        [Vars["AP_3_0"], Vars["AP_2_1"], Vars["AP_1_2"], Vars["AP_0_3"]],
        [Vars["BP_3_0"], Vars["BP_2_1"], Vars["BP_1_2"], Vars["BP_0_3"]]
    ])

    # r⁴ terms
    fwd_distortion_coeffs_r4_ki = np.array([
        [Vars["A_4_0"], Vars["A_3_1"], Vars["A_2_2"], Vars["A_1_3"], Vars["A_0_4"]],
        [Vars["B_4_0"], Vars["B_3_1"], Vars["B_2_2"], Vars["B_1_3"], Vars["B_0_4"]]
    ])
    inv_distortion_coeffs_r4_ki = np.array([
        [Vars["AP_4_0"], Vars["AP_3_1"], Vars["AP_2_2"], Vars["AP_1_3"], Vars["AP_0_4"]],
        [Vars["BP_4_0"], Vars["BP_3_1"], Vars["BP_2_2"], Vars["BP_1_3"], Vars["BP_0_4"]]
    ])

    # Preparing fast matmuls. Perhaps the coefficients directly are superior to this? I'd like to keep the option so users can choose which to optimize.
    fwd_distortion_coeffs_kij = np.zeros((2, 5, 5), dtype=FLOAT_T) # [A/B, i, j]
    inv_distortion_coeffs_kij = np.zeros((2, 5, 5), dtype=FLOAT_T) # [AP/BP, i, j]
    
    r2 = np.arange(3) # r² terms -> 0, 1, 2
    fwd_distortion_coeffs_kij[:, 2-r2, r2] = fwd_distortion_coeffs_r2_ki
    inv_distortion_coeffs_kij[:, 2-r2, r2] = inv_distortion_coeffs_r2_ki

    r3 = np.arange(4) # r³ terms -> 0, 1, 2, 3
    fwd_distortion_coeffs_kij[:, 3-r3, r3] = fwd_distortion_coeffs_r3_ki
    inv_distortion_coeffs_kij[:, 3-r3, r3] = inv_distortion_coeffs_r3_ki

    r4 = np.arange(5) # r⁴ terms -> 0, 1, 2, 3, 4
    fwd_distortion_coeffs_kij[:, 4-r4, r4] = fwd_distortion_coeffs_r4_ki
    inv_distortion_coeffs_kij[:, 4-r4, r4] = inv_distortion_coeffs_r4_ki

    # whatever
    #"A_DMAX", # =   46.929824173871566 / maximum distortion, axis 1  
    #"B_DMAX", # =    47.05180925988134 / maximum distortion, axis 2  

    # CD matrix transforms pixels → "degrees", NOT radians. 
    # The domain of this function is the tangent plane of the sphere, NOT RA/Dec coordinates. To first approximation they can be.
    cd_matrix_original_ij = np.array([
        [Vars["CD1_1"], Vars["CD1_2"]],
        [Vars["CD2_1"], Vars["CD2_2"]]
    ], dtype=FLOAT_T)

    # Corrected to handle orientation at the celestial north pole via a 90 degree rotation.
    # Wait. In testing, this is visually ***wrong.*** What? 
    cd_matrix_ij = np.array([
        [-Vars["CD2_1"], -Vars["CD2_2"]],
        [ Vars["CD1_1"],  Vars["CD1_2"]]
    ], dtype=FLOAT_T)
    cd_matrix_ij = cd_matrix_original_ij
    #cd_matrix_ij = np.array([[0.0, -1.0], [1.0, 0.0]]) @ cd_matrix_original_ij # Rotate by 90 degrees to handle north pole orientation. This is a hack but it works and is more efficient than computing the rotation from CRVAL1/2.
    cd_matrix_inv_ij = np.linalg.inv(cd_matrix_ij)

    # Gnomonic projection reference point: Step 4: tangent plane → native Cartesian. In degrees.
    ref_radec_coord_i = np.array([Vars["CRVAL1"], Vars["CRVAL2"]], dtype=FLOAT_T)

    # TAN projection constant: the radius of curvature of the unit sphere
    # in degrees.  This is the "focal length" of the gnomonic projection.
    ref_R0 = 180.0 / np.pi   # 57.29577951...°
    # Also represents the z coordinate of our vector.

    c_ra, s_ra = np.cos(ref_radec_coord_i[0] / ref_R0), np.sin(ref_radec_coord_i[0] / ref_R0)
    c_de, s_de = np.cos(ref_radec_coord_i[1] / ref_R0), np.sin(ref_radec_coord_i[1] / ref_R0)
    ref_normal_i = np.array([
        c_de * c_ra, # x
        c_de * s_ra, # y
        s_de         # z
    ], dtype=FLOAT_T) 

    # dn/dRA (normalized) = unit east
    ref_tan_i = np.array([-s_ra, c_ra, 0], dtype=FLOAT_T)

    # dn/dDec = unit north
    #ref_bin_i = np.cross(ref_normal_i, ref_tan_i)
    ref_bin_i = np.array([
        -s_de * c_ra, # x
        -s_de * s_ra, # y
         c_de         # z
    ], dtype=FLOAT_T)
    ref_R_ij = np.ascontiguousarray(np.array([ref_tan_i, ref_bin_i, ref_normal_i]).T) # Ensure contiguous memory
    
    # # FITS native frame: columns are [south, east, boresight]
    # # south = -north, per Calabretta & Greisen (2002) §2.4 (φ_p = 180°)
    # ref_south_i = -ref_north_i
    # R = np.ascontiguousarray(
    #     np.array([ref_south_i, ref_east_i, ref_normal_i]).T
    # )
    # ref_mat_ij = np.ascontiguousarray(np.array([-ref_bin_i, ref_tan_i, ref_normal_i]).T) # 
    # 
    # # 3×3 P matrix: folds sign-swap + CD + R0 for homography composition
    # #   P @ [u', v', 1] = native direction vector (unnormalized)
    # P = np.array([
    #     [-cd_matrix_original_ij[1,0], -cd_matrix_original_ij[1,1], 0.0],
    #     [ cd_matrix_original_ij[0,0],  cd_matrix_original_ij[0,1], 0.0],
    #     [                        0.0,                         0.0, ref_R0],
    # ], dtype=FLOAT_T)
    # P_inv = np.linalg.inv(P)

    return WCSMappingStruct(**{
        # Image metadata
        'ctype_i':  ctype_i,
        'naxis_int_i':  naxis_int_i,
        'naxis_i':  naxis_i,
        'ref_px_coord_i':  ref_px_coord_i,
        # SIP distortion
        'fwd_distortion_order_i': fwd_distortion_order_i,
        'inv_distortion_order_i': inv_distortion_order_i,
        'fwd_distortion_coeffs_kij': fwd_distortion_coeffs_kij,
        'inv_distortion_coeffs_kij': inv_distortion_coeffs_kij,
        'fwd_distortion_coeffs_r2_ki': fwd_distortion_coeffs_r2_ki,
        'inv_distortion_coeffs_r2_ki': inv_distortion_coeffs_r2_ki,
        'fwd_distortion_coeffs_r3_ki': fwd_distortion_coeffs_r3_ki,
        'inv_distortion_coeffs_r3_ki': inv_distortion_coeffs_r3_ki,
        'fwd_distortion_coeffs_r4_ki': fwd_distortion_coeffs_r4_ki,
        'inv_distortion_coeffs_r4_ki': inv_distortion_coeffs_r4_ki,
        # CD matrices
        'cd_matrix_original_ij': cd_matrix_original_ij,
        'cd_matrix_ij': cd_matrix_ij,
        'cd_matrix_inv_ij': cd_matrix_inv_ij,
        # Rotation matrix: native tangent-plane → ICRS Cartesian
        'R_ij':      ref_R_ij,
        'Normal_i': ref_normal_i, # Boresight
    })



TESS_Mapping_Struct_dtype = np.dtype([
    ('img_shape',      np.int32, (2,)),  #  8 bytes
    ('ref_px_coord', np.float64, (2,)),  #  16 bytes
    ('cd',           np.float64, (4,)),  #  32 bytes
    ('cd_inv',       np.float64, (4,)),  #  32 bytes
    ('Rotation',     np.float64, (9,)),  #  72 bytes
    ('fwd_AB',       np.float64, (50,)), # 400 bytes
    ('inv_AB',       np.float64, (50,)), # 400 bytes
], align=True)

def pack_TMS(TESS_Mapping_Struct):
    """Pack a WCSMappingStruct into a single numpy buffer matching the CUDA struct."""
    TMS = np.zeros(1, dtype=TESS_Mapping_Struct_dtype)
    #globals().update(locals())
    TMS['img_shape']    = np.array(TESS_Mapping_Struct.naxis_int_i, dtype=np.int32)
    TMS['ref_px_coord'] = TESS_Mapping_Struct.ref_px_coord_i.ravel().astype(np.float64)
    TMS['cd']           = TESS_Mapping_Struct.cd_matrix_ij.ravel().astype(np.float64)
    TMS['cd_inv']       = TESS_Mapping_Struct.cd_matrix_inv_ij.ravel().astype(np.float64)
    TMS['Rotation']     = TESS_Mapping_Struct.R_ij.ravel().astype(np.float64)
    TMS['fwd_AB']       = TESS_Mapping_Struct.fwd_distortion_coeffs_kij.ravel().astype(np.float64)
    TMS['inv_AB']       = TESS_Mapping_Struct.inv_distortion_coeffs_kij.ravel().astype(np.float64)
    return TMS

class CUDAHcongridContainer:
    def __init__(self):
        self.initialized = False
        self.ftype = np.float64
        #self.ftype = np.float32

    # Runs on the first call to initialize kernels as a "JIT" approach.
    def initialize(self, data, header): # loaded from data, header <<= fits.open("file.fits", header=True)
        if self.initialized:
            return
        
        self.cu_prefilter_rows = krnl.get_function("bspline3_prefilter_rows_f64")
        self.cu_prefilter_cols = krnl.get_function("bspline3_prefilter_cols_f64")
        self.cu_hcongrid_f64 = krnl.get_function("cuda_hcongrid_f64")

        dst_Mapping_Variables = wcs_variables_load_nonan(header, wcs_variables_validate(header))
        dst_Mapping_Struct = wcs_assemble_struct(dst_Mapping_Variables)
        self.dst_Mapping_Variables = dst_Mapping_Variables # Just in case we need it!
        self.dst_Mapping_Struct = dst_Mapping_Struct # Just in case we need it!

        # Initialize constant memory parameters
        g_dst_data, g_dst_data_size = krnl.get_global("dst_data")
        g_src_data, g_src_data_size = krnl.get_global("src_data")
        drv.memcpy_htod(g_dst_data, pack_TMS(dst_Mapping_Struct))
        drv.memcpy_htod(g_src_data, pack_TMS(dst_Mapping_Struct)) # Preventing garbage
        # incredible
        #drv.memcpy_htod(g_dst_data, 
        #     pack_TMS(wcs_assemble_struct(wcs_variables_load_nonan(header, wcs_variables_validate(header)))))

        # Initialize variable memory
        c_dst = np.empty_like(data.ravel(), dtype=self.ftype)
        g_dst = drv.mem_alloc(c_dst.nbytes)
        g_src = drv.mem_alloc(c_dst.nbytes) # src and dst are the same size, so we can reuse the buffer size.
        # We will copy the actual data in each call, but we need to allocate the GPU memory here.

        #c_dst[:] = data.ravel().astype(self.ftype) # Fill with trash to test
        #drv.memcpy_htod(g_dst, c_dst) # this will be immediately deleted

        self.g_dst_data, self.g_dst_data_size = g_dst_data, g_dst_data_size
        self.g_src_data, self.g_src_data_size = g_src_data, g_src_data_size

        self.g_dst, self.c_dst = g_dst, c_dst
        self.g_src, self.c_src = g_src, c_dst.copy()

        self.initialized = True
    
    def hcongrid(self, src_data, src_header):

        if src_data.ravel().shape != self.c_dst.shape:
            raise ValueError(f"Source data shape {src_data.shape} does not match destination shape {self.c_dst.shape}.")

        drv.memcpy_htod(self.g_src, src_data.ravel().astype(self.ftype))

        guh = wcs_assemble_struct(wcs_variables_load_nonan(src_header, wcs_variables_validate(src_header))) # This is the one that actually matters. We need to update the WCS parameters for the source image!
        #print(guh)
        drv.memcpy_htod(self.g_src_data, pack_TMS(guh))

        h, w = self.dst_Mapping_Struct.naxis_int_i
        h, w = int(h), int(w)

        # Prefilter the source image in-place on GPU
        self.cu_prefilter_rows(
            self.g_src, np.int32(w), np.int32(h),
            block=(256,1,1), grid=((h + 255) // 256, 1))
        self.cu_prefilter_cols(
            self.g_src, np.int32(w), np.int32(h),
            block=(256,1,1), grid=((w + 255) // 256, 1))

        self.cu_hcongrid_f64(
            self.g_dst,
            self.g_src,
            #block=(256,1,1), grid=((self.c_dst.size + 255) // 256, 1)
            #block=(16,16,1), grid=((w + 15) // 16, (h + 15) // 16)
            block=(256,1,1), grid=((w*h + 255) // 256, 1)
        )

        #print("Yippee!")
        #globals().update(locals()) # For interactive inspection

        drv.memcpy_dtoh(self.c_dst, self.g_dst)
        return self.c_dst.reshape(src_data.shape).copy()

    # ── SIP-only undistortion (no frame alignment) ────────────────

    def initialize_undistort(self, header, chunk_shape, chunk_offset=(0, 0)):
        """Set up GPU buffers for chunk-sized undistortion.
        
        chunk_shape:  (height, width) of the chunk, e.g. (512, 512) or (2048, 2048)
        chunk_offset: (y0, x0) pixel offset of chunk origin in full CCD frame
        """
        self.cu_undistort = krnl.get_function("cuda_resample_undistort_f64")

        self.chunk_h, self.chunk_w = int(chunk_shape[0]), int(chunk_shape[1])
        self.chunk_y0, self.chunk_x0 = int(chunk_offset[0]), int(chunk_offset[1])
        n_pixels = self.chunk_h * self.chunk_w

        # Upload this CCD's WCS to src_data constant memory
        wcs_vars = wcs_variables_load_nonan(header, wcs_variables_validate(header))
        wcs_struct = wcs_assemble_struct(wcs_vars)
        self.undistort_wcs_struct = wcs_struct
        g_src_data, _ = krnl.get_global("src_data")
        drv.memcpy_htod(g_src_data, pack_TMS(wcs_struct))

        # Allocate chunk-sized GPU buffers
        buf = np.empty(n_pixels, dtype=np.float64)
        self.g_ud_src = drv.mem_alloc(buf.nbytes)
        self.g_ud_dst = drv.mem_alloc(buf.nbytes)
        self.c_ud_dst = buf.copy()

        self.undistort_initialized = True

    def undistort(self, src_data):
        """Remove SIP distortion from a single frame (chunk-sized).
        B-spline prefilters on GPU, then resamples to ideal gnomonic grid."""
        w, h = self.chunk_w, self.chunk_h

        # Upload source to GPU
        drv.memcpy_htod(self.g_ud_src, src_data.ravel().astype(np.float64))

        # B-spline prefilter in-place on GPU
        self.cu_prefilter_rows(
            self.g_ud_src, np.int32(w), np.int32(h),
            block=(256,1,1), grid=((h + 255) // 256, 1))
        self.cu_prefilter_cols(
            self.g_ud_src, np.int32(w), np.int32(h),
            block=(256,1,1), grid=((w + 255) // 256, 1))

        # Launch undistort kernel
        n = w * h
        self.cu_undistort(
            self.g_ud_dst, self.g_ud_src,
            np.int32(w), np.int32(h),
            np.int32(self.chunk_x0), np.int32(self.chunk_y0),
            block=(256,1,1), grid=((n + 255) // 256, 1))

        # Download
        drv.memcpy_dtoh(self.c_ud_dst, self.g_ud_dst)
        return self.c_ud_dst.reshape(src_data.shape).copy()

    def undistort_batch(self, sector_data, out=None):
        """Undistort an entire (N, H, W) stack in-place or into out array."""
        n_frames = sector_data.shape[0]
        if out is None:
            out = np.empty_like(sector_data)
        for i in range(n_frames):
            out[i] = self.undistort(sector_data[i])
            if i % 100 == 0:
                print(f"undistort {i}/{n_frames}")
        return out

# Version using streams, adapted from my version above by Claude Opus 4.6 on 2026 March 21. This is untested and may contain errors, but it should be close to working.
# class CUDAHcongridContainer:
#     def __init__(self):
#         self.initialized = False
#         self.ftype = np.float64
#         #self.ftype = np.float32
# 
#     # Runs on the first call to initialize kernels as a "JIT" approach.
#     def initialize(self, data, header): # loaded from data, header <<= fits.open("file.fits", header=True)
#         if self.initialized:
#             return
#         
#         self.cu_prefilter_rows = krnl.get_function("bspline3_prefilter_rows_f64")
#         self.cu_prefilter_cols = krnl.get_function("bspline3_prefilter_cols_f64")
#         self.cu_hcongrid_f64 = krnl.get_function("cuda_hcongrid_f64")
# 
#         dst_Mapping_Variables = wcs_variables_load_nonan(header, wcs_variables_validate(header))
#         dst_Mapping_Struct = wcs_assemble_struct(dst_Mapping_Variables)
#         self.dst_Mapping_Variables = dst_Mapping_Variables # Just in case we need it!
#         self.dst_Mapping_Struct = dst_Mapping_Struct # Just in case we need it!
# 
#         # Initialize constant memory parameters
#         g_dst_data, g_dst_data_size = krnl.get_global("dst_data")
#         g_src_data, g_src_data_size = krnl.get_global("src_data")
#         drv.memcpy_htod(g_dst_data, pack_TMS(dst_Mapping_Struct))
#         drv.memcpy_htod(g_src_data, pack_TMS(dst_Mapping_Struct)) # Preventing garbage
#         # incredible
#         #drv.memcpy_htod(g_dst_data, 
#         #     pack_TMS(wcs_assemble_struct(wcs_variables_load_nonan(header, wcs_variables_validate(header)))))
# 
#         # Initialize variable memory
#         c_dst = np.empty_like(data.ravel(), dtype=self.ftype)
#         g_dst = drv.mem_alloc(c_dst.nbytes)
#         g_src = drv.mem_alloc(c_dst.nbytes) # src and dst are the same size, so we can reuse the buffer size.
#         # We will copy the actual data in each call, but we need to allocate the GPU memory here.
# 
#         #c_dst[:] = data.ravel().astype(self.ftype) # Fill with trash to test
#         #drv.memcpy_htod(g_dst, c_dst) # this will be immediately deleted
# 
#         self.g_dst_data, self.g_dst_data_size = g_dst_data, g_dst_data_size
#         self.g_src_data, self.g_src_data_size = g_src_data, g_src_data_size
# 
#         self.g_dst, self.c_dst = g_dst, c_dst
#         self.g_src, self.c_src = g_src, c_dst.copy()
# 
#         self.initialized = True
#     
#     def hcongrid(self, src_data, src_header):
# 
#         if src_data.ravel().shape != self.c_dst.shape:
#             raise ValueError(f"Source data shape {src_data.shape} does not match destination shape {self.c_dst.shape}.")
# 
#         drv.memcpy_htod(self.g_src, src_data.ravel().astype(self.ftype))
# 
#         guh = wcs_assemble_struct(wcs_variables_load_nonan(src_header, wcs_variables_validate(src_header))) # This is the one that actually matters. We need to update the WCS parameters for the source image!
#         #print(guh)
#         drv.memcpy_htod(self.g_src_data, pack_TMS(guh))
# 
#         h, w = self.dst_Mapping_Struct.naxis_int_i
#         h, w = int(h), int(w)
# 
#         # Prefilter the source image in-place on GPU
#         self.cu_prefilter_rows(
#             self.g_src, np.int32(w), np.int32(h),
#             block=(256,1,1), grid=((h + 255) // 256, 1))
#         self.cu_prefilter_cols(
#             self.g_src, np.int32(w), np.int32(h),
#             block=(256,1,1), grid=((w + 255) // 256, 1))
# 
#         self.cu_hcongrid_f64(
#             self.g_dst,
#             self.g_src,
#             #block=(256,1,1), grid=((self.c_dst.size + 255) // 256, 1)
#             #block=(16,16,1), grid=((w + 15) // 16, (h + 15) // 16)
#             block=(256,1,1), grid=((w*h + 255) // 256, 1)
#         )
# 
#         #print("Yippee!")
#         #globals().update(locals()) # For interactive inspection
# 
#         drv.memcpy_dtoh(self.c_dst, self.g_dst)
#         return self.c_dst.reshape(src_data.shape).copy()
# 
#     # ── SIP-only undistortion (no frame alignment) ────────────────
# 
#     def initialize_undistort(self, header, chunk_shape, chunk_offset=(0, 0)):
#         """Set up GPU buffers for chunk-sized undistortion.
#         
#         chunk_shape:  (height, width) of the chunk, e.g. (512, 512) or (2048, 2048)
#         chunk_offset: (y0, x0) pixel offset of chunk origin in full CCD frame
#         """
#         self.cu_undistort = krnl.get_function("cuda_resample_undistort_f64")
# 
#         self.chunk_h, self.chunk_w = int(chunk_shape[0]), int(chunk_shape[1])
#         self.chunk_y0, self.chunk_x0 = int(chunk_offset[0]), int(chunk_offset[1])
#         n_pixels = self.chunk_h * self.chunk_w
#         nbytes = n_pixels * np.dtype(np.float64).itemsize
# 
#         # Upload this CCD's WCS to src_data constant memory
#         wcs_vars = wcs_variables_load_nonan(header, wcs_variables_validate(header))
#         wcs_struct = wcs_assemble_struct(wcs_vars)
#         self.undistort_wcs_struct = wcs_struct
#         g_src_data, _ = krnl.get_global("src_data")
#         drv.memcpy_htod(g_src_data, pack_TMS(wcs_struct))
# 
#         # Double-buffered GPU allocations for streamed batch processing
#         self.g_ud_src = [drv.mem_alloc(nbytes) for _ in range(2)]
#         self.g_ud_dst = [drv.mem_alloc(nbytes) for _ in range(2)]
# 
#         # Pinned host memory for async transfers (required for overlap)
#         self.c_ud_upload   = [drv.pagelocked_empty(n_pixels, np.float64) for _ in range(2)]
#         self.c_ud_download = [drv.pagelocked_empty(n_pixels, np.float64) for _ in range(2)]
# 
#         # Two streams for double-buffered pipelining
#         self.ud_streams = [drv.Stream() for _ in range(2)]
# 
#         self.undistort_initialized = True
# 
#     def undistort(self, src_data):
#         """Remove SIP distortion from a single frame (chunk-sized).
#         B-spline prefilters on GPU, then resamples to ideal gnomonic grid.
#         Synchronous — uses slot 0."""
#         w, h = self.chunk_w, self.chunk_h
#         g_src, g_dst = self.g_ud_src[0], self.g_ud_dst[0]
# 
#         # Upload source to GPU (synchronous)
#         drv.memcpy_htod(g_src, src_data.ravel().astype(np.float64))
# 
#         # B-spline prefilter in-place on GPU
#         self.cu_prefilter_rows(
#             g_src, np.int32(w), np.int32(h),
#             block=(256,1,1), grid=((h + 255) // 256, 1))
#         self.cu_prefilter_cols(
#             g_src, np.int32(w), np.int32(h),
#             block=(256,1,1), grid=((w + 255) // 256, 1))
# 
#         # Launch undistort kernel
#         n = w * h
#         self.cu_undistort(
#             g_dst, g_src,
#             np.int32(w), np.int32(h),
#             np.int32(self.chunk_x0), np.int32(self.chunk_y0),
#             block=(256,1,1), grid=((n + 255) // 256, 1))
# 
#         # Download (synchronous)
#         out = np.empty(n, dtype=np.float64)
#         drv.memcpy_dtoh(out, g_dst)
#         return out.reshape(src_data.shape)
# 
#     def undistort_batch(self, sector_data, out=None):
#         """Undistort an entire (N, H, W) stack using double-buffered CUDA streams.
# 
#         Pipeline per slot: upload_async → prefilter_rows → prefilter_cols → undistort → download_async
#         Two slots alternate, so slot 1's compute overlaps slot 0's transfer and vice versa.
#         """
#         n_frames = sector_data.shape[0]
#         frame_shape = sector_data.shape[1:]
#         if out is None:
#             out = np.empty_like(sector_data)
# 
#         w, h = self.chunk_w, self.chunk_h
#         n = w * h
#         blk = (256, 1, 1)
#         grid_pf_r = ((h + 255) // 256, 1)
#         grid_pf_c = ((w + 255) // 256, 1)
#         grid_ud   = ((n + 255) // 256, 1)
# 
#         # Pre-bake int32 args (avoid re-boxing every iteration)
#         i32_w, i32_h = np.int32(w), np.int32(h)
#         i32_x0, i32_y0 = np.int32(self.chunk_x0), np.int32(self.chunk_y0)
# 
#         pending = [None, None]  # frame index in-flight on each slot
# 
#         for i in range(n_frames):
#             slot = i & 1
#             stream = self.ud_streams[slot]
#             g_src  = self.g_ud_src[slot]
#             g_dst  = self.g_ud_dst[slot]
#             h_up   = self.c_ud_upload[slot]
#             h_dn   = self.c_ud_download[slot]
# 
#             # ── drain previous work on this slot before reusing buffers ──
#             if pending[slot] is not None:
#                 stream.synchronize()
#                 prev = pending[slot]
#                 out[prev] = h_dn.reshape(frame_shape)
#                 if prev % 100 == 0:
#                     print(f"undistort {prev}/{n_frames}")
# 
#             # ── stage into pinned upload buffer (CPU memcpy) ──
#             h_up[:] = sector_data[i].ravel()
# 
#             # ── enqueue: upload → prefilter → undistort → download ──
#             drv.memcpy_htod_async(g_src, h_up, stream)
# 
#             self.cu_prefilter_rows(g_src, i32_w, i32_h,
#                 block=blk, grid=grid_pf_r, stream=stream)
#             self.cu_prefilter_cols(g_src, i32_w, i32_h,
#                 block=blk, grid=grid_pf_c, stream=stream)
#             self.cu_undistort(g_dst, g_src, i32_w, i32_h, i32_x0, i32_y0,
#                 block=blk, grid=grid_ud, stream=stream)
# 
#             drv.memcpy_dtoh_async(h_dn, g_dst, stream)
#             pending[slot] = i
# 
#         # ── drain remaining ──
#         for slot in range(2):
#             if pending[slot] is not None:
#                 self.ud_streams[slot].synchronize()
#                 prev = pending[slot]
#                 out[prev] = self.c_ud_download[slot].reshape(frame_shape)
# 
#         print(f"undistort {n_frames}/{n_frames}")
#         return out


# Constant memory rbf fitter. This avoids the constant recomputation 
#     of the same kernel parameters for every chunk in every image.
class Rbf_parameter_fitter:
    def __init__(self) -> None:
        # ── Precompute sample coordinates (same for every chunk) ──────────
        sample_x = np.empty(sze, dtype=np.float64)
        sample_y = np.empty(sze, dtype=np.float64)
        _nd = 0
        for _jj in range(0, bxs + pix, pix):
            for _kk in range(0, bxs + pix, pix):
                sample_x[_nd] = min(_jj, bxs - 1)
                sample_y[_nd] = min(_kk, bxs - 1)
                _nd += 1
        self.sample_x = sample_x
        self.sample_y = sample_y

        # ── Below code is written by Claude Opus 4.6 on 2026 March 18. ───
        # Bootstrap shift/scale from scipy to guarantee exact match
        rng = np.random.default_rng(42)
        rng = rng.uniform(0, 1, size=(sze,)).astype(np.float64) # dummy values, we only care about the shift/scale
        dummy_rbf = scipy_interp.RBFInterpolator(
            list(zip(sample_x, sample_y)), rng,
            kernel='thin_plate_spline', smoothing=0.0, degree=1)
        fixed_shift = dummy_rbf._shift.copy()
        fixed_scale = dummy_rbf._scale.copy()

        # Build the LHS ourselves
        xy_ik = np.column_stack([sample_x, sample_y]) # (289, 2)
        xy_norm = (xy_ik - fixed_shift) / fixed_scale

        # I took over for Claude here.
        # Kernel matrix: thin-plate spline φ(r) = r² ln(r)
        delta_ijk = xy_ik[:, None, :] - xy_ik[None, :, :] # (289, 289, 2)
        r_ij = np.linalg.norm(delta_ijk, axis=-1)
        
        with np.errstate(divide='ignore', invalid='ignore'): # numpy is going to complain
            rbf_ij = np.where(0 < r_ij, r_ij**2 * np.log(r_ij), 0.0) # Handle r=0 case to avoid NaNs. This is mathematically correct since lim_{r→0} r² ln(r) = 0.

        # Polynomial: [1, x, y] for degree=1
        P = np.column_stack([np.ones(sze), xy_norm])  # (289, 3)
        nmonos = P.shape[1]

        # Augmented system
        lhs = np.zeros((sze + nmonos, sze + nmonos), dtype=np.float64)
        lhs[:sze, :sze] = rbf_ij
        lhs[:sze, sze:] = P
        lhs[sze:, :sze] = P.T

        # LU factorize ONCE — this is all of the O(n³) work
        LU, Pivot = scipy.linalg.lu_factor(lhs)
        
        # Verify against dummy
        rhs_verify = np.zeros((sze + nmonos, 1))
        rhs_verify[:sze, 0] = rng
        coeffs_verify = scipy.linalg.lu_solve((LU, Pivot), rhs_verify)
        assert np.allclose(coeffs_verify, dummy_rbf._coeffs), "LHS reconstruction mismatch!"
        print("LHS reconstruction mismatch:", np.max(np.abs(coeffs_verify - dummy_rbf._coeffs)))
        
        # Wow, this was literally exact, value for value. max error of *zero*.
        # Caching time! That's the whole point here.
        self.LU = LU
        self.Pivot = Pivot
        self.rhs = rhs_verify
        self._coeffs = coeffs_verify
        self._shift = fixed_shift
        self._scale = fixed_scale
        
    def __call__(self, values):
        # reshld_cuda_f64 = cuda_rbf(sample_x, sample_y, rbf._coeffs, rbf._shift, rbf._scale)
        self.rhs[:sze, 0] = values.ravel()
        self.rhs[sze:] = 0.0
        self._coeffs[:] = scipy.linalg.lu_solve((self.LU, self.Pivot), self.rhs).reshape(self._coeffs.shape)
        #return self._coeffs, self._shift, self._scale
        return self # so we can return 'rbf'

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

    # # ── Precompute sample coordinates (same for every chunk) ──────────
    # sample_x = np.empty(sze, dtype=float)
    # sample_y = np.empty(sze, dtype=float)
    # _nd = 0
    # for _jj in range(0, bxs + pix, pix):
    #     for _kk in range(0, bxs + pix, pix):
    #         sample_x[_nd] = min(_jj, bxs - 1)
    #         sample_y[_nd] = min(_kk, bxs - 1)
    #         _nd += 1
    rbf_parameter_fitter = Rbf_parameter_fitter() # Caching LU factorization!

    # ── All-chunk sky value arrays (preserved across phases) ──────────
    n_big_chunks = naxis * naxis  # 16 for 2048/512
    v_all = np.empty((n_big_chunks, sze), dtype=float)
    s_all = np.empty((n_big_chunks, sze), dtype=float)

    # ── Gather buffers for batched local sky sigma clip ────────────────
    max_window_pixels = (2 * lop) ** 2  # 128² = 16384
    n_total_windows = n_big_chunks * sze  # 16 × 289 = 4624
    gather_data = np.empty(n_total_windows * max_window_pixels, dtype=np.float64)
    gather_init_mask = np.zeros(n_total_windows * max_window_pixels, dtype=np.int32)

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

    rbf_timer_warmup = True
    cuda_rbf = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze)
    cuda_rbf_f64s = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze, mode="f64_storage")
    cuda_rbf_f32 = CUDARBFInterpolator(krnl, width=bxs, height=bxs, max_n_points=sze, mode="f32")
    cuda_sigclip = CUDASigmaClip(krnl, max_n=n_chunk)
    cuda_sigclip_f32 = CUDASigmaClipF32(krnl, max_n=n_chunk)
    n_chunks_total = (axs // bxs) ** 2  # 16 for 2048/512
    cuda_sigclip_batched = CUDASigmaClipBatched(krnl, n_chunks=n_chunks_total, chunk_size=n_chunk)
    # Batched sigma clip for all local sky windows (4624 windows × 16384 pixels each)
    cuda_sigclip_local = CUDASigmaClipBatched(krnl, n_chunks=n_total_windows, chunk_size=max_window_pixels)
    cuda_hcongrid_container = CUDAHcongridContainer() # Lazy initialization of CUDA kernels and buffers for hcongrid

    #views = []
    frames = []
    frame_completion_times = np.zeros(nfiles, dtype=np.float64)
    global_t0 = time.time()
    timers = ProfileTimer()
    globals().update(locals())

    #begin cleaning
    for ii in range(0, nfiles):
        with timers['Frame Loop']:
            # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
            #if ii == 2:
            #if ii == 100:
            #    break
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
            if not outpath.exists() or ii == 0: # We can optionally reprocess the first file to warm up the caches and get more accurate timing on subsequent files.
            #if True:
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

                try:
                    wcs_variables_validate(header) # should be near instant
                except ValueError as e:
                    print(f"WCS validation failed for {files[ii]}: {e}")
                    continue # Skip it!

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
                            #print(f"{axs = }, {bxs = }, {oo = }, {ee = }")
                            with timers['image_chunking']:
                                img = bigimg[ee:ee+bxs, oo:oo+bxs] #split the image into small subsections
                                #sc_cimg[:] = bigimg[ee:ee+bxs, oo:oo+bxs].ravel() #split the image into small subsections.ravel()
                            
                            # warm start numba + CUDA sigma clip
                            # if oo == 0 and ee == 0: #if tts == 0:
                            #     sc_cimg[:] = bigimg[:bxs,:bxs].ravel() #split the image into small subsections.ravel()
                            #     cnum, clow, chigh, cstd = sigma_clip_mask_numba(sc_cimg, sc_mask, 2.5, 2.5)

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

                # ── Phase 1: Gather all local sky windows ─────────────────────────
                # Pack all 4624 overlapping windows (16 chunks × 289 per chunk)
                # into a flat buffer for batched CUDA sigma clipping.
                # Each window is up to 128×128 = 16384 pixels; edge windows are
                # smaller and padded with mask=0.
                with timers['local_sky_gather']:
                    gather_init_mask[:] = 0  # reset: padding = invalid
                    win_idx = 0
                    for oo in range(0, axs, bxs):
                        for ee in range(0, axs, bxs):
                            img = bigimg[ee:ee+bxs, oo:oo+bxs]
                            for jj in range(0, bxs + pix, pix):
                                for kk in range(0, bxs + pix, pix):
                                    il = max(jj - lop, 0)
                                    ih = min(jj + lop, bxs - 1)
                                    jl = max(kk - lop, 0)
                                    jh = min(kk + lop, bxs - 1)
                                    c = img[jl:jh, il:ih]
                                    overlap[ee+jl:ee+jh, oo+il:oo+ih] += 1
                                    c_flat = c.ravel()
                                    n_pix = len(c_flat)
                                    offset = win_idx * max_window_pixels
                                    gather_data[offset:offset + n_pix] = c_flat
                                    gather_init_mask[offset:offset + n_pix] = 1
                                    win_idx += 1
                    #print(f"Gathered {win_idx} local sky windows "
                    #      f"({win_idx * max_window_pixels * 8 / 1e6:.1f} MB data)")

                # ── Phase 2: Batched CUDA sigma clip ──────────────────────────────
                # All 4624 windows sigma-clipped in parallel on GPU.
                # Each window gets its own independent (mean, std, lo, hi).
                with timers['local_sky_batched_sigmaclip']:
                    clip_masks, clip_counts, clip_los, clip_his, clip_means, clip_stds = \
                        cuda_sigclip_local(gather_data, clip_low=2.5, clip_high=2.5,
                                        init_mask=gather_init_mask)

                #if True:  # warm start for the timers in Phase 3 (numba extraction of medians/stds)
                #    mask_flat_i32 = clip_masks.ravel().astype(np.int32)
                #    all_medians, all_stds = extract_medians_from_masks(
                #        gather_data, mask_flat_i32, n_total_windows, max_window_pixels)
                #    v_all[:] = all_medians.reshape(n_big_chunks, sze)
                #    s_all[:] = all_stds.reshape(n_big_chunks, sze)

                # ── Phase 3: Extract per-window median and std ────────────────────
                # Numba-parallel: sort survivors per widnow, compute median + std.
                with timers['local_sky_extract']:
                    mask_flat_i32 = clip_masks.ravel().astype(np.int32)
                    all_medians, all_stds = extract_medians_from_masks(
                        gather_data, mask_flat_i32, n_total_windows, max_window_pixels)
                    v_all[:] = all_medians.reshape(n_big_chunks, sze)
                    s_all[:] = all_stds.reshape(n_big_chunks, sze)


                with timers['total_background_subtraction_chunking']:
                    for ci in range(n_big_chunks):
                        oo = (ci // naxis) * bxs
                        ee = (ci % naxis) * bxs
                        #print(f"{axs = }, {bxs = }, {oo = }, {ee = }")
                        sig = sbk[oo//bxs, ee//bxs]

                        # Non-destructive repair: v_all/s_all preserved, v_clean is new
                        with timers['bad_sky_repair']:
                            v_clean = repair_bad_sky_values(
                                rbf_parameter_fitter.sample_x, rbf_parameter_fitter.sample_y,
                                v_all[ci], s_all[ci], sig)

                        # with timers['rbf_creation']:
                        #     rbf = scipy_interp.RBFInterpolator(
                        #         list(zip(rbf_parameter_fitter.sample_x, rbf_parameter_fitter.sample_y)), v_clean,
                        #         kernel='thin_plate_spline', smoothing=0.0, degree=1)
                        #     globals().update(locals())
                        with timers['custom_rbf_creation']:
                            rbf = rbf_parameter_fitter(v_clean) # This updates the internal coeffs, but the object itself is the same, so we can reuse CUDA buffers.
                        #with timers['rbf_interpolation']:
                        #    globals().update(locals())

                        #if rbf_timer_warmup:
                        #    print("Warming up RBF interpolation timer with a single evaluation...")
                        #    reshld = custom_rbf_eval(sample_x, sample_y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)
                        #    print("Warming up CUDA RBF interpolation...")
                        #    reshld_cuda = cuda_rbf(sample_x, sample_y, rbf._coeffs, rbf._shift, rbf._scale)
                        #    rbf_timer_warmup = False

                        #with timers['custom_rbf_interpolation']:
                        #    reshld = custom_rbf_eval(sample_x, sample_y, XI, YI, rbf._coeffs, rbf._shift, rbf._scale)
                        #    globals().update(locals())

                        with timers['cuda_rbf_f64']:
                            reshld_cuda_f64 = cuda_rbf(rbf.sample_x, rbf.sample_y, rbf._coeffs, rbf._shift, rbf._scale)
                            globals().update(locals())

                        #with timers['cuda_rbf_f64s']:
                        #    reshld_cuda_f64s = cuda_rbf_f64s(sample_x, sample_y, rbf._coeffs, rbf._shift, rbf._scale)
                        #    globals().update(locals())

                        #with timers['cuda_rbf_f32']:
                        #    reshld_cuda_f32 = cuda_rbf_f32(sample_x, sample_y, rbf._coeffs, rbf._shift, rbf._scale)
                        #    globals().update(locals())

                        with timers['residual_image_addition']:
                            #reshld_OLD = reshld.copy() # Fake for performance reasons :)
                            #res_orig[ee:ee+bxs, oo:oo+bxs] = reshld_OLD
                            #res[ee:ee+bxs, oo:oo+bxs] = reshld
                            res_cuda_f64[ee:ee+bxs, oo:oo+bxs]  = reshld_cuda_f64
                            #res_cuda_f64s[ee:ee+bxs, oo:oo+bxs] = reshld_cuda_f64s
                            #res_cuda_f32[ee:ee+bxs, oo:oo+bxs]  = reshld_cuda_f32




            
                with timers['sky_gradient_subtraction']:
                    #subtract the sky gradient and add back the median background
                    #sub = bigimg-res_orig
                    #sub = bigimg-res 
                    #sub = bigimg-res_cuda_f32 # Switch to res from the original. It should be faster to iterate now!
                    sub = bigimg-res_cuda_f64
                    sub = sub + mbck

                #align the image
                # NOTE: `hcongrid` was originally used for alignment. If available,
                # uncomment the import at the top and ensure the function is on PATH.
                with timers['hcongrid_alignment']:
                    #algn = hcongrid(sub, header, rhead) # Do I even need this? Its mapping nothing.
                    if not cuda_hcongrid_container.initialized:
                        print("Initializing CUDA hcongrid container...")
                        cuda_hcongrid_container.initialize(sub, header)
                        rhead = header.copy() # In case the header is modified during alignment, we want to preserve the original for the next iterations.
                    algn = cuda_hcongrid_container.hcongrid(sub, header) 

                    #if ii == 0:
                    #    # Frame 0 IS the reference — skip the expensive WCS
                    #    # round-trip that would just map the image onto itself.
                    #    algn = sub
                    #else:
                    #    algn = hcongrid(sub, header, rhead)
                    
                    #algn = sub.copy() # Placeholder for alignment step; always a no-op?

                    ## Save a cmap to a view as a frame
                    # save the first 50 frames
                    #frames.append(algn.copy())

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
                    print(f"Wrote {outpath.name}.")

                #stop the watch
                fn = time.time()
                frame_completion_times[ii] = fn - global_t0
                # predict an based on the last 10 frames
                # Display an error if ii < 2, since we need at least 2 to make a linear prediction.
                # ...assuming this is linear haha
                ETA = "[ETA: Calculating...]"
                if ii >= 2:
                    recent_nums = (np.arange(nfiles) + 1)[:ii][-50:]
                    recent_time = frame_completion_times[:ii][-50:]  # last 50 frames

                    # recent_nums = frame indices, recent_time = wall-clock timestamps
                    eta_coeffs = np.polyfit(recent_nums, recent_time, 1)  # frame → time
                    ETA_seconds = np.polyval(eta_coeffs, nfiles) - frame_completion_times[ii]

                    ETA = humanize.precisedelta(ETA_seconds, format="%0.3f")
                    ETA = f"[ETA: {ETA}]"
                Time_Taken = humanize.precisedelta(fn - global_t0, format="%0.3f")
                Time_Taken = f"[Time taken: {Time_Taken}]"

                prefix = f"[{ii+1}/{nfiles}]"
                print(prefix, ETA, Time_Taken)
                print(f'Background subtraction for {files[ii]} finished in {fn-st}s.')
                print()
                print(timers)
                print("-"*72)
                print("-"*72)
                print()
            

    print('All done! See ya later alliagtor.')
    global_t1 = time.time()
    print(f'Total processing time: {global_t1 - global_t0}s')

    # print the timer results
    print(timers)
    globals().update(locals())

big_cleanup()

#cv2.imshow("Aligned Image", algn.astype(np.float32) / np.max(algn))  # Normalize for display
# I want to update this imshow so I can monitor progress of a video encoder! How do I do this?
# Copilot: You can use OpenCV's `imshow` in a loop to update the displayed image. Here's how you can modify your code to show the aligned image in real-time as it's processed:
# After processing each frame and obtaining the `algn` image, add the following lines to display it:
# ```python
# cv2.imshow("Aligned Image", algn.astype(np.float32) / np.max(algn))  # Normalize for display
# if cv2.waitKey(1) & 0xFF == ord('q'):  #     break  # Exit the loop if 'q' is pressed
# ```


#raise Exception("Done with big cleanup. Stopping here to avoid cleaning up the code below. (debugging and visualization)")

########################################################################################################################
########################################################################################################################
########################################################################################################################
# MK_Master

from pathlib import Path
ROOT = Path('C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/')
cdedir = ROOT / "DIA" / "routines" / "Python"
#rawdir = ROOT / "DIA_TEMP" / "raw"
rawdir = ROOT / "TESS_sector_4"
clndir = ROOT / "DIA_TEMP" / "clean3"
mstdir = ROOT / "DIA_TEMP" / "master"

# ensure the output directories exist
ROOT = Path(ROOT)#.mkdir(parents=True, exist_ok=True)
cdedir = Path(cdedir)#.mkdir(parents=True, exist_ok=True)
rawdir = Path(rawdir)#.mkdir(parents=True, exist_ok=True)
clndir = Path(clndir)#.mkdir(parents=True, exist_ok=True)
mstdir = Path(mstdir)#.mkdir(parents=True, exist_ok=True

def get_file_list(camera=None, ccd=None):
    camera = str(camera) if camera is not None else None
    ccd = str(ccd) if ccd is not None else None

    from pathlib import Path
    ROOT = Path('C:/Users/Joe/Desktop/Projects/2026_Spring/DIA/')
    cdedir = ROOT / "DIA" / "routines" / "Python"
    #rawdir = ROOT / "DIA_TEMP" / "raw"
    rawdir = ROOT / "TESS_sector_4"
    #clndir = ROOT / "DIA_TEMP" / "clean"
    clndir = ROOT / "DIA_TEMP" / "clean3"
    mstdir = ROOT / "DIA_TEMP" / "master"

    # ensure the output directories exist
    ROOT = Path(ROOT)#.mkdir(parents=True, exist_ok=True)
    cdedir = Path(cdedir)#.mkdir(parents=True, exist_ok=True)
    rawdir = Path(rawdir)#.mkdir(parents=True, exist_ok=True)
    clndir = Path(clndir)#.mkdir(parents=True, exist_ok=True)
    mstdir = Path(mstdir)#.mkdir(parents=True, exist_ok=True

    files = [f for f in clndir.glob("*.fits") if isfile(join(clndir, f))] #gets the relevant files with the proper extension

    # Cull files that don't match the specified camera and CCD
    #camera, ccd = '1', '4'
    def get_camera_and_ccd(f):
        # with fits.open(f, memmap=True) as hdul:
        #     camera = hdul[1].header['CAMERA']
        #     ccd = hdul[1].header['CCD']
        # Grab these from the filename instead due to perf loss
        # tess2018297215939-s0004-1-4-0124-s_ffic.fits
        filename = f.stem  # gets filename without extension
        parts = filename.split('-')
        cameraNum = parts[2]
        ccdNum = parts[3]
        return cameraNum, ccdNum

    def filter_file(f):
        #img_data, img_header = fits.getdata(f, header=True)
        #img_header = fits.getheader(f)
        # Get these from the data's header manually
        camera_val, ccd_val = get_camera_and_ccd(f)
        result = True
        if camera is not None:
            result = result and (camera_val == camera)
        if ccd is not None:
            result = result and (ccd_val == ccd)
        return result
    files = list(filter(filter_file, files))
    files.sort()
    return files

# def make_file_generator(files):
#     for f in files:
#         yield fits.getdata(f, header=True)

def make_file_generator(files):
    for f in files:
        data, header = fits.getdata(f, header=True)
        yield data.astype(np.float64), header

# Lets make a maximum janky variant of both mk_master AND cmb_tmp.
def mk_master():
    
    # Load one file
    files = get_file_list(camera=1, ccd=4) # Filter for camera 1, CCD 4
    data, header = fits.getdata(files[0], header=True)
    print(f"{data.shape = }, {data.dtype = }")

    # malloc 30gb lol
    sector_data = np.empty((len(files), *data.shape), dtype=data.dtype)
    sector_headers = []

    # add this one in
    sector_data[0] = data  # First frame loaded, rest is uninitialized
    sector_headers.append(header)

    # draw the rest of the fucking owl
    t0 = time.time()
    for ii, (data, header) in enumerate(make_file_generator(files[1:]), start=1):
        sector_data[ii] = data
        sector_headers.append(header)
    t1 = time.time()
    print(f"Loaded {len(files)} files in {humanize.precisedelta(t1 - t0)}.")
    
    # Compute the mk_master median
    sector_median = np.empty_like(sector_data[0])
    t0 = time.time()
    for i in range(sector_data.shape[1]): # Slice into spans to prevent my ram from exploding more
        sector_median[i] = np.median(sector_data[:,i,:], axis=0)
        if i % 256 == 0:
            print(f"Computed median for row {i}/{sector_data.shape[1]}")
    t1 = time.time()
    print(f"Computed median in {humanize.precisedelta(t1 - t0)}.")

    # aight lets make the astronomers happy
    expt = np.array([header["EXPOSURE"] for header in sector_headers])
    sector_hdu = fits.PrimaryHDU(sector_median)
    sector_hdu.header["NUMCOMB"] = len(files)
    sector_hdu.header["EXPOSURE"] = np.median(expt) # NOTE: I CHANGED THIS FROM EXPTIME - Joe Kessler 2026-03-19

    # Build the master filename
    # files[0].stem : 'tess2018292095939-s0004-1-4-0124-s_ffic_sa'
    # We want to transform this into tess_median-s0004-1-4-n<numcomb>.fits
    filename = "-".join(["tess_median"] + files[0].stem.split('-')[1:4] + [f"n{len(files)}"]) + ".fits"
    sector_hdu.writeto(mstdir/filename, overwrite=True)
    print("Wrote master frame to", mstdir/filename)

    globals().update(locals())

#mk_master()

raise Exception("Done with mk_master. Stopping here to avoid cleaning up the code below. (debugging and visualization)")

# Yoink! <|/code_to_edit|> <- Stole your thing







# Profiler
#import cProfile

# with cProfile.Profile() as pr:
#     big_cleanup()
#     
#     import pstats
#     from pstats import SortKey
#     p = pstats.Stats(pr)
#     p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(20)

#import matplotlib.pyplot as plt

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


# def slog(img):
#     result = np.log(img)
#     # Find the value that is the smallest non nan, non inf value in the image
#     mask = np.isfinite(result) 
#     vmin = result[mask].min()
#     result[~mask] = vmin
#     return result


# ── Benchmark: all modes vs scipy reference (reshld_OLD) ─────────────
# Per-chunk precision (last chunk)
# print("\n" + "="*72)
# print("  PRECISION BENCHMARK (last chunk, vs scipy RBFInterpolator)")
# print("="*72)
# ref = reshld_OLD  # scipy is the f64 ground truth
# for label, arr in [("numba/CPU  ", reshld),
#                    ("CUDA f64   ", reshld_cuda_f64),
#                    ("CUDA f64s  ", reshld_cuda_f64s),
#                    ("CUDA f32   ", reshld_cuda_f32)]:
#     abs_err = np.max(np.abs(arr - ref))
#     rel_err = np.max(np.abs((arr - ref) / ref))
#     print(f"  {label}  max|abs|={abs_err:.3e}  max|rel|={rel_err:.3e}")
# print("="*72)
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
