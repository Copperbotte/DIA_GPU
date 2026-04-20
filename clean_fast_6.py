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
#rawdir = ROOT / "TESS_sector_4"
rawdir = ROOT / "TESS_sector_19"
#clndir = ROOT / "DIA_TEMP" / "clean"
#clndir = ROOT / "DIA_TEMP" / "clean3"
clndir = ROOT / "DIA_TEMP" / "clean_s19"

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
#sector, camera, ccd = '0004', '1', '4' # Sector 4
sector, camera, ccd = '0019', '1', '1' # Sector 19
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
    int chunk_x0, int chunk_y0,
    double scale
){
    // Removes SIP distortion only — no frame-to-frame alignment.
    // Uses src_data constant memory for WCS (same image's own WCS).
    // Output pixels are in ideal gnomonic coordinates (linear w.r.t. sky).
    TESS_Mapping_Struct* wcs = &src_data;

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= chunk_w * chunk_h) return;

    double x = (double)((idx % chunk_w) + chunk_x0) - (wcs->ref_px_coord[0] - 1.0);
    double y = (double)((idx / chunk_w) + chunk_y0) - (wcs->ref_px_coord[1] - 1.0);
    double uv[2] = {x, y};

    double uv_dist[2];
    vec2_distort_sip_inv_newton_cuda_f64(uv_dist, uv, wcs->fwd_AB, wcs->inv_AB);

    // TESTING
    for(int i=0; i<2; i++)
        uv_dist[i] *= scale;//(uv_dist[i] - uv[i])*0.5; // Shrink the distortion so the full image should appear in the output

    uv_dist[0] += ((wcs->ref_px_coord[0] - 1.0) - (double)chunk_x0);
    uv_dist[1] += ((wcs->ref_px_coord[1] - 1.0) - (double)chunk_y0);

    // B-spline sample the prefiltered source image
    double lum = texture_cubic_b(src_img, uv_dist[0], uv_dist[1], chunk_w, chunk_h);

    dst_img[idx] = lum;

    // These coordinates represent the SIP corrected position of the output pixel in the source image's pixel coordinate system.
    // There's a small bug. Lets sketch the forward pass to see if we can find it.
    // 
    // Forward pass:
    // 1. xy as pixel coordinates
    // 2. uv = xy - (CRPIX-1) // offset the zero pixel.
    // 3. distort = SIP_fwd(uv)
    // 4. uv_dist = uv + distort
    // 5. final = uv_dist + (CRPIX-1)
    // Note: CRPIX should commute through the expression:
    //     1. xy = CRPIX-1
    //     2. uv = xy - (CRPIX-1) = CRPIX-1 - (CRPIX-1) = 0
    //     3. distort = SIP_fwd(uv) = SIP_fwd(0) = 0
    //     4. uv_dist = uv + distort = 0 + 0 = 0
    //     5. final = uv_dist + (CRPIX-1) = 0 + (CRPIX-1) = (CRPIX-1).
    // The bug below is likely from this distortion. But, perhaps there isn't one!
    // It seems distorted only on one side.
    // 
    // Inverse pass:
    // 5. final as pixel coordinates
    // 4. uv_dist = final - (CRPIX-1)
    // 3. distort = SIP_inv(uv_dist)
    // 2. uv = uv_dist + distort
    // 1. xy = uv + (CRPIX-1)
    // Note: the implementation of SIP_inv includes step 2.



    // // Output pixel position in full-frame CCD coordinates
    // double full_x = (double)(x + chunk_x0);
    // double full_y = (double)(y + chunk_y0);
// 
    // // Offset from CRPIX — these are "ideal" undistorted pixel offsets
    // double uv_ideal[2] = {
    //     full_x - (wcs->ref_px_coord[0] - 1.0),
    //     full_y - (wcs->ref_px_coord[1] - 1.0)
    // };
// 
    // // SIP inverse Newton: find distorted uv such that uv_dist + SIP_fwd(uv_dist) = uv_ideal
    // double uv_dist[2];
    // vec2_distort_sip_inv_newton_cuda_f64(uv_dist, uv_ideal, wcs->fwd_AB, wcs->inv_AB);
// 
    // // Convert back to pixel coords, relative to the chunk origin
    // double src_x = uv_dist[0] + (wcs->ref_px_coord[0] - 1.0) - (double)chunk_x0;
    // double src_y = uv_dist[1] + (wcs->ref_px_coord[1] - 1.0) - (double)chunk_y0;
// 
    // // B-spline sample the prefiltered source image
    // double lum = texture_cubic_b(src_img, src_x, src_y, chunk_w, chunk_h);
// 
    // dst_img[idx] = lum;
}

// ─── Colormap + BGR prep for cv2 display ────────────────────────────
// Input:  grayscale double image (H×W), RGB LUT (256×3, doubles in [0,1])
// Output: BGR uint8 image (H×W×3), ready for cv2.imshow
__global__
void cu_prepcv2(
    unsigned char* colorbuffer,
    const double* img,
    const double* LUT,
    const double lo, const double inv_range,
    int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= height * width) return;

    double val = img[idx];
    if (isnan(val) || isinf(val)) val = lo;

    int lidx = (int)(255.0 * ((val - lo) * inv_range));
    if (lidx < 0)   lidx = 0;
    if (lidx > 255) lidx = 255;

    // LUT is RGB, output is BGR for cv2
    int out = idx * 3;
    int lut = lidx * 3;
    colorbuffer[out + 0] = (unsigned char)(LUT[lut + 2] * 255.0);
    colorbuffer[out + 1] = (unsigned char)(LUT[lut + 1] * 255.0);
    colorbuffer[out + 2] = (unsigned char)(LUT[lut + 0] * 255.0);
}


// ── Affine resample (center-relative, B-spline) ──────────────────────
// Applies  dst(x) = src( Λ⁻¹·(x - c) + c )  where c = image center.
// The source must already be B-spline prefiltered (bspline3_prefilter_*).
// Subpixel translation is NOT included — do that via FFT phasor outside.
//
// Constant memory layout for c_affine[8]:
//   [0..3] = Λ⁻¹ row-major: { Λ⁻¹₀₀, Λ⁻¹₀₁, Λ⁻¹₁₀, Λ⁻¹₁₁ }
//            (row,col) convention: src_row = Λ⁻¹₀₀·dr + Λ⁻¹₀₁·dc + cy
//                                  src_col = Λ⁻¹₁₀·dr + Λ⁻¹₁₁·dc + cx
//   [4..5] = center:  { cy, cx }  (typically H/2, W/2)
//   [6..7] = shape:   { H, W }    (as doubles for convenience)
__constant__ double c_affine[8];

__global__
void cuda_affine_bspline_f64(
    double* __restrict__ dst_img,
    const double* __restrict__ src_img,
    int height, int width, int mult
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= (height*mult) * (width*mult)) return;

    int col = idx % (width * mult);
    int row = idx / (width * mult);

    double fm = (double)mult;

    // Offset from center
    //double dr = ((double)row/(double)mult) - c_affine[4];
    //double dc = ((double)col/(double)mult) - c_affine[5];
    double dr = ((double)row / fm) - c_affine[4];
    double dc = ((double)col / fm) - c_affine[5];

    // Apply Λ⁻¹ (row-major)
    double src_row = c_affine[0] * dr + c_affine[1] * dc + c_affine[4];
    double src_col = c_affine[2] * dr + c_affine[3] * dc + c_affine[5];

    // B-spline sample (note: texture_cubic_b takes (img, x, y, w, h)
    // where x=col, y=row)
    dst_img[idx] = texture_cubic_b(src_img, src_col, src_row, width, height);
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

    def undistort(self, src_data, scale=1.0):
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
            np.float64(scale),
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

    # ── Affine resample (center-relative, matrix only) ──────────────

    def initialize_affine(self, height, width, mult=4):
        """Set up GPU buffers for affine resampling of (height, width) images.
        
        Call once; then call affine_resample() per frame.
        The B-spline prefilter kernels are shared with hcongrid/undistort.
        mult is the resolution scaling of the output.
        """
        self.cu_affine = krnl.get_function("cuda_affine_bspline_f64")
        self.g_c_affine, _ = krnl.get_global("c_affine")

        if not hasattr(self, 'cu_prefilter_rows'):
            self.cu_prefilter_rows = krnl.get_function("bspline3_prefilter_rows_f64")
            self.cu_prefilter_cols = krnl.get_function("bspline3_prefilter_cols_f64")

        self.aff_h, self.aff_w = int(height), int(width)
        n = self.aff_h * self.aff_w
        self.mult = int(mult)
        self.g_aff_src = drv.mem_alloc(n * 8) # wait. 8? # 8 bytes per float64
        self.g_aff_dst = drv.mem_alloc(n * 8 * mult * mult) # output is larger by mult²
        self.c_aff_dst = np.empty(n * mult * mult, dtype=np.float64)
        self.affine_initialized = True

    def affine_resample(self, src_data, Lambda_inv, shift=(0.0, 0.0)):
        """Resample src_data by center-relative affine matrix Λ⁻¹ using cubic B-spline.
        
        Parameters
        ----------
        src_data   : (H, W) float64 array
        Lambda_inv : (2, 2) inverse affine matrix in (row, col) convention.
                     Identity = no-op.  For a rotation by θ about center:
                         Λ⁻¹ = [[cos θ,  sin θ],
                                [-sin θ,  cos θ]]
        
        Returns
        -------
        dst : (H, W) float64 — resampled image.
        
        Notes
        -----
        Subpixel translation is NOT included here — apply that separately via
        FFT phasor shift (fft_affine_warp) for full float64 precision.
        The B-spline prefilter converts samples → coefficients so that
        texture_cubic_b recovers a C²-smooth interpolant.
        """
        h, w = self.aff_h, self.aff_w

        # Pack constant memory: [Λ⁻¹₀₀, Λ⁻¹₀₁, Λ⁻¹₁₀, Λ⁻¹₁₁, cy, cx, H, W]
        c_buf = np.array([
            Lambda_inv[0, 0], Lambda_inv[0, 1],
            Lambda_inv[1, 0], Lambda_inv[1, 1],
            h / 2.0, w / 2.0,
            float(h), float(w)
        ], dtype=np.float64)
        drv.memcpy_htod(self.g_c_affine, c_buf)

        # Upload source
        drv.memcpy_htod(self.g_aff_src, src_data.ravel().astype(np.float64))

        # B-spline prefilter in-place on GPU
        self.cu_prefilter_rows(
            self.g_aff_src, np.int32(w), np.int32(h),
            block=(256, 1, 1), grid=((h + 255) // 256, 1))
        self.cu_prefilter_cols(
            self.g_aff_src, np.int32(w), np.int32(h),
            block=(256, 1, 1), grid=((w + 255) // 256, 1))

        # Launch affine kernel
        n = h * w * self.mult * self.mult
        self.cu_affine(
            self.g_aff_dst, self.g_aff_src,
            np.int32(h), np.int32(w), np.int32(self.mult),
            block=(256, 1, 1), grid=((n + 255) // 256, 1))

        # Download
        drv.memcpy_dtoh(self.c_aff_dst, self.g_aff_dst)
        #return self.c_aff_dst.reshape((h*self.mult, w*self.mult)).copy()
        return self.c_aff_dst.reshape((h,self.mult, w,self.mult)).mean(axis=(1, 3)).copy() # Downsample by averaging to get back to (h, w)

class CUDAPrepcv2:
    """GPU-accelerated colormap + BGR conversion for cv2 display."""

    def __init__(self, LUT):
        """LUT: (256, 3) float64 RGB colormap, e.g. np.array(cmap.colors)."""
        self.cu_prepcv2 = krnl.get_function("cu_prepcv2")
        lut_flat = LUT.ravel().astype(np.float64)
        self.g_LUT = drv.mem_alloc(lut_flat.nbytes)
        drv.memcpy_htod(self.g_LUT, lut_flat)
        self._bufs = {}

    def _get_bufs(self, h, w):
        key = (h, w)
        if key not in self._bufs:
            n = h * w
            g_img = drv.mem_alloc(n * 8)
            g_out = drv.mem_alloc(n * 3)
            c_out = np.empty((h, w, 3), dtype=np.uint8)
            self._bufs[key] = (g_img, g_out, c_out)
        return self._bufs[key]

    def __call__(self, img, Min=None,Max=None):
        """Convert grayscale double image → BGR uint8 via LUT on GPU."""
        h, w = img.shape[:2]
        g_img, g_out, c_out = self._get_bufs(h, w)
        flat = np.ascontiguousarray(img.ravel(), dtype=np.float64)
        drv.memcpy_htod(g_img, flat)
        lo = float(np.nanmin(img)) if Min is None else float(Min)
        hi = float(np.nanmax(img)) if Max is None else float(Max)
        inv_range = 1.0 / (hi - lo) if hi != lo else 0.0
        n = h * w
        self.cu_prepcv2(
            g_out, g_img, self.g_LUT,
            np.float64(lo), np.float64(inv_range),
            np.int32(h), np.int32(w),
            block=(256, 1, 1), grid=((n + 255) // 256, 1))
        drv.memcpy_dtoh(c_out, g_out)
        return c_out

    def imshow(self, img, title='tess'):
        bgr = self(img)
        cv2.imshow('tess', bgr)
        cv2.setWindowTitle('tess', title)
        cv2.waitKey(1)


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




def slog(img):
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.log(img)
    # Find the value that is the smallest non nan, non inf value in the image
    mask = np.isfinite(result) 
    #vmin = result[mask].min() # ValueError: zero-size array to reduction operation minimum which has no identity
    vmin = result[mask].min() if np.any(mask) else 0
    result[~mask] = vmin
    return result

def slog_asinh(img, scale=1.0):
    return np.arcsinh(img * scale)

# Simple log luminance tonemapper. Taken from visualizer.py.
def tonemap(img, hstart=0, hend=-1, bins=4096, fltr='log'):
    with np.errstate(divide='ignore', invalid='ignore'):
        #ld = np.log(img)
        if fltr == 'log':
            ld = slog(img)
        else:
            ld = img
        #ld = slog_asinh(img)
        #ld = img
    ld_fltr = ld[~np.isnan(ld) & ~np.isinf(ld)]
    ld_nan = np.nan_to_num(ld, nan=np.nanmax(ld_fltr), neginf=np.nanmin(ld_fltr), posinf=np.nanmax(ld_fltr))
    histo, bin_edges = np.histogram(ld_nan[hstart:hend], bins=bins)
    #histo, bin_edges = np.histogram(ld_nan, bins=1024)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    cs_histo = np.cumsum(histo)
    cdf_histo = cs_histo.astype(np.float64) / cs_histo[-1].astype(np.float64)
    ld_mapped = np.interp(ld_nan, bins, cdf_histo)
    return ld_mapped

# tess_geo[1036]
# array([-1.79566938e+01, -1.07463152e+01, -7.66898922e+00,  2.45843671e+06])
# tess_geo[1035]
# array([-1.83195543e+01, -1.04503658e+01, -7.97285965e+00,  2.45843669e+06])
# # I want to print these not in scientific notation.
# with np.printoptions(precision=2, suppress=True):
#     print(tess_geo[1036])
#     print(tess_geo[1035])
# 
# And estimated velocities by finite difference:
# Actually, lets do something more stable than 2 point finite difference. Let's do a 5-point stencil, which should be more stable and less noisy. The formula for the 5-point stencil is:
# dt = 0.020832635927945375
# v[i] = (tess_geo[i-2] - 8*tess_geo[i-1] + 8*tess_geo[i+1] - tess_geo[i+2]) / (12*dt)
# tess_geo is in units of [x,y,z,time].
# What's this stencil called?
# 
# 


import matplotlib.pyplot as plt
cmap = plt.get_cmap("viridis")

# views[0].shape = [2048, 2048, 4]
# list(reversed(views[0].shape))[1:]) -> [4, 2048, 2048][1:] -> [2048, 2048]

def write_cv2(name, views, fps=60):
    import cv2
    frames = [cv2.cvtColor(view, cv2.COLOR_RGBA2BGR) for view in views]
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    writer = cv2.VideoWriter(name, fourcc, fps, list(reversed(views[0].shape))[1:])
    for frame in frames:
        writer.write(frame)
    writer.release()
    print("wrote", name)


def compress_ffmpeg(name, name2, audiosrc=None):
    command = 'ffmpeg -i "%s" -c:v libx264 -b:v 5M -preset veryslow -crf 22 -y "%s"'%(name, name2)
    if type(audiosrc) != type(None):
        command = 'ffmpeg -i "%s" -i "%s" -c:v libx264 -b:v 5M -preset veryslow -crf 22 -c:a aac -strict experimental -y "%s"'%(name, audiosrc, name2)
    print(">", command)
    #command = command.split(' ')
    #import subprocess as sp
    #sp.call(command)
    Call(command)


# for n, frame in enumerate(FRAME_CACHE):
#     path = r"C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\Geo_DIA_Matched_Phasecorr" + f"_{n:04d}.png"
#     cv2.imwrite(path, gpu_prep(frame, Min=vmin, Max=vmax))
# 
# ffmpeg -framerate 60 -start_number 0  -i "C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\geo_tmp\Geo_DIA_Matched_Phasecorr_%04d.png" -vf "format=yuv420p" -c:v libx264 -profile:v main -level:v 5.1 -movflags +faststart -tune film -preset veryslow -crf 22 -y "C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\geo_tmp\Geo_DIA_Matched_Phasecorr.mp4"














# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  write_cv2_low_memory & shader
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

cu_code = r"""

// def prepcv2(img, LUT):
//     lo = np.min(img)
//     hi = np.max(img)
//     colorbuffer = np.empty((*img.shape, len(LUT[0])), dtype=np.uint8)
//     for ii in numba.prange(img.shape[0]):
//         for jj in numba.prange(img.shape[1]):
//             val = img[ii,jj]
//             if np.isnan(val):
//                 val = lo
//             if np.isneginf(val):
//                 val = lo
//             idx = int(255.0*((val - lo) / (hi - lo)))
//             colorbuffer[ii,jj] = (LUT[idx][::-1]*255.0).astype(np.uint8)
//     return colorbuffer

__global__
void cu_prepcv2(
    char* colorbuffer, // Image with same dimensions as img, but rgba in uint8.
    const double* img,
    const double* LUT, // Colormap as a flat array of shape (256, 3)
    const double lo, const double hi,
    int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= height * width) return;

    for(int i=0; i<3; i++) {
        // handle nans
        double val = img[idx * 3 + i];
        if (isnan(val) || isinf(val)) {
            val = lo;
        }
        int lidx = int(255.0 * ((val - lo) / (hi - lo)));
        
        // Reverse for cv2 bgr irritation
        // I wonder how many people have been bitten by this in the past. It's so easy to forget that OpenCV uses BGR order instead of RGB.
        double clr = (LUT[lidx * 3 + (2-i)] * 255.0);
        // I wonder how many gigawatts have been wasted doing this over and over again in the history of computer vision. I can't remember an era when RGB wasn't the default, so if this is burning cycles out of spite, it's a fitting reason. A terrible reason, but fitting
        colorbuffer[idx * 4 + i] = (char)clr;
    }
}

"""


# For later:
# krnl = cuda_compile(cu_code)

import cv2
def write_cv2_low_memory():

    global_min = np.inf
    global_max = -np.inf

    beta1, beta2, eps = 0.9, 0.999, 1e-8*0
    m = np.zeros((2048, 2048), dtype=np.float64)
    v = np.zeros((2048, 2048), dtype=np.float64)
    #''' # Disable chunks
    t0 = time.time()
    cv2timing = []
    for i in range(len(sector_data)):
        group_median = np.median(sector_data[max(0,i-3):][:4], axis=0)
        g = sector_data[i] - group_median
        g = slog(np.abs(g))
        m[:] = beta1 * m + (1 - beta1) * g
        if i == 0:
            m[:] = g
        t = i+1
        m_hat = m / (1 - beta1**t)
        #frame = slog(np.abs(m_hat))
        frame = m_hat
        fmin, fmax = frame.min(), frame.max()
        global_min = min(global_min, fmin)
        global_max = max(global_max, fmax)

        t1 = time.time()
        cv2timing.append(t1 - t0)
        ETA = "[ETA: Calculating...]"
        if ii >= 2:
            recent_time = np.array(cv2timing)
            recent_nums = (np.arange(nfiles)+1)[:len(recent_time)]

            # recent_nums = frame indices, recent_time = wall-clock timestamps
            eta_coeffs = np.polyfit(recent_nums, recent_time, 1)  # frame → time
            ETA_seconds = np.polyval(eta_coeffs, nfiles) - cv2timing[-1]

            # ETA seconds
            ETA = humanize.precisedelta(ETA_seconds, format="%0.3f")
            ETA = f"[ETA: {ETA}]"

        if i % 10 == 0:
            print(f"{i}/{len(sector_data)}", ETA)

    # Now accumulate histogram with fixed bin edges
    n_bins = 4096
    bin_edges = np.linspace(global_min, global_max, n_bins + 1)
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2
    global_histo = np.zeros(n_bins, dtype=np.int64)

    t0 = time.time()
    cv2timing = []
    for i in range(len(sector_data)):
        group_median = np.median(sector_data[max(0,i-3):][:4], axis=0)
        g = sector_data[i] - group_median
        g = slog(np.abs(g))
        m[:] = beta1 * m + (1 - beta1) * g
        if i == 0:
            m[:] = g
        t = i+1
        m_hat = m / (1 - beta1**t)
        #frame = slog(np.abs(m_hat))
        frame = m_hat
        h, _ = np.histogram(frame, bins=bin_edges)
        global_histo += h

        t1 = time.time()
        cv2timing.append(t1 - t0)
        ETA = "[ETA: Calculating...]"
        if ii >= 2:
            recent_time = np.array(cv2timing)
            recent_nums = (np.arange(nfiles)+1)[:len(recent_time)]

            # recent_nums = frame indices, recent_time = wall-clock timestamps
            eta_coeffs = np.polyfit(recent_nums, recent_time, 1)  # frame → time
            ETA_seconds = np.polyval(eta_coeffs, nfiles) - cv2timing[-1]

            # ETA seconds
            ETA = humanize.precisedelta(ETA_seconds, format="%0.3f")
            ETA = f"[ETA: {ETA}]"

        if i % 10 == 0:
            print(f"{i}/{len(sector_data)}", ETA)

    globals().update(locals()) # For interactive inspection


    #fig, ax = plt.subplots()
    #aximg = ax.imshow(cmap(tonemap(sector_data[0])))
    # ── Build CDF (once) ────────────────────────────────────────────
    cdf = np.cumsum(global_histo).astype(np.float64)
    cdf /= cdf[-1]
    #'''
    fourcc = cv2.VideoWriter_fourcc(*'h264')
    writer = cv2.VideoWriter('../../../sector4_the_movie_group_median_2_raw.mp4', fourcc, 60, [2048,2048])



    t0 = time.time()
    cv2timing = []
    for i in range(len(sector_data)):
        t = i+1
        group_median = np.median(sector_data[max(0,i-3):][:4], axis=0)
        g = sector_data[i] - group_median
        g = slog(np.abs(g))
        m[:] = beta1 * m + (1 - beta1) * g
        if i == 0:
            m[:] = g
        t = i+1
        m_hat = m / (1 - beta1**t)
        #frame = slog(np.abs(m_hat))
        frame = m_hat

        mapped = np.interp(frame, bins, cdf)  # consistent global tonemap
        #mapped = np.interp(denoised, bins, cdf)  # consistent global tonemap
        rgb = (cmap(mapped)*255).astype(np.uint8)
        globals().update(locals())
        #aximg.set_data(rgb)
        #fig.canvas.draw_idle()
        cv2sucks = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR)
        cv2.imshow('tess s4 c1 ccd4',cv2sucks)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        writer.write(cv2sucks)
        t1 = time.time()
        cv2timing.append(t1 - t0)

        ETA = "[ETA: Calculating...]"
        if ii >= 2:
            recent_time = np.array(cv2timing)
            recent_nums = (np.arange(nfiles)+1)[:len(recent_time)]

            # recent_nums = frame indices, recent_time = wall-clock timestamps
            eta_coeffs = np.polyfit(recent_nums, recent_time, 1)  # frame → time
            ETA_seconds = np.polyval(eta_coeffs, nfiles) - cv2timing[-1]

            # ETA seconds
            ETA = humanize.precisedelta(ETA_seconds, format="%0.3f")
            ETA = f"[ETA: {ETA}]"

        if i % 10 == 0:
            print(f"{i}/{len(sector_data)}", ETA)

    writer.release()
    globals().update(locals()) # For interactive inspection


















r'''
>>> frames = np.array(frames)
>>> frames.shape
(100, 2048, 2048)
>>> meanimg = np.zeros_like(frames[0])
>>> metadata = np.array(metadata)
>>> times, mbcks = metadata.T
>>> coeffs = np.polyfit(times, mbcks, 1)
>>> trend = np.polyval(coeffs, times)
>>> drift = trend - np.median(mbcks)
>>> median_bg = np.median(frames - drift[:, None, None], axis=0)
#>>> scary = tonemap(frames - meanimg[None,:,:] - trend[:,None,None])  # scary ram consumption. I'm at ~80% of my 64gb of ram used! (with windows)
>>> scary = frames - median_bg[None,:,:] - trend[:,None,None]
>>> scary = tonemap(scary)
>>> views = [(cmap(img)*255).astype(np.uint8) for img in scary]
>>> write_cv2(r'../../../tess_sector4_cam1_ccd1_100frames_flat_nomean_raw3.mp4', views, fps=15)
wrote ../../../tess_sector4_cam1_ccd1_100frames_flat_nomean_raw3.mp4

# I've got it **all** in ram at the moment. We've gotta be careful with how we process with cv2 for now. I dont have much ram left.
import Optical_Tools_2025 as ot
import cv2
fourcc = cv2.VideoWriter_fourcc(*'h264')
writer = cv2.VideoWriter('..\..\..\sector4_the_movie_delta_01.mp4', fourcc, 60, [2048,2048])

# Lets stream each frame as a tonemapped, cmap like the views above. 
times = np.array([header["TSTART"] for header in sector_headers])
mbcks = np.array([header["MEDBACK"] for header in sector_headers])

# ── Pass 1: Build global histogram from ALL frames ──────────────
# First need global min/max of arcsinh(residuals)
global_min = np.inf
global_max = -np.inf
for i in range(len(sector_data)):
    frame = np.arcsinh(sector_data[i] - sector_median)
    fmin, fmax = frame.min(), frame.max()
    global_min = min(global_min, fmin)
    global_max = max(global_max, fmax)
    if i % 100 == 0: print(i)

# Now accumulate histogram with fixed bin edges
n_bins = 4096
bin_edges = np.linspace(global_min, global_max, n_bins + 1)
bins = (bin_edges[:-1] + bin_edges[1:]) / 2
global_histo = np.zeros(n_bins, dtype=np.int64)

for i in range(len(sector_data)):
    frame = np.arcsinh(sector_data[i] - sector_median)
    h, _ = np.histogram(frame, bins=bin_edges)
    global_histo += h
    if i % 100 == 0: print(i)

# ── Build CDF (once) ────────────────────────────────────────────
cdf = np.cumsum(global_histo).astype(np.float64)
cdf /= cdf[-1]

t0 = time.time()
cv2timing = []
for i in range(len(sector_data)):
    frame = np.arcsinh(sector_data[i] - sector_median)
    mapped = np.interp(frame, bins, cdf)  # consistent global tonemap
    rgb = (cmap(mapped)*255).astype(np.uint8)
    writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR))
    t1 = time.time()
    cv2timing.append(t1 - t0)

    ETA = "[ETA: Calculating...]"
    if ii >= 2:
        recent_time = np.array(cv2timing)
        recent_nums = (np.arange(nfiles)+1)[:len(recent_time)]

        # recent_nums = frame indices, recent_time = wall-clock timestamps
        eta_coeffs = np.polyfit(recent_nums, recent_time, 1)  # frame → time
        ETA_seconds = np.polyval(eta_coeffs, nfiles) - t1

        ETA = humanize.precisedelta(ETA_seconds, format="%0.3f")
        ETA = f"[ETA: {ETA}]"
    
    if i % 100 == 0:
        print(f"{i}/{len(sector_data)}", ETA)

writer.release()


#LUT = np.array(cmap.colors).astype(np.float64)
## LUT is guaranteed to be 256x3, with values in [0,1]. 

write_cv2(r'../../../tess_sector4_cam1_ccd1_100frames_flat_median_detrend_raw1.mp4', views, fps=15)
compress_ffmpeg(r'../../../tess_sector4_cam1_ccd1_100frames_flat_median_detrend_raw1.mp4', r'../../../tess_sector4_cam1_ccd1_100frames_flat_median_detrend_1.mp4')



# Lets denoise this image signal via a psuedospectral NUFT approach. I'm sure the astronomers would
# love seeing LS used in this way.
# times, mbcks = metadata.T
# "Well, the interesting thought here is to try a psuedospectral transform. Only fft per pixel in time, leave space alone. That way a signal now follows the local linear trajectory of [ω, x, y]*t, rather than [ω, kx, ky]*t, which would be a wave. We sigma clip away these good signals and what's left is the noise! I suppose it is a bit wavelet-like."
# So in this instance, we use times to build a NUFT *per pixel* of the "scary" image,
#     then use a sigma mask over the power spectra in [ω, x, y] space to isolate the noisy cells. 
#     Finally, we apply that mask, and perform an iNUFT.
# We need to be careful about file sizes. 'scary' is 3.12GB on disk. This is workable in ram, but we need to
# keep an eye on what we're doing.
# I'll sketch in this file, execute in the shell, and report back.

dt = (times[-1] - times[0])/len(times)

w = np.fft.fftfreq(times.size, dt) # shape (ω,)
phasor = np.exp(np.pi*2j * times[None,:] * w[:,None]) # shape (t, ω)

# Now the part we need to be very careful of to prevent this 3gb -> 300gb.
# psuedospectral = scary @ phasor # shape (x, y, ω) = (2048, 2048, 100) -> 600gb intermediate!
psuedospectral = np.zeros_like(scary, dtype=np.complex128) # shape (ω, x, y) = (100, 2048, 2048) -> 6gb.
#for i in range(scary.shape[0]):
#    psuedospectral += scary[i] * phasor[i][:,None,None]
for y in range(scary.shape[1]):
    psuedospectral[:, y, :] = phasor @ scary[:,y,:]
    if y%128 == 0:
        print(y)

>>> plt.imshow(np.abs(psuedospectral[0])**2)
<matplotlib.image.AxesImage object at 0x000001D1F0608F80>
>>> plt.show()
>>> plt.imshow(np.abs(psuedospectral[1])**2)
<matplotlib.image.AxesImage object at 0x000001D1F074C0E0>
>>> plt.show()
>>> plt.imshow(np.abs(psuedospectral[25])**2)
<matplotlib.image.AxesImage object at 0x000001D1F0670290>
>>> plt.show()

import matplotlib.colors as mcolors
def complex_colormap(z, vmax=None):
    """Map complex array to RGBA: phase → hue, magnitude → luminance."""
    mag = np.abs(z)
    phase = np.angle(z)  # [-π, π]
    if vmax is None:
        vmax = np.percentile(mag, 99.5)
    mag = np.clip(mag / vmax, 0, 1)
    # Phase → hue (0 to 1), mag → value
    hue = (phase + np.pi) / (2 * np.pi)  # [0, 1]
    sat = np.ones_like(mag)
    hsv = np.stack([hue, sat, mag], axis=-1)
    return mcolors.hsv_to_rgb(hsv)


# mag, phase are both ~4gb. We need to be careful with memory here.
def complex_to_oklch_rgb(mag, phase, chroma=0.5):
    """Map complex array → perceptually uniform RGB via OKLab LCH.
    
    phase → hue (perceptually uniform rotation)
    |F|   → lightness (perceptually uniform brightness)
    """
    #mag = np.abs(F_slice)
    #phase = np.angle(F_slice)  # [-π, π]
    
    #if mag_vmax is None:
    #    mag_vmax = np.percentile(mag, 99.5)
    mag_vmax = 1.0

    clr = np.zeros((100, 2048, 2048, 3), dtype=np.float64) # Hardcode this due to ram restriction.
    
    # L in [0, 1], C fixed, H from phase
    # L = np.clip(mag / mag_vmax, 0, 1)
    # C = np.full_like(L, chroma)    # constant saturation
    # H = phase                       # already in radians, oklab_from_oklch expects this
    clr[..., 0] = np.clip(mag / mag_vmax, 0, 1) # L
    clr[..., 1] = chroma # C
    clr[..., 2] = phase # H
    print("built oklch array")
    
    # lab <- lch
    clr[:] = ot.oklab_from_oklch(clr)                   # (..., 3) OKLab
    print("converted to oklab")

    # xyz <- lab
    clr[:] = ot.XYZ_from_oklab(clr)                     # (..., 3) CIE XYZ
    print("converted to xyz")

    # rgb <- xyz
    clr[:] = ot.RGB_from_XYZ(clr)                       # (..., 3) linear RGB
    print("converted to linear rgb")

    # srgb <- rgb
    clr[:] = ot.sRGB_from_RGB(np.clip(clr, 0, 1))     # (..., 3) sRGB gamma
    print("converted to sRGB")
    return np.clip(clr, 0, 1)

def nightmare(complex, chroma=0.5):
    """Map complex array → perceptually uniform RGB via OKLab LCH.
    
    phase → hue (perceptually uniform rotation)
    |F|   → lightness (perceptually uniform brightness)
    """
    mag = tonemap(np.abs(complex))
    phase = np.angle(complex)  # [-π, π]
    
    #if mag_vmax is None:
    #    mag_vmax = np.percentile(mag, 99.5)
    #mag_vmax = 1.0

    clr = np.zeros((100, 2048, 2048, 3), dtype=np.float64) # Hardcode this due to ram restriction.
    
    # L in [0, 1], C fixed, H from phase
    # L = np.clip(mag / mag_vmax, 0, 1)
    # C = np.full_like(L, chroma)    # constant saturation
    # H = phase                       # already in radians, oklab_from_oklch expects this
    clr[..., 0] = mag#np.clip(mag, 0, 1) # L
    clr[..., 1] = chroma # C
    clr[..., 2] = phase # H
    del mag, phase # Free up memory
    print("built oklch array")
    
    # lab <- lch
    clr[:] = ot.oklab_from_oklch(clr)                   # (..., 3) OKLab
    print("converted to oklab")

    # xyz <- lab
    clr[:] = ot.XYZ_from_oklab(clr)                     # (..., 3) CIE XYZ
    print("converted to xyz")

    # rgb <- xyz
    clr[:] = ot.RGB_from_XYZ(clr)                       # (..., 3) linear RGB
    print("converted to linear rgb")

    # srgb <- rgb
    clr[:] = ot.sRGB_from_RGB(np.clip(clr, 0, 1))     # (..., 3) sRGB gamma
    print("converted to sRGB")
    return np.clip(clr, 0, 1)

def nightmare(cplx, chroma=0.5):
    """Map complex array → perceptually uniform RGB via OKLab LCH.
    
    phase → hue (perceptually uniform rotation)
    |F|   → lightness (perceptually uniform brightness)
    """
    mag = tonemap(np.abs(cplx))
    phase = np.angle(cplx)  # [-π, π]
    
    #if mag_vmax is None:
    #    mag_vmax = np.percentile(mag, 99.5)
    #mag_vmax = 1.0

    #clr = np.zeros((100, 2048, 2048, 3), dtype=np.float64) # Hardcode this due to ram restriction.
    clr = np.zeros((*cplx.shape, 3), dtype=np.float64)
    
    # L in [0, 1], C fixed, H from phase
    # L = np.clip(mag / mag_vmax, 0, 1)
    # C = np.full_like(L, chroma)    # constant saturation
    # H = phase                       # already in radians, oklab_from_oklch expects this
    clr[..., 0] = mag#np.clip(mag, 0, 1) # L
    clr[..., 1] = chroma # C
    clr[..., 2] = phase # H
    #del mag, phase # Free up memory
    #print("built oklch array")
    
    # lab <- lch
    clr[:] = ot.oklab_from_oklch(clr)                   # (..., 3) OKLab
    #print("converted to oklab")

    # xyz <- lab
    clr[:] = ot.XYZ_from_oklab(clr)                     # (..., 3) CIE XYZ
    #print("converted to xyz")

    # rgb <- xyz
    clr[:] = ot.RGB_from_XYZ(clr)                       # (..., 3) linear RGB
    #print("converted to linear rgb")

    # srgb <- rgb
    clr[:] = ot.sRGB_from_RGB(np.clip(clr, 0, 1))     # (..., 3) sRGB gamma
    #print("converted to sRGB")
    return np.clip(clr, 0, 1)

scary = frames - median_bg[None,:,:] - trend[:,None,None]
psuedospectral = np.zeros_like(scary, dtype=np.complex128) # shape (ω, x, y) = (100, 2048, 2048) -> 6gb.
for y in range(scary.shape[1]):
    psuedospectral[:, y, :] = phasor @ scary[:,y,:]
    if y%128 == 0:
        print(y)
clr = nightmare(psuedospectral, chroma=0.5)
#'''

import Optical_Tools_2025 as ot
def nightmare(cplx, chroma=0.5):
    """Map complex array → perceptually uniform RGB via OKLab LCH.

    phase → hue (perceptually uniform rotation)
    |F|   → lightness (perceptually uniform brightness)
    """
    mag = tonemap(np.abs(cplx))
    #print("tonemapped abs")
    phase = np.angle(cplx)  # [-π, π]
    #print("grabbed phase")

    #if mag_vmax is None:
    #    mag_vmax = np.percentile(mag, 99.5)
    #mag_vmax = 1.0

    #clr = np.zeros((100, 2048, 2048, 3), dtype=np.float64) # Hardcode this due to ram restriction.
    clr = np.zeros((*cplx.shape, 3), dtype=np.float64)

    # L in [0, 1], C fixed, H from phase
    # L = np.clip(mag / mag_vmax, 0, 1)
    # C = np.full_like(L, chroma)    # constant saturation
    # H = phase                       # already in radians, oklab_from_oklch expects this
    print('guh')
    clr[..., 0] = mag#np.clip(mag, 0, 1) # L
    clr[..., 1] = chroma # C
    clr[..., 2] = phase # H
    print("built oklch array")
    del mag, phase # Free up memory
    #print("memory yeet")

    # # lab <- lch
    # clr[:] = ot.oklab_from_oklch(clr)                   # (..., 3) OKLab
    # print("converted to oklab")

    # # xyz <- lab
    # clr[:] = ot.XYZ_from_oklab(clr)                     # (..., 3) CIE XYZ
    # print("converted to xyz")

    # # rgb <- xyz
    # clr[:] = ot.RGB_from_XYZ(clr)                       # (..., 3) linear RGB
    # print("converted to linear rgb")

    # # srgb <- rgb
    # clr[:] = np.clip(clr, 0, 1)
    # clr[:] = ot.sRGB_from_RGB(clr)     # (..., 3) sRGB gamma
    # print("converted to sRGB")
    # return np.clip(clr, 0, 1)
    nb_srgb_from_oklch(clr)
    return clr

@numba.njit(parallel=True)
def nb_srgb_from_oklch(clr):
    # Lets rewrite the color transform of clr in the above function to use a vectorized, line-by-line approach with parallelized numba loops.
    # Lets assume the prerequisite libraries are imported.
    for t in numba.prange(clr.shape[0]):
        for y in numba.prange(clr.shape[1]):
            for x in range(clr.shape[2]):
                # Extract the color vector
                lch = clr[t, y, x]  # (3,) LCH

                # lab <- lch
                #lab = ot.oklab_from_oklch(lch)                   # (3,) OKLab
                l, c, h = lch[0], lch[1], lch[2]
                a = c*np.cos(h)
                b = c*np.sin(h)
                lab = lch.copy()
                lab[1] = a
                lab[2] = b

                # xyz <- lab
                #xyz = ot.XYZ_from_oklab(lab)                     # (3,) CIE XYZ
                # Inversed using Mathematica, as fractions instead of decimals.
                iM1 = np.array([
                    [ 1.227013851,   -0.5577999807, 0.2812561490],
                    [-0.04058017842,  1.112256870, -0.07167667867],
                    [-0.07638128451, -0.4214819784, 1.586163220]
                ])

                # The hell? Was this *defined* as this inverse?
                iM2 = np.array([
                    [0.9999999985, 0.3963377922, 0.2158037581],
                    [1.000000009, -0.1055613423, -0.06385417477],
                    [1.000000055, -0.08948418209, -1.291485538]
                ])

                # numba doesn't like einsums.
                #cbrt_lms = #np.einsum('dc,...c->...d', iM2, lab)
                #xyz      = #np.einsum('dc,...c->...d', iM1, cbrt_lms**3)
                cbrt_lms = iM2 @ lab
                xyz      = iM1 @ (cbrt_lms**3)

                # rgb <- xyz
                #rgb = ot.RGB_from_XYZ(xyz)                       # (3,) linear RGB
                M = np.array([
                    [+3.2406255, -1.5372080, -0.4986286],
                    [-0.9689307, +1.8757561, +0.0415175],
                    [+0.0557101, -0.2040211, +1.0569959]
                ])

                #return np.einsum('dc,...c->...d', M, xyz)
                rgb = M @ xyz

                # srgb <- rgb
                #srgb = ot.sRGB_from_RGB(np.clip(rgb, 0, 1))     # (3,) sRGB gamma
                rgb = np.clip(rgb, 0, 1)
                srgb = np.where(
                    rgb <= 0.0031308,
                    rgb * 12.92,
                    1.055*np.power(rgb, 1/2.4) - 0.055
                )

                # Store back the result
                clr[t, y, x] = np.clip(srgb, 0, 1)





@numba.njit(parallel=True)
def mini_batch_median(sector_data, batch, ii):
    if ii >= 4: # Update the sector with the new batch
        sector_data[ii-4] -= batch[0]
    batch[:-1] = batch[1:] # Shift back the batches
    #batch[-1] = np.median(sector_data[max(0,ii-3):min(ii+4, len(sector_data))], axis=0)

    for i in numba.prange(sector_data.shape[1]):
        for j in numba.prange(sector_data.shape[2]):
            #batch[-1, i, j] = np.median(sector_data[max(0,ii-3):min(ii+4, len(sector_data)), i, j])
            # Given a vector ``V`` of length ``N``, the median of ``V`` is the
            # middle value of a sorted copy of ``V``, ``V_sorted`` - i
            # e., ``V_sorted[(N-1)/2]``, when ``N`` is odd, and the average of the
            # two middle values of ``V_sorted`` when ``N`` is even.
            # In this case, I have exactly 8 elements in the batch (ii-4:ii+3 inclusive), so the median is the average of the 4th and 5th elements in the sorted batch.
            # Since this is numba, I dont think I can do a full sort. But since its just 8 elemenents, perhaps I can get away with something O(n^2).
            strand = sector_data[max(0,ii-3):min(ii+4, len(sector_data)), i, j]
            for a in range(strand.size):
                for b in range(a+1, strand.size):
                    if strand[b] < strand[a]:
                        strand[a], strand[b] = strand[b], strand[a]
            batch[-1, i, j] = 0.5 * (strand[3] + strand[4]) # Average of the two middle values for even-length batch

def kill_ram_chunk_backup(sy, sx):
    sz = (sy, sx)

    # Load one file
    #"""
    files = get_file_list(camera=1, ccd=4) # Filter for camera 1, CCD 4
    data, header = fits.getdata(files[0], header=True)
    data = data.astype(np.float64)
    print(f"{data.shape = }, {data.dtype = }")

    # malloc 30gb lol
    sector_data = np.empty((len(files), *sz), dtype=np.float64)
    sector_headers = []

    # add this one in
    #sector_data[0] = data  # First frame loaded, rest is uninitialized
    globals().update(locals())

    # draw the rest of the fucking owl
    t0 = time.time()

    # Variance estimate compared to an EMA as a denoiser for the visualization
    β1, β2 = 0.9, 0.999
    m = np.zeros(sz, dtype=np.float64)
    v = np.zeros(sz, dtype=np.float64)
    last_m_hat = np.zeros(sz, dtype=np.complex128)

    for ii, (data, header) in enumerate(make_file_generator(files)):
        sector_data[ii] = data[:sz[0],:sz[1]] - np.median(data[:sz[0],:sz[1]], axis=0)[None,:]
        sector_headers.append(header)
        #offset_sector_data(sector_data, ii, data)
        if ii > 0:
            #diff = sector_data[ii] - sector_data[ii-1]
            curr = np.fft.fft2(sector_data[ii])

            m[:] = β1 * m + (1 - β1) * curr
    
            # 2. Update Second Moment (The Noise Estimate)
            # Using squared difference from mean is more robust for denoising (like AdaBelief)
            v[:] = β2 * v + (1 - β2) * np.abs(curr - m)**2

            # 3. Bias Correction (Critical for the first ~10-50 frames)
            m_hat = m / (1 - β1**(ii+1))
            v_hat = v / (1 - β2**(ii+1))

            # 4. Final Denoised Output
            # We use the variance to scale how much we "trust" the current pixel
            # High variance = high noise = more smoothing
            denoised_frame = m_hat / (np.sqrt(v_hat)) # Standard Adam Step
            #snr = np.abs(m_hat)**2 / (v_hat + 1e-12)# + 1e-5)
            #denoised_frame = m_hat * (snr / (snr + 1))

            # Visualization: Show the 'Adaptive Delta'
            # This is essentially the "SNR-scaled" change between frames
            delta = denoised_frame - (last_m_hat if ii > 1 else 0)
            delta = np.fft.ifft2(delta).real
            cv2_imshow(tonemap(delta), title=f'tess s4 c1 ccd4 f{ii}')
            last_m_hat[:] = denoised_frame

            #cv2_imshow(tonemap(sector_data[ii]-sector_data[ii-1]), title=f'tess s4 c1 ccd4 f{ii}')
    print('aah done')
    globals().update(locals())
    #"""

    batch = np.zeros((4, *sector_data.shape[1:]), dtype=np.float64)
    for ii in range(len(sector_data)):
        # Write to sector_data when we no longer need the original data, to save RAM. This is a rolling buffer of the last 4 frames, which we use to compute the median for the current frame.
        if ii >= 4:
            sector_data[ii-4] -= batch[0]
        batch[:-1] = batch[1:]
        batch[-1] = np.median(sector_data[max(0,ii-3):min(ii+4, len(sector_data))], axis=0)
        #mini_batch_median(sector_data, batch, ii)
        cv2_imshow(tonemap(sector_data[ii]), title=f'tess s4 c1 ccd4 f{ii}')
    print('almost')
    # Fill in the last frames left in the batch
    for i in range(4):
        sector_data[-i] -= batch[i]
    print('adjusted for medians. Time to undistort!')
    
    m.fill(0)
    v.fill(0)
    last_m_hat.fill(0)

    cuda_hcongrid_container.initialize_undistort(sector_headers[0], sector_data[0].shape)
    for ii in range(len(sector_data)):
        sector_data[ii] = cuda_hcongrid_container.undistort(sector_data[ii])
        if ii % 100 == 0:
            print(f"undistort {ii}/{len(sector_data)}")

            curr = np.fft.fft2(sector_data[ii])

            m[:] = β1 * m + (1 - β1) * curr

            # 2. Update Second Moment (The Noise Estimate)
            # Using squared difference from mean is more robust for denoising (like AdaBelief)
            v[:] = β2 * v + (1 - β2) * np.abs(curr - m)**2

            # 3. Bias Correction (Critical for the first ~10-50 frames)
            m_hat = m / (1 - β1**(ii+1))
            v_hat = v / (1 - β2**(ii+1))

            # 4. Final Denoised Output
            # We use the variance to scale how much we "trust" the current pixel
            # High variance = high noise = more smoothing
            denoised_frame = m_hat / (np.sqrt(v_hat)) # Standard Adam Step

            # Visualization: Show the 'Adaptive Delta'
            # This is essentially the "SNR-scaled" change between frames
            delta = denoised_frame - (last_m_hat if ii > 1 else 0)
            delta = np.fft.ifft2(delta).real
            cv2_imshow(tonemap(delta), title=f'tess s4 c1 ccd4 f{ii}')
            last_m_hat[:] = denoised_frame

    t1 = time.time()
    print(f"Loaded {len(files)} files in {humanize.precisedelta(t1 - t0)}.")
    globals().update(locals())

# Alright, so I've ran the following:

# times = np.array([header["TSTART"] for header in sector_headers])
# N, height, width = sector_data.shape
# dt = (times[-1] - times[0])/N
# print(f"{N = }, {height = }, {width = }") # N = 32, height = 2048, width = 2048
# print(f"{dt = }") # dt = 0.020182540915591574

# def spatial_frequency_grids(Y: int, X: int):
#     """
#     Returns angular spatial frequencies (radians / pixel).
#     """
#     ky = 2 * np.pi * np.fft.fftfreq(Y)   # (Y,)
#     kx = 2 * np.pi * np.fft.fftfreq(X)   # (X,)
#     KX, KY = np.meshgrid(kx, ky)         # both (Y, X)
#     return kx, ky, KX, KY

# kx, ky, KX, KY = spatial_frequency_grids(height, width)
# w = np.fft.fftfreq(N, dt)
# ph = np.exp(2j * np.pi * w[:,None] * times[None,:])

# sector_psuedo_fft_t_xy = np.fft.fft2(sector_data)
# sector_spectral_fft_txy = (ph @ sector_psuedo_fft_t_xy.reshape(N, -1)).reshape((N,height,width))

def solve_kepler(M, e, max_iter=50, tol=1e-14):
    """
    Solve M = E - e sin(E) for E using Newton iterations.
    M should already be wrapped to [0, 2pi).
    """
    E = M.copy()
    for _ in range(max_iter):
        f = E - e*np.sin(E) - M
        fp = 1.0 - e*np.cos(E)
        dE = f / fp
        E -= dE
        if np.max(np.abs(dE)) < tol:
            break
    return E

    
def rv2b_model(times, P, e, w, M0, K, v0):
    """
    Reproduce rv2b's RV model convention.

    times must be in the SAME time system used when rv2b fit the data.
    If rv2b fit JD, pass JD here. If rv2b fit BTJD, pass BTJD here.
    """
    times = np.asarray(times, dtype=np.float64)

    # rv2b defines t0 from M0 this way
    t0 = P * M0 / (2.0 * np.pi)

    # Wrapped mean anomaly exactly like rv2b:
    phase = (times - t0) / P
    M = 2.0 * np.pi * (phase - np.floor(phase))

    # Kepler solve
    E = solve_kepler(M, e)

    # Same true-anomaly transform as rv2b
    B = e / (1.0 + np.sqrt(1.0 - e*e))
    nu = E + 2.0 * np.arctan((B * np.sin(E)) / (1.0 - B * np.cos(E)))

    # The actual rv2b radial-velocity expression
    rv = v0 + K * (np.cos(w + nu) + e * np.cos(w))
    return rv

def lombscargle():
    N = len(times)
    w_ls = np.linspace(4,8,10000)
    dt = tstops - tstarts
    phasor = np.exp(-2j * np.pi * w_ls[:,None] * times[None,:])# * np.sinc(w_ls[:,None] * dt[None,:]) * dt[None,:]
    NUFT = phasor @ select
    Q = np.sum(phasor**2, axis=1)
    den = N * N - np.abs(Q) ** 2
    ls = 2.0 * (N * np.abs(NUFT) ** 2 - np.real(Q * np.conj(NUFT) ** 2)) / den
    fig, ax = plt.subplots()
    ax.plot(w_ls, ls)
    plt.show()
    print(w_ls[np.argmax(ls)]/(np.pi*2))

def lombscargle_resonator():
    bkg = np.array([np.median(scipy.stats.sigmaclip(img, low=1, high = 1)[0]) for img in sector_data])
    adj_data = sector_data - bkg[:,None,None]# - mk_master[None,:,:]
    adj2 = adj_data - adj_data.min()
    adj4 = adj2 - np.median(adj2, axis=0)[None,:,:]
    img = adj4 - adj4.min()

    N = len(times)
    #w_ls = np.linspace(4,8,10000)
    w0 = 5.893389338933893
    dt = tstops - tstarts
    phasor = np.exp(-2j * np.pi * w0 * times) * np.sinc(w0 * dt) * dt
    NUFT = (phasor @ img.reshape((len(adj4),-1))).reshape(img.shape[1:])
    Q = np.sum(phasor**2)
    den = N * N - np.abs(Q) ** 2
    ls_resonator = 2.0 * (N * np.abs(NUFT) ** 2 - np.real(Q * np.conj(NUFT) ** 2)) / den
    fig, ax = plt.subplots()
    ax.imshow(slog(np.abs(cv2.resize(ls_resonator, (512, 512), interpolation=cv2.INTER_CUBIC))),
        interpolation='lanczos')
    plt.show()
    return ls_resonator

if False:
    bkg = np.array([np.median(scipy.stats.sigmaclip(img, low=1, high = 1)[0]) for img in sector_data])
    adj_data = sector_data - bkg[:,None,None]# - mk_master[None,:,:]
    adj_data_min = adj_data.min()
    adj2 = adj_data - adj_data_min
    adj2_median = np.median(adj2, axis=0)
    adj4 = adj2 - adj2_median[None,:,:]
    select = adj4[:,45:52,52:58].mean(axis=(1,2))
    rv_model = rv2b_model(times + 2457000.0, 0.16969, 0.2657, 3.1211, 5.13006, 13.79602, -1.51294)

    ls_resonator = lombscargle_resonator()
    lsr_pmf = (ls_resonator - ls_resonator.min()) / (ls_resonator.max() - ls_resonator.min())
    lsr_pmf = np.maximum(lsr_pmf, 0.1)
    lsr_pmf = (lsr_pmf - lsr_pmf.min()) / (lsr_pmf.max() - lsr_pmf.min())
    lsr_pmf = lsr_pmf / lsr_pmf.sum()
    Ym = np.sum(lsr_pmf*Y)
    Xm = np.sum(lsr_pmf*X)

    pmf, A, xmean, ymean = fit_psf(adj4 - adj4.min(), rv_model, KX, KY, lsr_pmf, Xm, Ym)

def spectral_psf_loss_and_grads(
    sector_data,   # (N,Y,X), real
    Yk, # Yk = np.fft.fft2(sector_data, axes=(-2, -1))   # (N,Y,X), complex
    b,             # (N,), real or complex template
    KX, KY,        # (Y,X), radians/pixel
    p,             # (Py,Px), real-space PSF patch, real
    A,             # scalar amplitude, real
    x0, y0,        # source position, real
):
    N, Y, X = sector_data.shape
    Py, Px = p.shape

    # Spatial FFT of the data cube
    #Yk = np.fft.fft2(sector_data, axes=(-2, -1))   # (N,Y,X), complex

    # Embed PSF patch into full image grid before FFT
    p_full = np.zeros((Y, X), dtype=np.float64)
    oy = (Y - Py) // 2
    ox = (X - Px) // 2
    p_full[oy:oy+Py, ox:ox+Px] = p

    Pk = np.fft.fft2(p_full)  # (Y,X), complex

    # Shift phase
    shift = np.exp(-1j * (KX * x0 + KY * y0))   # (Y,X)

    # Model
    # b[:,None,None] broadcasts across k-space
    M = A * b[:, None, None] * Pk[None, :, :] * shift[None, :, :]

    # Residual
    r = M - Yk

    # Loss
    loss = 0.5 * np.sum(np.abs(r)**2).real

    # Common factor used in gradients wrt A, x0, y0
    base = b[:, None, None] * Pk[None, :, :] * shift[None, :, :]

    # dL/dA
    dA = np.real(np.sum(np.conj(r) * base))

    # dL/dx0, dL/dy0
    dx0 = np.real(np.sum(np.conj(r) * (A * base * (-1j * KX[None, :, :]))))
    dy0 = np.real(np.sum(np.conj(r) * (A * base * (-1j * KY[None, :, :]))))

    # dL/dp[u,v]
    dp = np.zeros_like(p, dtype=np.float64)

    # Precompute scalar-in-time matched sum over n:
    # S(k) = sum_n conj(r[n,k]) * A * b[n]
    S = np.sum(np.conj(r) * (A * b[:, None, None]), axis=0)  # (Y,X), complex

    # Gradient for each PSF pixel
    # p[u,v] enters via exp(-i K·(x0+u, y0+v))
    # for uy in range(Py):
    #     for ux in range(Px):
    #         phase_uv = np.exp(-1j * (KX * (x0 + (uy - Py//2)) + KY * (y0 + (ux - Px//2))))
    #         dp[uy, ux] = np.real(np.sum(S * phase_uv))
    # Have at thee numba
    spectral_psf_grads_nb_helper(dp, Py, Px, KX, KY, x0, y0, S)

    return loss, dA, dx0, dy0, dp
    

@numba.njit(parallel=True)
def spectral_psf_grads_nb_helper(dp, Py, Px, KX, KY, x0, y0, S):
    # Gradient for each PSF pixel
    # p[u,v] enters via exp(-i K·(x0+u, y0+v))
    # for uy in numba.prange(Py):
    #     for ux in numba.prange(Px):
    #         phase_uv = np.exp(-1j * (KX * (x0 + (uy - Py//2)) + KY * (y0 + (ux - Px//2))))
    #         dp[uy, ux] = np.real(np.sum(S * phase_uv))
    # Reindex to make this embarrasingly parallel
    for uxy in numba.prange(Py * Px):
        uy = uxy // Px
        ux = uxy % Px
        phase_uv = np.exp(-1j * (KX * (x0 + (uy - Py//2)) + KY * (y0 + (ux - Px//2))))
        dp[uy, ux] = np.real(np.sum(S * phase_uv))



def fit_psf(adj4, rv_model, KX, KY, lsr_pmf, Xm, Ym, n_iters=1000):
    params = np.zeros(lsr_pmf.size + 3)
    params[:-3] = lsr_pmf.ravel()
    params[-3:] = 1.0, Xm, Ym
    m_ema = np.zeros_like(params)
    v_ema = np.zeros_like(params)
    m_hat = np.zeros_like(params)
    v_hat = np.zeros_like(params)
    grad = np.zeros_like(params)

    adj4_fft = np.fft.fft2(adj4, axes=(-2, -1))   # (N,Y,X), complex

    for idx in range(1000):
        pmf = params[:-3].reshape(lsr_pmf.shape)
        A, xy0 = params[-3], params[-2:]
        loss, dA, dx0, dy0, dp = spectral_psf_loss_and_grads(adj4, adj4_fft, rv_model, KX, KY, pmf, A, xy0[0], xy0[1])
        if (idx+1) % 100 == 0:
            print(f"Iter {idx+1}, Loss: {loss:.4e}, A: {A:.4f}, x0: {xy0[0]:.2f}, y0: {xy0[1]:.2f}")
        grad[:-3] = dp.ravel()
        grad[-3:] = dA, dx0, dy0
        
        β1,β2,α = 0.9, 0.999, 1e-3
        m_ema[:] = β1 * m_ema + (1 - β1) * grad
        v_ema[:] = β2 * v_ema + (1 - β2) * (grad**2)
        m_hat[:] = m_ema / (1 - β1**(idx+1))
        v_hat[:] = v_ema / (1 - β2**(idx+1))
        grad[:] = m_hat / (np.sqrt(v_hat) + 1e-8)
        params -= α * grad
        #cv2_imshow_tm_smol(pmf)

        # Lets do a quick deconvolution on adj4 to 

        cv2_imshow(cv2.resize(pmf, (512, 512), interpolation=cv2.INTER_CUBIC))
    return params[:-3].reshape(lsr_pmf.shape), params[-3], params[-2:], params[-1]
    


# and I've got this fascinating spectral volume with a clear gaussian on the DC slice I'm assuming are the asteroids. Ordinarily I'd just delete them, but I'd like some help building the full Spatiotemporal spectral Lomb-Scargle analyzer. I believe in the full spectral domain the grahm matrix commutes like you said, so lets build it! I've done this before, but I'll need some help with the accumulator machinery.

def trajectory_gls_with_dc(
    sector_k,
    times,
    w_grid,
    vx,
    vy,
    KX,
    KY,
    spatial_weight=None,
):
    """
    Full pseudospectral GLS for one candidate velocity (vx, vy).

    Model per spatial mode:
        y_n(k) ≈ A(k) + B(k) exp(+i Ω(k) t_n) + C(k) exp(-i Ω(k) t_n)

    where
        Ω(k) = 2π * w - (KX*vx + KY*vy)
    with KX, KY in radians/pixel and vx, vy in pixels/time-unit.

    Inputs
    ------
    sector_k : (N, Y, X) complex128   spatial FFT movie
    times    : (N,)
    w_grid   : (W,) cycles / time-unit
    vx, vy   : velocity in pixels / time-unit
    KX, KY   : (Y, X) radians / pixel
    spatial_weight : optional (Y, X) real weighting mask

    Returns
    -------
    dict with:
      Acoef, Bcoef, Ccoef : (W, Y, X)
      power               : (W,) aggregated score over k-space
      power_map           : (W, Y, X) simple per-mode fitted energy proxy
      locA, locB, locC    : inverse FFT localization maps, (W, Y, X)
      S, Q, Ysum, F, G    : intermediates
    """
    sector_k = np.asarray(sector_k)
    times = np.asarray(times, dtype=np.float64)
    w_grid = np.asarray(w_grid, dtype=np.float64)

    N, Y, X = sector_k.shape
    W = len(w_grid)
    YX = Y * X

    if spatial_weight is None:
        spatial_weight = np.ones((Y, X), dtype=np.float64)
    else:
        spatial_weight = np.asarray(spatial_weight, dtype=np.float64)

    # ── Factored phasors ──────────────────────────────────────────
    # z[n,w,y,x] = p[n,w] · q[n,y,x]   (rank-1 in w vs yx)
    #   p[n,w]   = exp(+i 2π w_grid[w] t_n)           — (N, W), tiny
    #   q[n,y,x] = exp(-i (KX·vx + KY·vy) t_n)        — (N, Y, X)
    # This avoids materializing the full (N, W, Y, X) array.
    # Peak RAM: ~6·(W·Y·X·16) + 2·(N·Y·X·16)  instead of (N·W·Y·X·16).
    # For N=W=32, Y=X=2048: ~20 GB instead of 64+ GB.

    p = np.exp(1j * (2 * np.pi) * np.outer(times, w_grid))   # (N, W)
    spatial_shift = KX * vx + KY * vy                          # (Y, X)
    q = np.exp((-1j * times[:, None, None]) * spatial_shift[None, :, :])  # (N, Y, X)
    q_flat = q.reshape(N, YX)                                  # view
    sk_flat = sector_k.reshape(N, YX)                          # view
    print(f"factored phasors: p {p.shape}, q {q.shape}")

    # ── Time-axis accumulation via matmul ─────────────────────────
    # S[w,yx] = Σ_n p[n,w] · q[n,yx]
    S = (p.T @ q_flat).reshape(W, Y, X)

    # Q[w,yx] = Σ_n p[n,w]² · q[n,yx]²
    tmp = q_flat * q_flat                                      # (N, YX) reusable buffer
    Q = ((p * p).T @ tmp).reshape(W, Y, X)
    print("built S and Q")

    # Ysum = Σ_n y[n] — no w-dependence
    Ysum = np.sum(sector_k, axis=0)                            # (Y, X)
    print("built Ysum")

    # F[w,yx] = Σ_n conj(z) · y = Σ_n conj(p)·conj(q)·y
    np.conjugate(q_flat, out=tmp)                              # tmp = conj(q)
    tmp *= sk_flat                                             # tmp = conj(q)·y
    F = (np.conj(p).T @ tmp).reshape(W, Y, X)

    # G[w,yx] = Σ_n z · y = Σ_n p · q · y
    np.multiply(q_flat, sk_flat, out=tmp)                      # tmp = q·y
    G = (p.T @ tmp).reshape(W, Y, X)

    del tmp, q_flat, q, sk_flat, p                             # free ~4 GB
    print(f"solving {W} × ({Y}×{X}) 3×3 systems…")

    # ── Solve 3×3 Gram system per frequency slice ─────────────────
    # M per w-slice is only (Y, X, 3, 3) — 576 MB instead of 18 GB.
    Acoef = np.empty((W, Y, X), dtype=np.complex128)
    Bcoef = np.empty((W, Y, X), dtype=np.complex128)
    Ccoef = np.empty((W, Y, X), dtype=np.complex128)
    power = np.empty(W, dtype=np.float64)

    M   = np.empty((Y, X, 3, 3), dtype=np.complex128)         # reused each wi
    rhs = np.empty((Y, X, 3),    dtype=np.complex128)

    for wi in range(W):
        Sw, Qw = S[wi], Q[wi]
        Sc, Qc = np.conj(Sw), np.conj(Qw)

        # When Ω→0 (e.g. ω=0, v=0), all phasors≡1 so S=Q=N and M is rank-1.
        # The model A + B·1 + C·1 can't distinguish the three terms, but the
        # physical content is just the time-mean: A = Ysum/N, B=C=0.
        # Detect via |S/N|≈1 (phasors nearly constant across all times).
        degeneracy = np.abs(Sw / N)                            # (Y, X), ≈1 when degenerate
        degenerate = np.mean(degeneracy) > 0.999

        if degenerate:
            print(f"  w={wi}/{W} — degenerate (phasors≈const), assigning to DC")
            Acoef[wi] = Ysum / N
            Bcoef[wi] = 0
            Ccoef[wi] = 0
        else:
            M[..., 0, 0] = N;   M[..., 0, 1] = Sw;  M[..., 0, 2] = Sc
            M[..., 1, 0] = Sc;  M[..., 1, 1] = N;   M[..., 1, 2] = Qc
            M[..., 2, 0] = Sw;  M[..., 2, 1] = Qw;  M[..., 2, 2] = N

            rhs[..., 0] = Ysum
            rhs[..., 1] = F[wi]
            rhs[..., 2] = G[wi]

            coeff = np.linalg.solve(M, rhs)                    # (Y, X, 3)
            Acoef[wi] = coeff[..., 0]
            Bcoef[wi] = coeff[..., 1]
            Ccoef[wi] = coeff[..., 2]

        power[wi] = np.sum(
            (np.abs(Acoef[wi])**2 + np.abs(Bcoef[wi])**2 + np.abs(Ccoef[wi])**2)
            * spatial_weight
        )
        if wi % 8 == 0:
            print(f"  w={wi}/{W}")

    del S, Q, F, G, M, rhs

    # Simple fitted-energy proxy in spectral space
    power_map = np.abs(Acoef)**2 + np.abs(Bcoef)**2 + np.abs(Ccoef)**2

    # Localization maps in image space
    locA = np.fft.ifft2(Acoef, axes=(-2, -1))
    locB = np.fft.ifft2(Bcoef, axes=(-2, -1))
    locC = np.fft.ifft2(Ccoef, axes=(-2, -1))

    return {
        "Acoef": Acoef,
        "Bcoef": Bcoef,
        "Ccoef": Ccoef,
        "power": power,
        "power_map": power_map,
        "locA": locA,
        "locB": locB,
        "locC": locC,
    }

# sector_k = prep["sector_k"]
# w_grid   = prep["w_grid"]
# KX, KY   = prep["KX"], prep["KY"]

# # Start with a moving-DC hypothesis:
# # omega is handled by the returned power[w], so vx,vy is the candidate here.
# vx = 1.0   # pixels / day or whatever time-unit BTJD implies
# vy = 0.0

# gls = trajectory_gls_with_dc(
#     sector_k=sector_k,
#     times=times,
#     w_grid=w_grid,
#     vx=vx,
#     vy=vy,
#     KX=KX,
#     KY=KY,
# )

def kill_ram_chunk_v1(sy, sx):
    sz = (sy, sx)

    files = get_file_list(camera=1, ccd=4)
    data, header = fits.getdata(files[0], header=True)
    data = data.astype(np.float64)
    print(f"{data.shape = }, {data.dtype = }")

    sector_data = np.empty((len(files), *sz), dtype=np.float64)
    sector_headers = []
    globals().update(locals())

    t0 = time.time()

    # ── Phase 1: Load all files + column bias subtraction ──
    for ii, (data, header) in enumerate(make_file_generator(files)):
        sector_data[ii] = data[:sz[0],:sz[1]] - np.median(data[:sz[0],:sz[1]], axis=0)[None,:]
        sector_headers.append(header)
        if ii % 100 == 0:
            print(f"load {ii}/{len(files)}")
    print(f'Phase 1 done: {time.time()-t0:.1f}s')
    globals().update(locals())

    # ── Phase 2: Rolling median subtraction (fast via presorted insertion) ──
    # Instead of calling np.median on 7 frames each time (sorts 7×4M pixels),
    # compute a single global temporal median (much cheaper for DIA — we want
    # to subtract the static sky, not a local rolling window).
    t1 = time.time()
    sector_median = np.median(sector_data, axis=0)
    sector_data -= sector_median[None, :, :]
    print(f'Phase 2 done (global median): {time.time()-t1:.1f}s')

    # ── Phase 3: Undistort all frames on GPU ──
    t2 = time.time()
    cuda_hcongrid_container.initialize_undistort(sector_headers[0], sector_data[0].shape)
    for ii in range(len(sector_data)):
        sector_data[ii] = cuda_hcongrid_container.undistort(sector_data[ii])
        if ii % 100 == 0:
            print(f"undistort {ii}/{len(sector_data)}")
    print(f'Phase 3 done: {time.time()-t2:.1f}s')

    t_end = time.time()
    print(f"Total: {humanize.precisedelta(t_end - t0)}.")
    globals().update(locals())


import scipy.special
def airy_disk(r):
    res = 2*scipy.special.j1(np.pi*r)/(np.pi*r)
    return np.nan_to_num(res, nan=1.0) # Define airy_disk(0) = 1 via limit

def airy_filter(img):
    fft = np.fft.fft2(img)
    y,x = np.indices(fft.shape)
    r = np.sqrt((x-1024)**2 + (y-1024)**2)
    std = 160.6739921229835*4
    #gauss = np.exp(-r**2/(2*std**2))
    gauss = np.abs(airy_disk(r/std))**2
    fft_filtered = fft * np.fft.ifftshift(gauss)
    #plt.imshow(np.log(np.abs(np.fft.fftshift(fft_filtered))))
    return np.fft.ifft2(fft_filtered).real

# def sigclip_sorted_cumsum(data, low, high):
#     data = np.sort(data.ravel())
#     cumsum = np.cumsum(data)
#     cumsum2 = np.cumsum(data**2)
#     ilow = 0
#     ihigh = len(data)
#     while True:
#         count = ihigh - ilow
#         if count == 0:
#             break
#         mean = (cumsum[ihigh-1] - (cumsum[ilow-1] if ilow > 0 else 0)) / count
#         std = np.sqrt(((cumsum2[ihigh-1] - (cumsum2[ilow-1] if ilow > 0 else 0)) / count) - mean**2)
#         clow = mean - std*low
#         chigh = mean + std*high
#         print(f"[{ilow}:{ihigh}] mean={mean:.2f} std={std:.2f} clow={clow:.2f} chigh={chigh:.2f}")
#         ilow_prev, ihigh_prev = ilow, ihigh
#         ilow = np.searchsorted(data, clow, side='left')
#         ihigh = np.searchsorted(data, chigh, side='right')
#         # Handle case where ilow == ihigh, which culls all data
#         if ilow == ihigh:
#             ihigh = ilow+1
#             break
#         if ilow == ilow_prev and ihigh == ihigh_prev:
#             break
#     return data[ilow:ihigh], clow, chigh

# Claude Opus 4.6's rewrite
def sigclip_sorted_cumsum(data, low, high):
    data = data.ravel()
    N = len(data)
    if N == 0:
        return data, 0.0, 0.0
    print("sorting...")
    data = np.sort(data)
    print("cumsumming...")
    cumsum = np.cumsum(data)
    cumsum2 = np.cumsum(data**2)
    print("iterating...")
    ilow = 0
    ihigh = N
    num_iters_done = 0
    while True:
        count = ihigh - ilow
        if count <= 1:
            break
        # Cumsum slice sums
        s1 = cumsum[ihigh-1] - (cumsum[ilow-1] if ilow > 0 else 0)
        s2 = cumsum2[ihigh-1] - (cumsum2[ilow-1] if ilow > 0 else 0)
        mean = s1 / count
        # Clamp variance to 0 — the one-pass formula E[X²]-E[X]² cancels catastrophically
        # when values are large. max(0, ...) prevents NaN from sqrt.
        var = max(0.0, s2 / count - mean * mean)
        std = np.sqrt(var)
        clow = mean - std * low
        chigh = mean + std * high
        ilow_new = np.searchsorted(data, clow, side='left')
        ihigh_new = np.searchsorted(data, chigh, side='right')
        if ilow_new >= ihigh_new:
            break  # keep previous ilow/ihigh
        if (ilow_new == ilow) and (ihigh_new == ihigh):
            break  # converged
            # This may not work on image 7. Strange.
        ilow, ihigh = ilow_new, ihigh_new
        num_iters_done += 1
        if (num_iters_done+1)%100 == 0:
            print(f"[{ilow}:{ihigh}] mean={mean:.2f} std={std:.2f} clow={clow:.2f} chigh={chigh:.2f} iters={num_iters_done+1}")
    # Recompute final clip bounds for caller
    s1 = cumsum[ihigh-1] - (cumsum[ilow-1] if ilow > 0 else 0)
    s2 = cumsum2[ihigh-1] - (cumsum2[ilow-1] if ilow > 0 else 0)
    count = ihigh - ilow
    mean = s1 / count
    var = max(0.0, s2 / count - mean * mean)
    std = np.sqrt(var)
    print("Done!")
    return data[ilow:ihigh], mean - std * low, mean + std * high


def sigclip_histogram(data, low, high, nbins=1 << 16, returnmean=False, returnstd=False):
    """O(N) sigma-clipped median via histogram. No sort needed.

    Returns (median, clow, chigh) — scalar background estimate + clip bounds.
    Memory: ~2 MB for 65536 bins (vs 32 MiB sorted copy per 2048² frame).
    """
    flat = data.ravel()
    dmin, dmax = float(flat.min()), float(flat.max())
    if dmin == dmax:
        return dmin, dmin, dmin

    # ── Build histogram once: O(N) ──
    edges = np.linspace(dmin, dmax, nbins + 1)
    bw = edges[1] - edges[0]
    centers = edges[:-1] + 0.5 * bw                             # (nbins,)
    hist = np.histogram(flat, bins=edges)[0].astype(np.float64)  # (nbins,)

    # ── Prefix sums for O(1) range queries ──
    cs_n  = np.cumsum(hist)                                      # counts
    cs_s  = np.cumsum(hist * centers)                            # Σ x
    cs_s2 = np.cumsum(hist * centers**2)                         # Σ x²

    def _range_stats(lo, hi):
        """Mean, std from prefix sums over bin range [lo, hi)."""
        n  = cs_n[hi - 1]  - (cs_n[lo - 1]  if lo > 0 else 0)
        s  = cs_s[hi - 1]  - (cs_s[lo - 1]  if lo > 0 else 0)
        s2 = cs_s2[hi - 1] - (cs_s2[lo - 1] if lo > 0 else 0)
        if n <= 0:
            return 0.0, 0.0, 0
        mean = s / n
        std = np.sqrt(max(0.0, s2 / n - mean * mean))
        return mean, std, n

    # ── Iterative sigma clip on O(nbins) data ──
    ilo, ihi = 0, nbins
    while True:
        mean, std, n = _range_stats(ilo, ihi)
        if n <= 1:
            break
        clow  = mean - std * low
        chigh = mean + std * high
        # Map clip bounds → bin indices
        ilo_new = max(ilo, int((clow - dmin) / bw))
        ihi_new = min(ihi, int((chigh - dmin) / bw) + 1)
        if ilo_new >= ihi_new:
            break
        if ilo_new == ilo and ihi_new == ihi:
            break
        ilo, ihi = ilo_new, ihi_new

    mean, std, n = _range_stats(ilo, ihi)

    # ── Median from cumulative histogram: O(nbins) ──
    local_cum = np.cumsum(hist[ilo:ihi])
    half = local_cum[-1] * 0.5
    med_bin = np.searchsorted(local_cum, half)
    median_val = centers[ilo + min(med_bin, ihi - ilo - 1)]

    result = (median_val, mean - std * low, mean + std * high)
    if returnmean: result = result + (mean,)
    if returnstd: result = result + (std,)# else:
    return result
# Line 4146.


def process_ORIGINAL(sector_data):
    #bkg = np.array([np.median(scipy.stats.sigmaclip(img, low=1, high = 1)[0]) for img in sector_data])
    #adj_data = sector_data - bkg[:,None,None]# - mk_master[None,:,:]
    #adj_data_min = adj_data.min()
    #adj2 = adj_data - adj_data_min
    for n in range(len(sector_data)):
        #bkg = np.median(scipy.stats.sigmaclip(sector_data[n], low=1, high = 1)[0])
        #bkg = np.median(sigclip_sorted_cumsum(sector_data[n], low=1, high = 1)[0])
        bkg, clow, chigh = sigclip_sorted_cumsum(sector_data[n], low=1, high=1)
        bkg = bkg[len(bkg)//2] if len(bkg)%2 == 1 else (bkg[len(bkg)//2 - 1] + bkg[len(bkg)//2]) / 2
        sector_data[n] = sector_data[n] - bkg[None,None]
    sector_data -= sector_data.min()
    globals().update(locals())
    print("Built adj2")
    return

    """
    fft2 = np.fft.fft2(adj2[:,:16,:16])
    R = fft2[:1]*np.conj(fft2)
    R = R / np.abs(R)
    shifted = np.fft.ifft2(R).real
    YY,XX = np.indices(shifted[0].shape)
    Ymeans = np.sum(shifted*YY[None,:,:], axis=(1,2))
    Xmeans = np.sum(shifted*XX[None,:,:], axis=(1,2))
    KY,KX = np.meshgrid(np.fft.fftfreq(adj2[0].shape[0]), np.fft.fftfreq(adj2[0].shape[1]), indexing='ij')
    adj3 = np.fft.ifft2((np.fft.fft2(adj2)*np.exp(-2j*np.pi*(KX[None,:,:]*Xmeans[:,None,None] + KY[None,:,:]*Ymeans[:,None,None])))).real
    """
    #fig, ax = plt.subplots(1,2)
    #ax[0].plot(Xmeans)
    #ax[0].plot(Ymeans)
    #ax[1].plot(Xmeans, Ymeans)
    #plt.show()
    adj3 = adj2

    #KY,KX = np.meshgrid(np.fft.fftfreq(adj2[0].shape[0]), np.fft.fftfreq(adj2[0].shape[1]), indexing='ij')
    #adj3 = np.fft.ifft2((np.fft.fft2(adj2)*np.exp(-2j*np.pi*(KX[None,:,:]*Xmeans[:,None,None] + KY[None,:,:]*Ymeans[:,None,None])))).real
    print("Built adj3")

    #KYY,KXX = np.meshgrid(np.fft.fftfreq(adj3.shape[-2]),np.fft.fftfreq(adj3.shape[-1]), indexing='ij')
    #KRR = (KXX**2 + KYY**2)**.5
    adj2_blur = adj3 - adj3.min()
    #adj2_blur = np.fft.ifft2(logit(100.0*(0.2-(KRR*2)))*np.fft.fft2(adj2_blur)).real

    adj2_big = np.array([cv2.resize(slog(img), (512, 512), interpolation=cv2.INTER_LANCZOS4) for img in adj2_blur])#(adj2 - adj2.min())])
    print("lanczos'd")
    globals().update(locals())

    #fft2 = np.fft.fft2(adj2[:,:16,:16])
    #fft2 = np.fft.fft2(adj2_big[:,:16*8,:16*8])
    fft2 = np.fft.fft2(adj2_big[:,-128:,:128])
    R = fft2[:1]*np.conj(fft2)
    R = R / np.abs(R)
    shifted = np.fft.ifft2(R).real
    YY,XX = np.indices(shifted[0].shape)
    Ymeans = np.sum(shifted*YY[None,:,:], axis=(1,2))
    Xmeans = np.sum(shifted*XX[None,:,:], axis=(1,2))
    #KY,KX = np.meshgrid(np.fft.fftfreq(adj2[0].shape[0]), np.fft.fftfreq(adj2[0].shape[1]), indexing='ij')
    KY,KX = np.meshgrid(np.fft.fftfreq(adj2_big[0].shape[0]), np.fft.fftfreq(adj2_big[0].shape[1]), indexing='ij')
    #adj3 = np.fft.ifft2((np.fft.fft2(adj2)*np.exp(-2j*np.pi*(KX[None,:,:]*Xmeans[:,None,None] + KY[None,:,:]*Ymeans[:,None,None])))).real
    adj2_big = np.fft.ifft2((np.fft.fft2(adj2_big)*np.exp(-2j*np.pi*(KX[None,:,:]*Xmeans[:,None,None] + KY[None,:,:]*Ymeans[:,None,None])))).real


    #KYY,KXX = np.meshgrid(np.fft.fftfreq(adj2_big.shape[-2]),np.fft.fftfreq(adj2_big.shape[-1]), indexing='ij')
    #KRR = (KXX**2 + KYY**2)**.5
    #adj2_big = np.fft.ifft2(logit(100.0*(0.2-(KRR*2)))*np.fft.fft2(adj2_big)).real

    adj2_big_adj = np.exp(adj2_big) - np.exp(np.median(adj2_big, axis=(1,)))[:,None,:]
    print("suppressed colmeds")
    globals().update(locals())

    bckgrnd = np.exp(np.median(adj2_big, axis=(0,)))
    adj2_big_adj -= -bckgrnd[None,:,:]
    print("log median adjusted")
    globals().update(locals())

    #adj2_big_adj2 = adj2_big_adj - adj2_big_adj.min()
    #adj2_big_adj2 = slog(adj2_big_adj - adj2_big_adj.min())
    globals().update(locals())
    #anim(adj2_big_adj2)
    return

    sclip, cmin, cmax = sigclip_sorted_cumsum(adj2_big_adj, 5.0,5.0)
    del sclip
    print("sigmaclip done")
    globals().update(locals())

    goodresult = np.arcsinh((adj2_big_adj-np.median(adj2_big_adj, axis=1)[:,None,:])/cmax)
    print("arcsinh adj done")
    globals().update(locals())

    #anim(adj2_big_adj)
    #return

    # per-time medians removed. This should be the true DIA step!
    #adj2_big_adj -= adj2_big_adj.mean()

    # Now the true mean is removed. Nufts behave interesting. Lets take the log of each frequency.
    #arr = adj2_big_adj
    arr = goodresult
    N = len(times)
    Nw = 1200
    dt = tstops - tstarts
    w = np.fft.fftfreq(Nw, 25/Nw)
    phasor = np.exp(-2j * np.pi * w[:,None] * times[None,:]) * np.sinc(w[:,None] * dt[None,:]) * dt[None,:]
    print("phasor built")
    globals().update(locals())

    NUFT = (phasor @ arr.reshape((len(arr),-1))).reshape((Nw, *arr.shape[1:]))
    print("NUFT built")
    globals().update(locals())

    mag = np.abs(NUFT[1:])
    anim(slog(mag))

    """
    mag = np.abs(NUFT[1:])
    maxmags = mag.max(axis=0)
    mag[:] = maxmags[None,:,:]/mag # normalized so the largest mag is 1 per pixel
    NUFT[1:] *= mag

    #"" "
    mag = np.abs(NUFT[1:])
    #mag[:] = tonemap(mag)/mag
    maxmags = mag.max(axis=0)
    mag[:] = maxmags[None,:,:]/mag # normalized so the largest mag is 1 per pixel
    tm = tonemap(maxmags)
    mag[:] *= tm[None,:,:]
    NUFT[1:] *= mag
    NUFT[0] *= 0.0
    #NUFT[1:] /= mag.max(axis=0)[None,:,:]

    #mag = np.abs(NUFT[0])
    #for ii in range(1, len(NUFT)):
    #    mag[:] = np.abs(NUFT[ii])
    #    mag[:] = slog(mag)/mag
    #    NUFT[ii] *= mag
    #    NUFT[ii] = np.nan_to_num(NUFT[ii], nan=0, posinf=0, neginf=0)
    print("NUFT tonemapped")
    globals().update(locals())
    """

    dt2 = 25/Nw
    tstarts2 = tstarts[0] + dt2*np.arange(Nw)
    tstops2 = tstarts[0] + dt2*(np.arange(Nw)+1)
    times2 = (tstarts2+tstops2)*0.5
    globals().update(locals())

    phasor = np.exp(2j * np.pi * w[None,:] * times2[:,None])# * np.sinc(w[None,:] * dt2) * dt2
    globals().update(locals())
    result = (phasor @ NUFT.reshape((Nw,-1))).reshape((Nw, *adj2_big_adj.shape[1:])).real
    print("NUFT reversed (new frames)")
    globals().update(locals())

    animate_cv2(result, Min=result.min(), Max=result.max())
    #res = np.abs(result)
    #animate_cv2(res, Min=res.min(), Max=res.max())

def process(sector_data):
    vminimax = np.array([1e300, 1e-300], dtype=np.float64)
    for n in range(len(sector_data)):
        bkg, clow, chigh = sigclip_histogram(sector_data[n], low=1, high=1)
        sector_data[n] -= bkg

        vminimax[0] = min(vminimax[0], sector_data[n].min()) # global min across all frames
        # slog the difference to get a stable log scaling
        sector_frame_slog = slog(sector_data[n] - vminimax[0])
        # Find the global slog max across all frames to ensure consistent scaling
        vminimax[1] = max(vminimax[1], sector_frame_slog.max()) # global max across all frames
        sminimax = slog(vminimax[:1])[0]
         # gkivak suggested tracking the min/max in linear space and only slogging for display, which is more robust to outliers dominating the log scaling
        cv2_imshow(sector_frame_slog, Min=sminimax, Max=vminimax[1],
            title=f'tess s{sector} c{camera} ccd{ccd} f{n}')
        #print(f"bkg {n}/{len(sector_data)}: {bkg:.2f}", end=(' | ' if (n+1)%10 != 0 else '\n'))
        #if n % 100 == 0: print(f"  bkg {n}/{len(sector_data)}: {bkg:.2f}")
    sector_data -= sector_data.min()
    globals().update(locals())
    print("Built adj2")
    return

def kill_ram_chunk(sy=2048, sx=2048, lim=-1):
    sz = (sy, sx)

    # Load one file
    #"""
    camera_ = int(camera)
    ccd_ = int(ccd)
    #files = get_file_list(camera=1, ccd=4)[:lim] # Filter for camera 1, CCD 4
    files = get_file_list(camera=camera_, ccd=ccd_)[:lim] # Filter for specified camera and CCD
    data, header = fits.getdata(files[0], header=True)
    data = data.astype(np.float64)
    print(f"{data.shape = }, {data.dtype = }")

    # malloc 30gb lol
    sector_data = np.empty((len(files), *sz), dtype=np.float64)
    sector_headers = []

    # add this one in
    #sector_data[0] = data  # First frame loaded, rest is uninitialized
    globals().update(locals())

    # draw the rest of the fucking owl
    t0 = time.time()

    # Variance estimate compared to an EMA as a denoiser for the visualization
    β1, β2 = 0.9, 0.999
    m = np.zeros(sz, dtype=np.float64)[:64,:64] # TEMP CROP TO 64X64
    v = np.zeros(sz, dtype=np.float64)[:64,:64]
    last_m_hat = np.zeros(sz, dtype=np.complex128)
    px, py = 1426, 1616

    for ii, (data, header) in enumerate(make_file_generator(files)):
        crop = data[:sz[0],:sz[1]]
        sector_data[ii] = crop# - np.median(crop, axis=0)[None,:]  # This median removes the column bias. lets ignore that for now.
        sector_headers.append(header)
        #offset_sector_data(sector_data, ii, data)
        if ii > 0:
            #cv2_imshow(tonemap(sector_data[ii]-sector_data[ii-1], bins=512), title=f'tess s4 c1 ccd4 f{ii}')
            curr = sector_data[ii][py-32:py+32, px-32:px+32] - sector_data[ii-1][py-32:py+32, px-32:px+32]
            curr = tonemap(curr)
            m[:] = β1 * m + (1 - β1) * curr
            m_hat = m / (1 - β1**(ii+1))
            #cv2_imshow(m_hat, title=f'tess s{sector} c{camera} ccd{ccd} f{ii}')
            # I can't see anything. Lets scale it up from 64x64 to 512x512 with a nice interpolation and see if we can spot the asteroid streaks in the noise.
            m_scaled = cv2.resize(m_hat, (512, 512), interpolation=cv2.INTER_CUBIC)
            #m_scaled = airy_filter(m_scaled)
            cv2_imshow(m_scaled, title=f'tess s{sector} c{camera} ccd{ccd} f{ii}')

        #if ii > 0 and ii%100 == 0:
            """
            #diff = sector_data[ii] - sector_data[ii-1]
            curr = np.fft.fft2(sector_data[ii])

            m[:] = β1 * m + (1 - β1) * curr

            # 2. Update Second Moment (The Noise Estimate)
            # Using squared difference from mean is more robust for denoising (like AdaBelief)
            v[:] = β2 * v + (1 - β2) * np.abs(curr - m)**2

            # 3. Bias Correction (Critical for the first ~10-50 frames)
            m_hat = m / (1 - β1**(ii+1))
            v_hat = v / (1 - β2**(ii+1))

            # 4. Final Denoised Output
            # We use the variance to scale how much we "trust" the current pixel
            # High variance = high noise = more smoothing
            denoised_frame = m_hat / (np.sqrt(v_hat)) # Standard Adam Step
            #snr = np.abs(m_hat)**2 / (v_hat + 1e-12)# + 1e-5)
            #denoised_frame = m_hat * (snr / (snr + 1))

            # Visualization: Show the 'Adaptive Delta'
            # This is essentially the "SNR-scaled" change between frames
            delta = denoised_frame - (last_m_hat if ii > 1 else 0)
            delta = np.fft.ifft2(delta).real
            cv2_imshow(tonemap(delta), title=f'tess s4 c1 ccd4 f{ii}')
            last_m_hat[:] = denoised_frame
            """
            #cv2_imshow(tonemap(sector_data[ii]-sector_data[ii-1]), title=f'tess s4 c1 ccd4 f{ii}')
    print('aah done')
    globals().update(locals())

    # Construct the global median background frame
    # sector_median = np.empty_like(sector_data[0])
    # for yy in range(sector_data.shape[1]):
    #     sector_median[yy] = np.median(sector_data[:, yy, :], axis=0)
    #     if (yy+1)%128 == 0:
    #         print(yy+1)
    # 
    # row_medians = np.empty_like(sector_data[0,0,:])
    # # In-place subtract sector medians along the [t,y] plane to reduce *most* ccd errors.
    # for xx in range(sector_data.shape[2]):
    #     #sector_data[:, :, xx] -= np.median(sector_data[:, :, xx], axis=0)
    #     row_medians[xx] = np.median(sector_data[:, :, xx])
    #     if (xx+1)%128 == 0:
    #         print(xx+1)
    
    #bkg = np.array([np.median(scipy.stats.sigmaclip(img, low=1, high = 1)[0]) for img in sector_data])
    #mk_master = np.median(sector_data - bkg[:,None,None], axis=0)
    #sector_data - bkg[:,None,None] - mk_master[None,:,:]


    # For later
    cuda_hcongrid_container.initialize_undistort(sector_headers[0], sector_data[0].shape)#, chunk_offset=(1024,1024))

    """
    batch = np.zeros((4, *sector_data.shape[1:]), dtype=np.float64)
    for ii in range(len(sector_data)):
        # Write to sector_data when we no longer need the original data, to save RAM. This is a rolling buffer of the last 4 frames, which we use to compute the median for the current frame.
        if ii >= 4:
            sector_data[ii-4] -= batch[0]
        batch[:-1] = batch[1:]
        batch[-1] = np.median(sector_data[max(0,ii-3):min(ii+4, len(sector_data))], axis=0)
        #mini_batch_median(sector_data, batch, ii)
        if ii > 0 and ii%100 == 0:
            cv2_imshow(tonemap(sector_data[ii]), title=f'tess s4 c1 ccd4 f{ii}')
    print('almost')
    # Fill in the last frames left in the batch
    for i in range(4):
        sector_data[-i] -= batch[i]
    print('adjusted for medians. Time to undistort!')

    m.fill(0)
    v.fill(0)
    last_m_hat.fill(0)

    t1 = time.time()
    """
    
    """
    for ii in range(len(sector_data)):
        sector_data[ii] = cuda_hcongrid_container.undistort(sector_data[ii])
        if ii % 100 == 0:
            print(f"undistort {ii}/{len(sector_data)}")

            curr = np.fft.fft2(sector_data[ii])

            m[:] = β1 * m + (1 - β1) * curr

            # 2. Update Second Moment (The Noise Estimate)
            # Using squared difference from mean is more robust for denoising (like AdaBelief)
            v[:] = β2 * v + (1 - β2) * np.abs(curr - m)**2

            # 3. Bias Correction (Critical for the first ~10-50 frames)
            m_hat = m / (1 - β1**(ii+1))
            v_hat = v / (1 - β2**(ii+1))

            # 4. Final Denoised Output
            # We use the variance to scale how much we "trust" the current pixel
            # High variance = high noise = more smoothing
            denoised_frame = m_hat / (np.sqrt(v_hat)) # Standard Adam Step

            # Visualization: Show the 'Adaptive Delta'
            # This is essentially the "SNR-scaled" change between frames
            delta = denoised_frame - (last_m_hat if ii > 1 else 0)
            delta = np.fft.ifft2(delta).real
            cv2_imshow(tonemap(delta), title=f'tess s4 c1 ccd4 f{ii}')
            last_m_hat[:] = denoised_frame
    """

    t1 = time.time()
    print(f"Loaded {len(files)} files in {humanize.precisedelta(t1 - t0)}.")
    globals().update(locals())

def cv2_imshow_tm_smol(img):
    cv2_imshow(tonemap(cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)))











if False:
#if True:
    tstarts = np.array([header["TSTART"] for header in sector_headers])
    tstops = np.array([header["TSTOP"] for header in sector_headers])
    times = 0.5 * (tstarts + tstops)
    N, height, width = sector_data.shape
    Nw = N*4 # 32 -> 128. Be sure you dont kill the ram!!!
    dt = (tstarts[-1] - tstarts[0])/N
    print(f"{N = }, {height = }, {width = }") # N = 32, height = 2048, width = 2048
    print(f"{Nw = }, {dt = }") # dt = 0.020182540915591574

    def spatial_frequency_grids(Y: int, X: int):
        """
        Returns angular spatial frequencies (radians / pixel).
        """
        ky = 2 * np.pi * np.fft.fftfreq(Y)   # (Y,)
        kx = 2 * np.pi * np.fft.fftfreq(X)   # (X,)
        KX, KY = np.meshgrid(kx, ky)         # both (Y, X)
        return kx, ky, KX, KY

    kx, ky, KX, KY = spatial_frequency_grids(height, width)
    w = np.fft.fftfreq(Nw, dt)
    ph = np.exp(2j * np.pi * w[:,None] * times[None,:])

    # Setup a NUFT but using a discrete timescale for sinc.
    # Proof (will delete when i'm confident)
    # ∫ 1*exp(-2πi wt) dt ∈[t₀, t₁]
    # u = -2πi wt -> du = -2πi w dt -> dt = -du/(2πi w)
    # -∫ 1*exp(u) du/(2πi w) ∈[t₀, t₁]
    # = -1/(2πi w) * ∫ exp(u) du ∈ t ∈[t₀, t₁]
    # = -1/(2πi w) * (exp(-2πi wt) t ∈[t₀, t₁]
    # = -1/(2πi w) * (exp(-2πi wt₁) - exp(-2πi wt₀))
    # Let t = (t₁+t₀)/2, and Δth = (t₁-t₀)/2, where Δth is the half-bin width, rather than Δt is the full bin width.
    # t₁ = t + Δth, t₀ = t - Δth
    # = -1/(2πi w) * (exp(-2πi w(t + Δth)) - exp(-2πi w(t - Δth)))
    # = -1/(2πi w) * exp(-2πi wt) * (exp(-2πi wΔth) - exp(2πi wΔth))
    # = exp(-2πi wt) * (exp(2πi wΔth) - exp(-2πi wΔth)) / (2πi w)
    # = exp(-2πi wt) * sin(2π w Δth) / (π w)
    # In numpy, sinc is sin(πx)/(πx)
    # = exp(-2πi wt) * (sin(π∙(2 w Δth)) / (π∙(2 w Δth)))∙2Δth
    # = exp(-2πi wt) * sinc(2 w Δth) ∙ 2Δth, Lets swap 2Δth for Δt = t₁-t₀.
    # = exp(-2πi wt) * sinc(w (t₁-t₀)) ∙ (t₁-t₀)

    #dt = (tstarts[-1] - tstarts[0])/N
    sector_psuedo_fft_t_xy = np.fft.fft2(sector_data)
    print("Finished psuedo")

    dt = tstops - tstarts
    phasor_wt = np.exp(-2j * np.pi * w[:,None] * times[None,:]) * np.sinc(w[:,None] * dt[None,:]) * dt[None,:]

    print("Starting NUFT matmul")
    sector_spectral_fft_txy = (phasor_wt @ sector_psuedo_fft_t_xy.reshape((N, -1))).reshape((Nw, height, width))
    print("Finished NUFT matmul")

    # sector_spectral_fft_txy = np.zeros((Nw, height, width), dtype=np.complex128)
    # for idxt,(ts, te, tm) in enumerate(zip(tstarts, tstops, times)):
    #     dt = te - ts
    #     ph = np.exp(-2j * np.pi * w * tm) * np.sinc(w * dt) * dt
    #     sector_spectral_fft_txy += ph[:,None,None] * sector_psuedo_fft_t_xy[idxt,:,:]
    #     print(f"NUFT {idxt+1}/{len(times)}")

    # Construct a causal + retrocausal EMA filter:
    # fft(exp(-αt)) = 1/(α + 2πi ω) -> causal + retrocausal ~ 1/(α² + (2πω)²). Lets define a "speed" so we can relate kx, ky to our temporal "distance":
    # ~ 1/((kx² + ky²)α² + (2πω)²)
    # Would a truly lorentzian metric on the denominator work better here? That is:
    # ~ 1/((kx² + ky²)α² - (2πω)²)?
    # Copilot: The choice between a Gaussian-like filter (1/((kx² + ky²)α² + (2πω)²)) and a Lorentzian-like filter (1/((kx² + ky²)α² - (2πω)²)) depends on the specific characteristics of the data and the desired outcome.

    # Combines causal + retrocausal filter + epsilon tolerance. I may need to tune the implied velocity parameter.
    c2 = 1.0
    ds2 = ((kx[None,None,:]**2 + ky[None,:,None]**2)*c2 - w[:,None,None]**2)**2#)**.5# + 1e-5**2)
    # Gate ds2 by ((np.tanh(x)+1)/2) to clip spacelike frequencies. Oh shit!!!
    # Lets see if we got the sign right.
    # Ω = k.v - ω
    # For a max speed of c,
    # Ω = (kc)^2 - ω^2
    # At no speed:
    # Ω = -ω^2, so we need to use -ds2.

    ds2 = (np.tanh(-ds2)+1.0)/2.0

    # Lets extract *one* frame of a resampled time.
    ts0, te0 = tstarts[0], tstarts[0] + dt[0]/4
    t0 = 0.5*(ts0 + te0)
    dt0 = te0 - ts0
    ph0 = np.exp(-2j * np.pi * w * t0) * np.sinc(w * dt0) * dt0

    #frame0_fft_t_xy = (ph0[:,None] * sector_spectral_fft_txy.reshape((Nw, -1))).sum(axis=0).reshape((height, width))
    frame0_fft_t_xy = (ph0[:,None] * (ds2 * sector_spectral_fft_txy).reshape((Nw, -1))).sum(axis=0).reshape((height, width))
    print("Extracted one NUFT frame")
    plt.imshow(tonemap(np.fft.ifft2(frame0_fft_t_xy).real))
    plt.show()

def airy_filter(img):
    fft = np.fft.fft2(img)
    y,x = np.indices(fft.shape)
    r = np.sqrt((x-1024)**2 + (y-1024)**2)
    std = 160.6739921229835*4
    #gauss = np.exp(-r**2/(2*std**2))
    gauss = np.abs(airy_disk(r/std))**2
    fft_filtered = fft * np.fft.ifftshift(gauss)
    #plt.imshow(np.log(np.abs(np.fft.fftshift(fft_filtered))))
    return np.fft.ifft2(fft_filtered).real

if False:
    # Lets extract *one* frame of a resampled time.
    #c2 = (1/1000.0)**2
    #c2 = 1.0
    c2 = 1000.0**2
    ds2 = ((kx[None,None,:]**2 + ky[None,:,None]**2) - c2*w[:,None,None]**2)#**2#)**.5# + 1e-5**2)
    ds2 = ds2*((np.tanh(-100.0*ds2)+1.0)/2.0) # Gate ds2 by ((np.tanh(x)+1)/2) to clip spacelike frequencies. Oh shit!!!
    #ds2 = 1/(np.abs(ds2)**.5 + 1e-12)

    # Lets also include a deconvolution!
    kr = (kx[None,:]**2 + ky[:,None]**2)**.5
    # std = 2048/np.pi#2048/(4*np.pi) * 4 = 160.6739921229835*4
    # airy = np.abs(airy_disk(kr/std))**2
    # ds2 = ds2 * airy
    # 
    # ds2 = 1/(np.abs(ds2)**.5 + 1e-12)
    std = 2048/np.pi
    airy = airy_disk(kr/(8*std)) - airy_disk(kr/(2*std))
    ds2 = np.nan_to_num(1/airy)

    idx = 16
    ts0, te0 = tstarts[idx], tstarts[idx] + dt[idx]#/4
    t0 = 0.5*(ts0 + te0)
    dt0 = te0 - ts0
    ph0 = np.exp(-2j * np.pi * w * t0) * np.sinc(w * dt0) * dt0

    #frame0_fft_t_xy = (ph0[:,None] * sector_spectral_fft_txy.reshape((Nw, -1))).sum(axis=0).reshape((height, width))
    frame0_fft_t_xy = (ph0[:,None] * (ds2 * sector_spectral_fft_txy).reshape((Nw, -1))).sum(axis=0).reshape((height, width))
    print("Extracted one NUFT frame")
    plt.imshow(tonemap(np.fft.ifft2(frame0_fft_t_xy).real))
    plt.show()
    #cv2_imshow(tonemap(np.fft.ifft2(frame0_fft_t_xy).real))

# 2048/160.6739921229835
# 12.746306810080469
# 12.746306810080469/np.pi
# 4.05727546998039
# 4.0*np.pi = (2048/(160.6739921229835))
# 160.67... = 2048/(4.0*np.pi)
# (2048/(160.6739921229835*np.pi*4))
# 1.0143188674950976
# 1.0 = 2048/(160.6739921229835*np.pi*4)


if False:
    # Lets extract *one* frame of a resampled time.
    #c2 = (1/1000.0)**2
    #c2 = 1.0
    c2 = 1000.0**2
    ds2 = ((kx[None,None,:]**2 + ky[None,:,None]**2) - c2*w[:,None,None]**2)#**2#)**.5# + 1e-5**2)
    gate = ((np.tanh(-1.0*ds2)+1.0)/2.0) # Gate ds2 by ((np.tanh(x)+1)/2) to clip spacelike frequencies. Oh shit!!!
    #ds2 = 1/(np.abs(ds2)**.5 + 1e-12)

    # Lets also include a deconvolution!
    #kr = (kx[None,:]**2 + ky[:,None]**2)**.5
    #std = 160.6739921229835*4
    #airy = np.abs(airy_disk(kr/std))**2
    #airy = airy_disk(kr/std)
    #ds2 = ds2 * airy

    #ds2 = 1/(np.abs(ds2)**.5*airy + 1e-12)
    #ds2 = airy/(np.abs(ds2)**.5 + 1e-12)
    #ds2 = 1/(np.abs(ds2)**.5*gate + (1.0-gate)*airy**2 + 1e-12)
    ds2 = 1/(ds2*gate + (1.0-gate)*airy)
    ds2 = np.nan_to_num(ds2)


    seq = []
    idx = 0
    framenum_lim = len(sector_spectral_fft_txy)
    for framenum in range(framenum_lim):
        ts0, te0 = tstarts[idx], tstarts[idx] + dt[idx]/4
        t0 = 0.5*(ts0 + te0)
        dt0 = te0 - ts0

        t0 = t0 + dt0*framenum
        ts0 = ts0 + dt0*framenum
        te0 = te0 + dt0*framenum
        t0 = 0.5*(ts0 + te0)

        ph0 = np.exp(-2j * np.pi * w * t0) * np.sinc(w * dt0) * dt0

        #frame0_fft_t_xy = (ph0[:,None] * sector_spectral_fft_txy.reshape((Nw, -1))).sum(axis=0).reshape((height, width))
        frame0_fft_t_xy = (ph0[:,None] * (ds2 * sector_spectral_fft_txy).reshape((Nw, -1))).sum(axis=0).reshape((height, width))
        print("Extracted one NUFT frame")
        #plt.imshow(tonemap(np.fft.ifft2(frame0_fft_t_xy).real))
        #plt.show()
        #cv2_imshow(tonemap(np.fft.ifft2(frame0_fft_t_xy).real))
        img = np.fft.ifft2(frame0_fft_t_xy).real
        seq.append(img)
        cv2_imshow(tonemap(img), title=f'tess s4 c1 ccd4 f{idx}')

@numba.njit(parallel=True)
def prepcv2(img, LUT):
    lo = np.min(img)
    hi = np.max(img)
    colorbuffer = np.empty((*img.shape, len(LUT[0])), dtype=np.uint8)
    for ii in numba.prange(img.shape[0]):
        for jj in numba.prange(img.shape[1]):
            val = img[ii,jj]
            if np.isnan(val):
                val = lo
            if np.isneginf(val):
                val = lo
            idx = int(255.0*((val - lo) / (hi - lo)))
            colorbuffer[ii,jj] = (LUT[idx][::-1]*255.0).astype(np.uint8)
    return colorbuffer

import cv2
LUT = np.array(cmap.colors).astype(np.float64)
gpu_prep = CUDAPrepcv2(LUT)
#gpu_prep.imshow(tonemap(sector_data[0]), title='tess s4 c1 ccd4')
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(2) # google ai mode reccomended this to fix cv2 being irritating
def cv2_imshow(img, title='tess', prep=True, fps=-1, Min=None, Max=None):
    if prep:
        #img = prepcv2(img, LUT)
        img = gpu_prep(img, Min=Min, Max=Max)
    cv2.imshow('tess', img)
    cv2.setWindowTitle('tess', title)
    delay_ms = int(1000/fps) if fps > 0 else 1
    cv2.waitKey(delay_ms)

@numba.njit(parallel=True)
def compute_psuedospectral(sector_data_ftfxfy, sector_data_t_fxfy, times, w):
    for ii in numba.prange(sector_data_ftfxfy.shape[0]):
        sector_data_ftfxfy += sector_data_t_fxfy[ii]*np.exp(np.pi*2j*times[ii]*w[:,None,None])


# >>> tm_data[0].shape
# (2048, 2048)
# kidx = np.fft.fftfreq(2048)
# ky, kx = np.meshgrid(kidx, kidx, indexing='ij')
# kr = (kx**2 + ky**2)**.5
# >>> kr.shape
# (2048, 2048)
# σ = 1.0 # 1 pixel blur. Tunable!
# mask = np.exp(-0.5*(kr*σ)**2)
# if True:
#     EMA = True
#     ema_alpha = 0.1
#     framenum = 0
#     ema = cuda_hcongrid_container.undistort(tm_data[0], scale=1/1.1).copy()
#     while cv2.waitKey(1) & 0xFF != ord("q"):
#         framenum = framenum % len(tm_data) # Handles negatives gracefully
#         if EMA:
#             # Do a TAA-style ema blur
#             ema[:] = np.fft.ifft2(np.fft.fft2(ema)*mask).real
#             ema[:] = ema*(1-ema_alpha) + ema_alpha*cuda_hcongrid_container.undistort(tm_data[framenum], scale=1/1.1)
#         else:
#             ema[:] = cuda_hcongrid_container.undistort(tm_data[framenum], scale=1/1.1)

#         cv2_imshow(ema, title=f'tess s4 c1 ccd4 f{framenum}')
#         framenum += 1


# plt.plot(times, select)
                     
# [<matplotlib.lines.Line2D object at 0x000001FA39504C80>]
# plt.show()
                     
# times.shape
# import csv
# # Write this as a csv with 3 columns: time, select, and one with all ones. That should be errors, which I dont have.
# with open('select_times.csv', 'w', newline='\n') as csvfile:
#     writer = csv.writer(csvfile)
#     for t, s in zip(times, select):
#         writer.writerow([t, s, 1.0]) # Placeholder error value of 1.0
# need btjd -> julian
# times = times + 2457000.0

#raise Exception("This is a warning that the code below is for testing and may consume a lot of RAM. Proceed with caution.")

def animate_cv2(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    framenum = 0
    ema = seq[0].copy()
    if cplx:
        fltr = nightmare
    if fltr is None:
        fltr = lambda x: x
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        if EMA:
            ema[:] = ema*(1-ema_alpha) + ema_alpha*seq[framenum]
        else:
            ema[:] = seq[framenum]

        cv2_imshow(fltr(ema), title=f'tess s4 c1 ccd4 f{framenum}', prep=prep, fps=fps, Min=Min, Max=Max)
        framenum += 1

def animate_cv2_diff(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0
    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum])
        diff = raw-ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 4: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale

        cv2_imshow(diff, title=f'tess s4 c1 ccd4 f{framenum}', prep=prep, fps=fps, Min=vmin, Max=vmax)
        ema[:] = raw
        framenum += 1
        total_framenum += 1

def phase_corr_iterator(img0, img1):
    fft0 = np.fft.fft2(img0)
    fft1 = np.fft.fft2(img1)
    z = fft0*fft1.conj()
    zph = np.nan_to_num(z / np.abs(z), nan=0.0, posinf=0.0, neginf=0.0)
    ky,kx = np.meshgrid(np.fft.fftfreq(img0.shape[0]), np.fft.fftfreq(img0.shape[1]), indexing='ij')
    tau = np.pi*2
    itau = tau*1j

    # This SHOULD be a weight. Unfortunately, this test didn't work out as well as just using the argmax. Alas!
    # ifft = np.fft.ifft2(zph).real 
    # yy,xx = np.indices(ifft.shape)
    # yy = np.fft.fftshift(yy) - (ifft.shape[0]//2)
    # xx = np.fft.fftshift(xx) - (ifft.shape[1]//2)
    # dyx0 = np.array([(ifft**2*yy).sum(), (ifft**2*xx).sum()]) / (ifft**2).sum()
    # print(f"{dyx0 = }")

    dyx = np.array(np.unravel_index(np.argmax(np.fft.ifft2(zph).real), img0.shape), np.float64)
    #dyx = np.array(np.unravel_index(np.argmax(np.fft.ifft2(zph_pad).real), zph_pad.shape), np.float64)
    #del zph_pad
    #dyx /= target_multiplier # Correct for target multiplier padding

    #print(f"{dyx = }")
    for n in range(10):
        phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1])) * zph
        Jx = (-itau*kx*phasor).sum().real
        Jy = (-itau*ky*phasor).sum().real
        Hxx = (-tau**2*kx*kx*phasor).sum().real
        Hxy = (-tau**2*kx*ky*phasor).sum().real
        Hyy = (-tau**2*ky*ky*phasor).sum().real
        Hdet = Hxx*Hyy - Hxy*Hxy
        Hinv = np.array([[Hxx,-Hxy],[-Hxy,Hyy]])/Hdet
        step = Hinv @ np.array([Jy, Jx])
        dyx -= step
        #print(f"  iter {n}: dyx={dyx}, |step|={np.linalg.norm(step):.12e}")
        #dyx -= H_inv @ np.array([Jy, Jx])
        #print(f"{dyx = }")
        #cv2_imshow(slog(np.abs(np.fft.ifft2(fft1*phasor).real-img0)))
    return dyx

def phase_corr_iterator_crop(idx2):

    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))

    idx0 = (idx2-1)%len(sector_data)
    #idx0 = 0

    img0 = slog(sector_data[idx0])
    img1 = slog(sector_data[idx2])

    fft0 = np.fft.fft2(img0[idx_crop])
    fft1 = np.fft.fft2(img1[idx_crop])
    z = fft0*fft1.conj()
    zph = np.nan_to_num(z / np.abs(z), nan=0.0, posinf=0.0, neginf=0.0)
    ky,kx = np.meshgrid(np.fft.fftfreq(fft0.shape[0]), np.fft.fftfreq(fft0.shape[1]), indexing='ij')
    tau = np.pi*2
    itau = tau*1j

    # This SHOULD be a weight. Unfortunately, this test didn't work out as well as just using the argmax. Alas!
    # ifft = np.fft.ifft2(zph).real
    # yy,xx = np.indices(ifft.shape)
    # yy = np.fft.fftshift(yy) - (ifft.shape[0]//2)
    # xx = np.fft.fftshift(xx) - (ifft.shape[1]//2)
    # dyx0 = np.array([(ifft**2*yy).sum(), (ifft**2*xx).sum()]) / (ifft**2).sum()
    # print(f"{dyx0 = }")

    dyx = np.array(np.unravel_index(np.argmax(np.fft.ifft2(zph).real), fft0.shape), np.float64)
    #dyx = np.array(np.unravel_index(np.argmax(np.fft.ifft2(zph_pad).real), zph_pad.shape), np.float64)
    #del zph_pad
    #dyx /= target_multiplier # Correct for target multiplier padding

    #print(f"{dyx = }")
    for n in range(10):
        phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1])) * zph
        Jx = (-itau*kx*phasor).sum().real
        Jy = (-itau*ky*phasor).sum().real
        Hxx = (-tau**2*kx*kx*phasor).sum().real
        Hxy = (-tau**2*kx*ky*phasor).sum().real
        Hyy = (-tau**2*ky*ky*phasor).sum().real
        Hdet = Hxx*Hyy - Hxy*Hxy
        Hinv = np.array([[Hxx,-Hxy],[-Hxy,Hyy]])/Hdet
        step = Hinv @ np.array([Jy, Jx])
        dyx -= step
        #print(f"  iter {n}: dyx={dyx}, |step|={np.linalg.norm(step):.12e}")
        # No crop for full display!
        # crop = np.fft.ifft2(np.fft.fft2(img1)*phasor).real - img0
        # Crop display:
        #crop = np.fft.ifft2(fft1*phasor).real - img0[idx_crop]
        #crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        #cv2_imshow(crop, title=f'tess s4 c1 ccd4 f{idx2} iter {n}')
    
    #crop = np.fft.ifft2(fft1*phasor).real - img0[idx_crop]
    #crop = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)
    #cv2_imshow(crop, title=f'tess s4 c1 ccd4 f{idx2}')

    # Oh, we'll need to recalculate the whole phasor.
    ky,kx = np.meshgrid(np.fft.fftfreq(img0.shape[0]), np.fft.fftfreq(img0.shape[1]), indexing='ij')
    phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1]))
    crop = np.fft.ifft2(np.fft.fft2(img1)*phasor).real - img0
    cv2_imshow(crop, title=f'tess s4 c1 ccd4 f{idx2}')

# I hope I don't accidentally erase this one. With sector 19, cam 1, ccd 1, this is the coolest result I've seen so far.
# Its EXTREMELY stable. Holy shit. Lets see if we can first precompute the per-pixel median before starting the sequence. Its THAT cool.
# animate_cv2_diff_with_subpixel_adj_phasecorr_V1_BEWARE_OF_FLASHING_LIGHTS(sector_data, fltr=slog, ema_alpha=1-1e-3)
def animate_cv2_diff_with_subpixel_adj_phasecorr_V1_BEWARE_OF_FLASHING_LIGHTS(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    # Lets iterate through the entire sequence, and cache the resulting shifts in a map. Then we can apply the shifts with a single fft per frame, rather than iterating to convergence each frame. That should be much faster.
    print("Precomputing phase correlation shifts...")
    delta_coords = np.zeros((len(seq), 2), dtype=np.float64)
    for idt in range(1, len(seq)):
        delta_coords[idt] = phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
    print("Done precomputing phase correlation shifts!")

    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum][idx_crop])

        dyx = delta_coords[framenum] #phase_corr_iterator(ctrl_slice, raw)#[idx_slice])
        phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real

        diff = raw - ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 1: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        crop = diff#[idx_crop]
        big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


def animate_cv2_diff_with_subpixel_adj_phasecorr_V1_BEWARE_OF_FLASHING_LIGHTS_Full(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    #idx_crop = (slice(None), slice(None))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    #ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    # Lets iterate through the entire sequence, and cache the resulting shifts in a map. Then we can apply the shifts with a single fft per frame, rather than iterating to convergence each frame. That should be much faster.
    print("Precomputing phase correlation shifts...")
    delta_coords = np.zeros((len(seq), 2), dtype=np.float64)
    for idt in range(1, len(seq)):
        delta_coords[idt] = phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_slice]))
    print("Done precomputing phase correlation shifts!")

    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum])#[idx_crop])

        dyx = delta_coords[framenum] #phase_corr_iterator(ctrl_slice, raw)#[idx_slice])
        phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real

        diff = raw - ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 1: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        crop = diff#[idx_crop]
        #big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        #cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)
        cv2_imshow(crop, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)

        ema[:] = raw
        framenum += 1
        total_framenum += 1

# Has a few stable iterations. Its EXTREMELY promising. Lets go to the full linear matched filter later.
def animate_cv2_diff_with_subpixel_adj_phasecorr_V2(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    # Lets iterate through the entire sequence, and cache the resulting shifts in a map. Then we can apply the shifts with a single fft per frame, rather than iterating to convergence each frame. That should be much faster.
    print("Precomputing phase correlation shifts...")
    delta_coords = np.zeros((len(seq), 2), dtype=np.float64)
    for idt in range(1, len(seq)):
        delta_coords[idt] = phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
    print("Done precomputing phase correlation shifts!")

    # Now the potentially awful one. Lets iterate through the sequence, applying the subpixel shift, then cache a y axis slice to apply a median to get the median background.
    sector_median = np.zeros_like(ctrl_slice)
    
    # We should select our chunk size wisely. We're throwing out a LOT of data, but I cannot do the full spectral nuft for this without destroying the sector data.
    # sector_data.shape -> (1145, 2048, 2048)
    # at float64, that's 1145*2048*2048*8 bytes = 38.4 GB. Thats a lot, but maybe we can do it if we only keep one chunk in memory at a time. Lets say we do 128 pixel chunks, and we only keep the y axis slice for the median in memory. That would be 1145*128*2048*8 bytes = 2.3 GB, which is much more reasonable.
    # I can do 2.3GB! I'm at 73% usage of 64GB, which leaves ~17GB free. So our theoretical max chunk size is:
    # 17GB / (1145*2048*8 bytes) = 896 pixels. Lets go with 512. That grants us 4 iterations to compute the entire background.
    # Actually, this first test is a slice. Lets first validate that we wont kill the ram and throw an error if we will.

    chunksize = 512
    float64_sizeof = 8
    chunksize = min(chunksize, ctrl_slice.shape[0])
    num_chunks = (ctrl_slice.shape[0] + chunksize - 1) // chunksize

    est_nbytes = len(seq) * chunksize * ctrl_slice.shape[1] * float64_sizeof
    est_n_gb = est_nbytes / 1024**3
    print(f"Estimated memory usage for precomputing shifted frames: {est_n_gb:.2f} GB")
    lim_n_gb = 4.0
    print(f"Limiting malloc to {lim_n_gb:.2f} GB to avoid a scary pagefile.")
    if lim_n_gb < est_n_gb:
        raise MemoryError(f"Estimated memory usage of {est_n_gb:.2f} GB exceeds the limit of {lim_n_gb:.2f} GB. Aborting to avoid crashing the system.")
    
    tcx_chunk = np.zeros((len(seq), chunksize, ctrl_slice.shape[1]), dtype=np.float64)
    # icky O(n²) loop
    #for idy in range(len(seq)):
    for idc in range(num_chunks):
        y_start = idc*chunksize
        y_end = min((idc+1)*chunksize, ctrl_slice.shape[0])
        y_slice = slice(y_start, y_end)
        for idt in range(len(seq)):
            raw = fltr(seq[idt][idx_crop])
            phasor = np.exp(-2j*np.pi*(ky*delta_coords[idt][0] + kx*delta_coords[idt][1]))
            shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real
            tcx_chunk[idt, :y_end-y_start] = shifted[y_start:y_end]
            #tcx_chunk[idt, y_slice] = shifted[y_slice]
        # Now we have a chunk of the shifted frames in memory. We can compute the median across time for this chunk, and store it in the sector_median.
        sector_median[y_slice] = np.median(tcx_chunk[:,:y_end-y_start], axis=0)
        print(f"Computed median for chunk {idc+1}/{num_chunks}")
    del tcx_chunk # Free memory

    print("Done precomputing median background!")

    print("Recomputing phase correlation shifts from the median background...")
    # Iterate again to recompute delta_coords from the median as our control!
    # Also get bounds.
    vmin, vmax = np.inf, -np.inf
    for idt in range(len(seq)):
        raw = fltr(seq[idt][idx_crop])
        delta_coords[idt] = phase_corr_iterator(sector_median, raw)
        phasor = np.exp(-2j*np.pi*(ky*delta_coords[idt][0] + kx*delta_coords[idt][1]))
        shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real
        diff = shifted - sector_median
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        vmin = min(vmin, clow)
        vmax = max(vmax, chigh)
    print(f"Done recomputing phase correlation shifts! vmin={vmin}, vmax={vmax}")

    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum][idx_crop])

        dyx = delta_coords[framenum] #phase_corr_iterator(ctrl_slice, raw)#[idx_slice])
        phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real

        """
        diff = raw - ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 1: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale
        """

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


# Lets make a quick estimate of the scientific way of writing "3e-15 degrees".
# e-3: milli
# e-6: micro
# e-9: nano
# e-12: pico
# e-15: femto
# So 3e-15 degrees is 3 femto-degrees. That sounds about right for the precision we're talking about here.
if True:
    framenum = 75

    theta = np.radians(0*5e-2)  # 50 millidegrees
    #Lambda = np.array([[np.cos(theta),  np.sin(theta)],
    #                       [-np.sin(theta), np.cos(theta)]])
    Lambda = np.array([
        [1.0+1e-4, 0.0],
        [0.0, 1.0]
    ])
    Lambda_inv = np.linalg.inv(Lambda)

    rotated = hc.affine_resample(sector_data[framenum], Lambda_inv)
    #ctrl = hc.affine_resample(sector_data[framenum], np.identity(2, dtype=np.float64))
    #cv2_imshow(cv2.resize(rotated - ctrl, (512,512), interpolation=cv2.INTER_LANCZOS4))

    # ky,kx,delta_coords were computed earlier

    A,B,dy,dx = delta_coords[framenum]
    phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
    result = np.fft.ifft2(np.fft.fft2(rotated)*phasor).real*A + B

    med_thru_fltr = hc.affine_resample(sector_median, np.identity(2, dtype=np.float64))

    cv2_imshow(cv2.resize(result - med_thru_fltr, (512,512), interpolation=cv2.INTER_LANCZOS4))

# Now lets do the same but with a sweeping grid of ε values for scales on the affine transform!
def test_affine_matched_filter_model(framenum=75, scale=1e-4):

    ky,kx = np.meshgrid(
        np.fft.fftfreq(sector_data[framenum].shape[0]),
        np.fft.fftfreq(sector_data[framenum].shape[1]),
    indexing='ij')

    yax_ = np.linspace(-scale, scale, 17)
    xax_ = np.linspace(-scale, scale, 17)

    med_thru_fltr = hc.affine_resample(sector_median, np.identity(2, dtype=np.float64))
    
    results = np.zeros((len(yax_), len(xax_)), dtype=np.float64)
    for iy, εy in enumerate(yax_):
        for ix, εx in enumerate(xax_):
            Lambda = np.array([
                [1.0+εy, 0.0],
                [0.0, 1.0+εx]
            ])
            Lambda_inv = np.linalg.inv(Lambda)

            rotated = hc.affine_resample(sector_data[framenum], Lambda_inv)
            A,B,dy,dx = delta_coords[framenum]
            
            phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
            result = np.fft.ifft2(np.fft.fft2(rotated)*phasor).real*A + B
            results[iy, ix] = np.mean((result - med_thru_fltr)**2)

    return results

# 3D surface plot!!!
def plot_affine_matched_filter_results():
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    results = test_affine_matched_filter_model()

    # the arange doesn't quite work, as our outputs are a grid.
    xgrid, ygrid = np.meshgrid(np.linspace(-1e-4, 1e-4, 17), np.linspace(-1e-4, 1e-4, 17))

    ax.plot_surface(xgrid, ygrid, results, cmap='viridis')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('MSE')
    ax.set_aspect('equal')
    plt.show()

# It actually shows an optimum exists. Classic L2 loss landscape with an obvious minima. Exciting!!
# Lets make a super simple optimizer to ignore gradient desent for the time being by using an ensemble of 16 samples in a gaussian around our current value, picking the best, estimating the curvature from the output, and iterating. Lets only use uniform gaussians for this.
def monte_carlo_anneal_affine_matched_filter(delta_coords_affine, med, framenum=75, scale=1e-4, spp=16, iters=100):

    params = np.copy(delta_coords_affine[framenum])
    n_params = len(params)

    # Scale-aware initial std: each parameter gets std proportional to
    # its expected magnitude, so all parameters are explored proportionally.
    # params = [A, B, dy, dx, a, b, c, d]
    #           ~1  ~0.4  ~0.001  ~0.001  ~1  ~0  ~0  ~1
    param_scales = np.array([
        max(abs(params[0]), 0.01),   # A
        max(abs(params[1]), 0.01),   # B
        0.01,                         # dy (shifts are ~millipixels)
        0.01,                         # dx
        max(abs(params[4]), 0.01),   # a
        0.01,                         # b (off-diag start near 0)
        0.01,                         # c
        max(abs(params[7]), 0.01),   # d
    ])
    params_std = param_scales * scale

    ky, kx = np.meshgrid(
        np.fft.fftfreq(sector_data[framenum].shape[0]),
        np.fft.fftfreq(sector_data[framenum].shape[1]),
        indexing='ij')

    # Precompute the reference once (it never changes within this call)
    med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))

    def eval_loss(p):
        A_s, B_s, dy_s, dx_s, a_s, b_s, c_s, d_s = p
        Lambda = np.array([[a_s, b_s], [c_s, d_s]])
        Lambda_inv = np.linalg.inv(Lambda)
        rotated = hc.affine_resample(sector_data[framenum], Lambda_inv)
        phasor = np.exp(-2j * np.pi * (ky * dy_s + kx * dx_s))
        result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A_s + B_s
        return np.mean((result - med_thru_fltr)**2), result

    current_loss, _ = eval_loss(params)

    for epoch in range(iters):
        dp = np.random.normal(0.0, params_std, size=(spp, n_params))
        samples = params + dp
        losses = np.zeros(spp, dtype=np.float64)
        for i in range(spp):
            losses[i], _ = eval_loss(samples[i])

        # ── Build design matrix: [1, dp_i, dp_i·dp_j (upper triangle)] ──
        n_quad = n_params * (n_params + 1) // 2
        n_coeffs = 1 + n_params + n_quad
        X = np.empty((spp, n_coeffs), dtype=np.float64)
        for i in range(spp):
            X[i, 0] = 1.0
            X[i, 1:1 + n_params] = dp[i]
            idx = 1 + n_params
            for j in range(n_params):
                for k in range(j, n_params):
                    X[i, idx] = dp[i, j] * dp[i, k]
                    idx += 1

        # ── Least-squares fit ──
        coeffs, residuals_ls, rank, sv = np.linalg.lstsq(X, losses, rcond=None)
        pred = X @ coeffs
        s2 = np.mean((losses - pred)**2)  # residual variance of quadratic fit

        # Extract gradient B and Hessian-like C
        B_grad = coeffs[1:1 + n_params]
        C = np.zeros((n_params, n_params), dtype=np.float64)
        idx = 0
        for j in range(n_params):
            for k in range(j, n_params):
                C[j, k] = coeffs[1 + n_params + idx]
                C[k, j] = coeffs[1 + n_params + idx]
                idx += 1

        improved = False
        try:
            # ── Newton step: minimum of quadratic at dp* = -0.5 C^{-1} B ──
            step = -0.5 * np.linalg.solve(C, B_grad)

            # ── Gaussian posterior std for the step ──
            # The covariance of the LS coefficients is: Cov(β) = s² (XᵀX)⁻¹
            # The step dp* = -0.5 C⁻¹ B depends linearly on B (holding C fixed).
            # Error propagation: Cov(dp*) = 0.25 C⁻¹ Cov(B) C⁻ᵀ
            # where Cov(B) is the [1:d+1, 1:d+1] sub-block of Cov(β).
            XtX_inv = np.linalg.inv(X.T @ X)
            Cov_B = s2 * XtX_inv[1:1 + n_params, 1:1 + n_params]
            C_inv = np.linalg.inv(C)
            Cov_step = 0.25 * C_inv @ Cov_B @ C_inv.T

            # Posterior std = uncertainty in the optimum location.
            # Blend with current std via geometric mean to avoid jumps.
            posterior_std = np.sqrt(np.maximum(np.diag(Cov_step), 1e-30))
            # Also include a "trust region" from the Hessian curvature:
            # the Laplace approximation gives σ_i = sqrt((2C)^{-1}_{ii})
            # which is how far you can move before the loss increases by 0.5.
            hessian_std = np.sqrt(np.maximum(np.diag(np.linalg.inv(2.0 * C)), 1e-30))

            # Take the smaller of propagated uncertainty and Hessian trust region,
            # but allow growth (no np.minimum with old params_std).
            params_std = np.minimum(posterior_std, hessian_std)
            # Floor: never go below 1e-3 of the parameter scale to maintain exploration.
            params_std = np.maximum(params_std, param_scales * 1e-8)

            new_params = params + step
            new_loss, result = eval_loss(new_params)

            # Best of: current, Newton step, best sample
            best_idx = np.argmin(losses)
            candidates = [(current_loss, params), (new_loss, new_params), (losses[best_idx], samples[best_idx])]
            best_loss, best_params = min(candidates, key=lambda x: x[0])

            if best_loss < current_loss:
                current_loss = best_loss
                params = best_params
                improved = True
                _, result = eval_loss(params)

        except np.linalg.LinAlgError:
            best_idx = np.argmin(losses)
            if losses[best_idx] < current_loss:
                current_loss = losses[best_idx]
                params = samples[best_idx]
                improved = True
                _, result = eval_loss(params)
            # Widen std when the Hessian is singular — we need more exploration
            params_std *= 2.0

        if improved:
            cv2_imshow(cv2.resize(result - med_thru_fltr, (512, 512),
                       interpolation=cv2.INTER_LANCZOS4),
                       title=f'[{framenum:04d}] loss={current_loss:.2e} @ epoch={epoch}')

    return params, result

# Fit ALL of the items in the current dataset.
def fit_affine_matched_filter_model_to_dataset(delta_coords_affine, med, scale=1e-4, spp=128, iters=5):
    # >>> delta_coords.shape
    # (128,4)
    # Lets make a NEW delta_coords for this variant of the dataset.
    delta_coords_affine_new = np.zeros_like(delta_coords_affine)
    frames = []
    for i in range(delta_coords_affine.shape[0]):
        delta_coords_affine_new[i], frame = monte_carlo_anneal_affine_matched_filter(delta_coords_affine, med, framenum=i, scale=scale, spp=spp, iters=iters)
        frames.append(frame)
    return delta_coords_affine_new, np.array(frames)

def iterate_fit_affine_matched_filter_model_to_dataset(scale=1e-4, spp=128, iters=5, epochs=10):
    #delta_coords_affine, np.array(frames)
    delta_coords_affine = np.zeros((delta_coords.shape[0], 8), dtype=np.float64)
    delta_coords_affine[:,:4] = delta_coords
    delta_coords_affine[:,4] = 1.0 # a
    delta_coords_affine[:,5] = 0.0 # b
    delta_coords_affine[:,6] = 0.0 # c
    delta_coords_affine[:,7] = 1.0 # d

    med = sector_median.copy()
    for epoch in range(epochs):
        delta_coords_affine, frames = fit_affine_matched_filter_model_to_dataset(delta_coords_affine, med, scale=scale, spp=spp, iters=iters if epoch != 0 else 1)
        med = np.median(frames, axis=0)
        #med = np.mean(frames, axis=0)
    return delta_coords_affine, frames, med

def animate_cv2_from_median(seq, median):
    framenum = 0
    seq = seq - median
    Min = seq.min()
    Max = seq.max()
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        #diff = seq[framenum] - median
        diff = cv2.resize(seq[framenum], (512,512), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(diff, title=f'tess s4 c1 ccd4 f{framenum}', Min=Min, Max=Max)
        framenum += 1




from scipy.ndimage import map_coordinates

def centered_xy(shape):
    H, W = shape
    y = np.arange(H, dtype=np.float64) - H / 2.0
    x = np.arange(W, dtype=np.float64) - W / 2.0
    Y, X = np.meshgrid(y, x, indexing='ij')
    return Y, X

def centered_k(shape):
    H, W = shape
    ky1 = np.fft.fftshift(np.fft.fftfreq(H))
    kx1 = np.fft.fftshift(np.fft.fftfreq(W))
    ky, kx = np.meshgrid(ky1, kx1, indexing='ij')
    return ky, kx

def spectral_moment_fields(img):
    """
    Build the FFT fields needed by affine_loss_grad_hess.

    F0   = FFT(L)
    Fx   = FFT(x * L)
    Fy   = FFT(y * L)
    Fxx  = FFT(x^2 * L)
    Fxy  = FFT(x y * L)
    Fyy  = FFT(y^2 * L)

    Everything is fftshifted so interpolation happens on a centered frequency plane.

    In addition, we isolate the unshifted DC component f0 = F0[0,0].
    """
    img = np.asarray(img, dtype=np.float64)
    Y, X = centered_xy(img.shape)

    #F0  = np.fft.fftshift(np.fft.fft2(img))
    F0 = np.fft.fft2(img)
    f0 = F0[0,0].real

    F0  = np.fft.fftshift(F0)
    Fy  = np.fft.fftshift(np.fft.fft2(Y * img))
    Fx  = np.fft.fftshift(np.fft.fft2(X * img))
    Fyy = np.fft.fftshift(np.fft.fft2((Y * Y) * img))
    Fxy = np.fft.fftshift(np.fft.fft2((X * Y) * img))
    Fxx = np.fft.fftshift(np.fft.fft2((X * X) * img))
    return f0, F0, Fx, Fy, Fxx, Fxy, Fyy

def interp_conj_cpu(F, qy, qx, order=1):
    """
    F   : complex array on fftshifted frequency grid
    qx,qy : frequency coordinates in fftfreq units, same shape as target grid
            expected range roughly [-0.5, 0.5)
    order : 1=bilinear, 3=cubic

    Returns conj(F)(qy,qx) sampled with periodic wrap.
    """
    H, W = F.shape

    # map fftfreq units to fftshifted pixel coordinates
    # qx = -0.5 maps near 0, qx = 0 maps near W/2, qx -> +0.5 maps near W
    y = qy * H + H / 2.0
    x = qx * W + W / 2.0
    
    coords = np.array([y, x])  # ndimage expects row,col
    #coords = np.array([x, y])  # ndimage expects row,col

    real = map_coordinates(F.real, coords, order=order, mode='wrap')
    imag = map_coordinates(F.imag, coords, order=order, mode='wrap')
    return real - 1j * imag   # conjugate after interpolation

def affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1):
    ky, kx = centered_k(F0.shape)

    # q = Λᵀ·k  in (row, col) convention:  Λ = [[a,b],[c,d]]
    # Λᵀ = [[a,c],[b,d]],  k = [ky, kx]ᵀ
    qy = a * ky + c * kx    # row-frequency component
    qx = b * ky + d * kx    # col-frequency component

    Lc  = interp_conj_cpu(F0,  qy, qx, order=order)
    Lx  = interp_conj_cpu(Fx,  qy, qx, order=order)
    Ly  = interp_conj_cpu(Fy,  qy, qx, order=order)
    Lxx = interp_conj_cpu(Fxx, qy, qx, order=order)
    Lxy = interp_conj_cpu(Fxy, qy, qx, order=order)
    Lyy = interp_conj_cpu(Fyy, qy, qx, order=order)

    return Lc, Lx, Ly, Lxx, Lxy, Lyy, kx, ky

def affine_loss_grad_hess_original(params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky):#, interp_conj):
    A, B, dy, dx, a, b, c, d = params
    det = a*d - b*c

    # q = Λᵀk in (row,col) convention:  Λ = [[a,b],[c,d]], k = [ky,kx]ᵀ
    # qy = a*ky + c*kx (row-freq),  qx = b*ky + d*kx (col-freq)
    #
    # Lx = ∂L̂†/∂q_col (from Fx = FFT(X·img), X = col coord)
    # Ly = ∂L̂†/∂q_row (from Fy = FFT(Y·img), Y = row coord)
    #
    # Chain rule for Λ elements (row,col convention):
    #   a → ∂q_row/∂a = ky  →  Ka uses ky·Ly
    #   b → ∂q_col/∂b = ky  →  Kb uses ky·Lx
    #   c → ∂q_row/∂c = kx  →  Kc uses kx·Ly
    #   d → ∂q_col/∂d = kx  →  Kd uses kx·Lx

    #affine_spectral_fields
    Lc, Lx, Ly, Lxx, Lxy, Lyy, kx, ky = affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1)

    ph = np.exp(-2j*np.pi*(kx*dx + ky*dy))
    Z  = ph * Lc * Mhat

    Dx = -2j*np.pi*kx
    Dy = -2j*np.pi*ky

    # Base moments
    K   = np.sum(Z).real
    Kx  = np.sum(Dx*Z).real
    Ky  = np.sum(Dy*Z).real
    Kxx = np.sum(Dx*Dx*Z).real
    Kxy = np.sum(Dx*Dy*Z).real
    Kyy = np.sum(Dy*Dy*Z).real

    # First affine moments  (row,col convention: a→ky·Ly, b→ky·Lx, c→kx·Ly, d→kx·Lx)
    Ka = np.sum(ph * (ky*Ly) * Mhat).real
    Kb = np.sum(ph * (ky*Lx) * Mhat).real
    Kc = np.sum(ph * (kx*Ly) * Mhat).real
    Kd = np.sum(ph * (kx*Lx) * Mhat).real

    # Translation-affine mixed moments
    Kxa = np.sum(Dx * ph * (ky*Ly) * Mhat).real
    Kxb = np.sum(Dx * ph * (ky*Lx) * Mhat).real
    Kxc = np.sum(Dx * ph * (kx*Ly) * Mhat).real
    Kxd = np.sum(Dx * ph * (kx*Lx) * Mhat).real

    Kya = np.sum(Dy * ph * (ky*Ly) * Mhat).real
    Kyb = np.sum(Dy * ph * (ky*Lx) * Mhat).real
    Kyc = np.sum(Dy * ph * (kx*Ly) * Mhat).real
    Kyd = np.sum(Dy * ph * (kx*Lx) * Mhat).real

    # Second affine moments  (same row,col pattern applied to L̂† second derivatives)
    Kaa = np.sum(ph * (ky*ky*Lyy) * Mhat).real
    Kab = np.sum(ph * (ky*ky*Lxy) * Mhat).real
    Kac = np.sum(ph * (ky*kx*Lyy) * Mhat).real
    Kad = np.sum(ph * (ky*kx*Lxy) * Mhat).real

    Kbb = np.sum(ph * (ky*ky*Lxx) * Mhat).real
    Kbc = np.sum(ph * (ky*kx*Lxy) * Mhat).real
    Kbd = np.sum(ph * (ky*kx*Lxx) * Mhat).real

    Kcc = np.sum(ph * (kx*kx*Lyy) * Mhat).real
    Kcd = np.sum(ph * (kx*kx*Lxy) * Mhat).real
    Kdd = np.sum(ph * (kx*kx*Lxx) * Mhat).real
    # These sure feel like an object from general relativity! I suppose since we started from a "distance" that it makes sense that the curvature of the loss landscape would be described by a metric-like object. Very cool.

    # Constants
    EL = np.sum(np.abs(F0)**2).real / F0.size
    #l0 = F0[0, 0].real
    #m0 = Mhat[0, 0].real
    l0 = f0
    Om = F0.size

    S0 = 0.5*A*A*EL + A*B*l0 - A*K
    SA = A*EL + B*l0 - K

    # Compute the reduced loss (not including the constant term from Mhat, which doesn't affect optimization)
    L = det * S0 + 0.5*Om*B*B - m0*B
    # Add the Mhat term back
    L = L + m0 

    # det first derivatives
    ja, jb, jc, jd = d, -c, -b, a

    # Gradient
    J = np.zeros(8, dtype=np.float64)
    J[0] = det * SA
    J[1] = Om*B + (A*det*l0 - m0)
    J[2] = -A*det*Ky   # dy
    J[3] = -A*det*Kx   # dx
    J[4] = ja*S0 - A*det*Ka
    J[5] = jb*S0 - A*det*Kb
    J[6] = jc*S0 - A*det*Kc
    J[7] = jd*S0 - A*det*Kd

    # Hessian
    H = np.zeros((8, 8), dtype=np.float64)

    # A/B block
    H[0,0] = det * EL
    H[0,1] = det * l0
    H[1,1] = Om

    # A / translation
    H[0,2] = -det * Ky
    H[0,3] = -det * Kx

    # translation / translation
    H[2,2] = -A * det * Kyy
    H[2,3] = -A * det * Kxy
    H[3,3] = -A * det * Kxx

    # A / affine
    H[0,4] = ja*SA - det*Ka
    H[0,5] = jb*SA - det*Kb
    H[0,6] = jc*SA - det*Kc
    H[0,7] = jd*SA - det*Kd

    # B / affine
    H[1,4] = A * ja * l0
    H[1,5] = A * jb * l0
    H[1,6] = A * jc * l0
    H[1,7] = A * jd * l0

    # translation / affine
    H[2,4] = -A * (ja*Ky + det*Kya)
    H[2,5] = -A * (jb*Ky + det*Kyb)
    H[2,6] = -A * (jc*Ky + det*Kyc)
    H[2,7] = -A * (jd*Ky + det*Kyd)

    H[3,4] = -A * (ja*Kx + det*Kxa)
    H[3,5] = -A * (jb*Kx + det*Kxb)
    H[3,6] = -A * (jc*Kx + det*Kxc)
    H[3,7] = -A * (jd*Kx + det*Kxd)

    # affine / affine second derivatives of det
    # only h_ad = h_da = +1, h_bc = h_cb = -1
    h = np.zeros((4,4), dtype=np.float64)
    h[0,3] = h[3,0] = 1.0   # a,d
    h[1,2] = h[2,1] = -1.0  # b,c

    j = np.array([ja, jb, jc, jd], dtype=np.float64)
    K1 = np.array([Ka, Kb, Kc, Kd], dtype=np.float64)
    K2 = np.array([
        [Kaa, Kab, Kac, Kad],
        [Kab, Kbb, Kbc, Kbd],
        [Kac, Kbc, Kcc, Kcd],
        [Kad, Kbd, Kcd, Kdd],
    ], dtype=np.float64)

    for r in range(4):
        for s in range(r, 4):
            #val = h[r,s]*S0 + j[r]*j[s]*S0 - A*det*(j[r]*K1[s] + j[s]*K1[r] + K2[r,s])
            val = h[r,s]*S0 - A*(j[r]*K1[s] + j[s]*K1[r] + det*K2[r,s])
            H[4+r, 4+s] = val
            #H[4+s, 4+r] = val

    # Symmetrize
    H = H + H.T - np.diag(np.diag(H))
    return L, J, H

def affine_loss_grad_hess(params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky):#, interp_conj):
    A, B, dy, dx, a, b, c, d = params
    det = a*d - b*c
    iΛ = np.array([
        [a, b],
        [c, d],
    ])
    Λ = np.array([
        [d, -b],
        [-c, a],
    ]) / det

    # q = Λᵀk in (row,col) convention:  Λ = [[a,b],[c,d]], k = [ky,kx]ᵀ
    # qy = a*ky + c*kx (row-freq),  qx = b*ky + d*kx (col-freq)
    #
    # Lx = ∂L̂†/∂q_col (from Fx = FFT(X·img), X = col coord)
    # Ly = ∂L̂†/∂q_row (from Fy = FFT(Y·img), Y = row coord)
    #
    # Chain rule for Λ elements (row,col convention):
    #   a → ∂q_row/∂a = ky  →  Ka uses ky·Ly
    #   b → ∂q_col/∂b = ky  →  Kb uses ky·Lx
    #   c → ∂q_row/∂c = kx  →  Kc uses kx·Ly
    #   d → ∂q_col/∂d = kx  →  Kd uses kx·Lx

    #affine_spectral_fields
    Lc, Lx, Ly, Lxx, Lxy, Lyy, kx, ky = affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1)
    
    im = 1j # why does python do this
    
    # Phase correlator arguments
    ph = np.exp(-2*im*np.pi*(kx*dx + ky*dy))
    Z  = ph * Mhat
    
    C =  (Lc*Z).real.mean()
    Cy = ((-im*ky)*Lc*Z).real.mean()
    Cx = ((-im*kx)*Lc*Z).real.mean()
    
    Cy_y = ((-im*ky)*Ly*Z).real.mean()
    Cy_x = ((-im*ky)*Lx*Z).real.mean()
    Cx_y = ((-im*kx)*Ly*Z).real.mean()
    Cx_x = ((-im*kx)*Lx*Z).real.mean()
    
    Cyy = ((-im*ky)*(-im*ky)*Lc*Z).real.mean()
    #Cyx = ((-im*ky)*(-im*kx)*Lc*Z).real.mean() # redundant
    Cxy = ((-im*kx)*(-im*ky)*Lc*Z).real.mean()
    Cxx = ((-im*kx)*(-im*kx)*Lc*Z).real.mean()
    
    Cyy_y = ((-im*ky)*(-im*ky)*Ly*Z).real.mean()
    Cyy_x = ((-im*ky)*(-im*ky)*Lx*Z).real.mean()
    Cyx_y = ((-im*ky)*(-im*kx)*Ly*Z).real.mean()
    Cyx_x = ((-im*ky)*(-im*kx)*Lx*Z).real.mean()
    Cxy_y = ((-im*kx)*(-im*ky)*Ly*Z).real.mean() # redundant?
    Cxy_x = ((-im*kx)*(-im*ky)*Lx*Z).real.mean() # redundant?
    Cxx_y = ((-im*kx)*(-im*kx)*Ly*Z).real.mean()
    Cxx_x = ((-im*kx)*(-im*kx)*Lx*Z).real.mean()
    
    Cyy_yy = ((-im*ky)*(-im*ky)*Lyy*Z).real.mean()
    Cyy_yx = ((-im*ky)*(-im*ky)*Lxy*Z).real.mean()
    Cyy_xy = ((-im*ky)*(-im*ky)*Lxy*Z).real.mean() # redundant? # Cyy_xy = Cyy_yx
    Cyy_xx = ((-im*ky)*(-im*ky)*Lxx*Z).real.mean()
    Cyx_yy = ((-im*ky)*(-im*kx)*Lyy*Z).real.mean()
    Cyx_yx = ((-im*ky)*(-im*kx)*Lxy*Z).real.mean()
    Cyx_xy = ((-im*ky)*(-im*kx)*Lxy*Z).real.mean() # redundant? # Cyx_xy = Cyx_yx
    Cyx_xx = ((-im*ky)*(-im*kx)*Lxx*Z).real.mean()
    Cxy_yy = ((-im*kx)*(-im*ky)*Lyy*Z).real.mean()
    Cxy_yx = ((-im*kx)*(-im*ky)*Lxy*Z).real.mean()
    Cxy_xy = ((-im*kx)*(-im*ky)*Lxy*Z).real.mean() # redundant? # Cxy_xy = Cxy_yx
    Cxy_xx = ((-im*kx)*(-im*ky)*Lxx*Z).real.mean()
    Cxx_yy = ((-im*kx)*(-im*kx)*Lyy*Z).real.mean()
    Cxx_yx = ((-im*kx)*(-im*kx)*Lxy*Z).real.mean()
    Cxx_xy = ((-im*kx)*(-im*kx)*Lxy*Z).real.mean() # redundant? # Cxx_xy = Cxx_yx
    Cxx_xx = ((-im*kx)*(-im*kx)*Lxx*Z).real.mean()
    
    
    
    # Constants
    Om = 1.0#F0.size
    LL = (np.abs(F0)**2).real.mean()#/Om
    MM = (np.abs(Mhat)**2).real.mean()#/Om
    #l0 = F0[0, 0].real
    #m0 = Mhat[0, 0].real
    l0 = f0
    #Om = F0.size

    # Compute the Loss
    L = 0.5*(det*A**2*LL + B**2 + MM + 2.0*B*(l0*A*det - m0) - 2.0*A*det*C)
    
    # Compute the Gradient
    dA = det*(A*LL + B*l0 - C)
    dB = B*Om + l0*A*det - m0
    dΔy = -A*det*Cy
    dΔx = -A*det*Cx
    dΛyy = iΛ.T[0,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cy_y
    dΛyx = iΛ.T[0,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cy_x
    dΛxy = iΛ.T[1,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cx_y
    dΛxx = iΛ.T[1,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cx_x
    J = np.array([dA, dB, dΔy, dΔx, dΛyy, dΛyx, dΛxy, dΛxx], dtype=np.float64)
    
    # Compute the Hessian
    dAdA = det*LL
    dAdB = det*l0
    dAdΔy = -det*Cy
    dAdΔx = -det*Cx
    dAdΛyy = iΛ.T[0,0]*det*(A*LL + B*l0 - C) + iΛ.T[0,0]*det*A*LL - det*Cy_y
    dAdΛyx = iΛ.T[0,1]*det*(A*LL + B*l0 - C) + iΛ.T[0,1]*det*A*LL - det*Cy_x
    dAdΛxy = iΛ.T[1,0]*det*(A*LL + B*l0 - C) + iΛ.T[1,0]*det*A*LL - det*Cx_y
    dAdΛxx = iΛ.T[1,1]*det*(A*LL + B*l0 - C) + iΛ.T[1,1]*det*A*LL - det*Cx_x
    
    dBdB = Om
    dBdΔy = 0.0
    dBdΔx = 0.0
    dBdΛyy = iΛ.T[0,0]*det*l0
    dBdΛyx = iΛ.T[0,1]*det*l0
    dBdΛxy = iΛ.T[1,0]*det*l0
    dBdΛxx = iΛ.T[1,1]*det*l0
    
    dΔydΔy = -A*det*Cyy
    dΔydΔx = -A*det*Cxy
    dΔydΛyy = -A*iΛ.T[0,0]*det*Cy - A*det*Cyy_y
    dΔydΛyx = -A*iΛ.T[0,1]*det*Cy - A*det*Cyy_x
    dΔydΛxy = -A*iΛ.T[1,0]*det*Cy - A*det*Cyx_y
    dΔydΛxx = -A*iΛ.T[1,1]*det*Cy - A*det*Cyx_x
    
    dΔxdΔx = -A*det*Cxx
    dΔxdΛyy = -A*iΛ.T[0,0]*det*Cx - A*det*Cxy_y
    dΔxdΛyx = -A*iΛ.T[0,1]*det*Cx - A*det*Cxy_x
    dΔxdΛxy = -A*iΛ.T[1,0]*det*Cx - A*det*Cxx_y
    dΔxdΛxx = -A*iΛ.T[1,1]*det*Cx - A*det*Cxx_x
    
    #dΛijdΛyz = det*((iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(A**2*LL/2 + A*B*l0
    # - A*det*(
    # (iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*C
    # + iΛ[j,y]*C_yz
    # + iΛ[z,y]*C_ij
    # + C_yijz
    # )
    
    Ci_j = np.array([
        [Cyy, Cxy],
        [Cxy, Cxx],
    ], dtype=np.float64)
    
    Cij_yz = np.array([
        Cyy_yy, Cyy_yx, Cyy_xy, Cyy_xx,
        Cyx_yy, Cyx_yx, Cyx_xy, Cyx_xx,
        Cxy_yy, Cxy_yx, Cxy_xy, Cxy_xx,
        Cxx_yy, Cxx_yx, Cxx_xy, Cxx_xx,
    ], dtype=np.float64).reshape((2,2,2,2))
    
    dΛijdΛyz = np.zeros((2,2,2,2), dtype=np.float64)
    for i in range(2):
        for j in range(2):
            for y in range(2):
                for z in range(2):
                    dΛijdΛyz[i,j,y,z] = det*(iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(0.5*A**2*LL + A*B*l0 - A*C)
                    dΛijdΛyz[i,j,y,z] -= A*det*(iΛ[z,y]*Ci_j[i,j] + iΛ[j,i]*Ci_j[y,z] + Cij_yz[y,i,j,z])
                    
    dΛyydΛyy = dΛijdΛyz[0,0,0,0]
    dΛyydΛyx = dΛijdΛyz[0,0,0,1]
    dΛyydΛxy = dΛijdΛyz[0,0,1,0]
    dΛyydΛxx = dΛijdΛyz[0,0,1,1]
    dΛyxdΛyy = dΛijdΛyz[0,1,0,0]
    dΛyxdΛyx = dΛijdΛyz[0,1,0,1]
    dΛyxdΛxy = dΛijdΛyz[0,1,1,0]
    dΛyxdΛxx = dΛijdΛyz[0,1,1,1]
    dΛxydΛyy = dΛijdΛyz[1,0,0,0]
    dΛxydΛyx = dΛijdΛyz[1,0,0,1]
    dΛxydΛxy = dΛijdΛyz[1,0,1,0]
    dΛxydΛxx = dΛijdΛyz[1,0,1,1]
    dΛxxdΛyy = dΛijdΛyz[1,1,0,0]
    dΛxxdΛyx = dΛijdΛyz[1,1,0,1]
    dΛxxdΛxy = dΛijdΛyz[1,1,1,0]
    dΛxxdΛxx = dΛijdΛyz[1,1,1,1]
    
    H = np.array([
        [dAdA, dAdB, dAdΔy, dAdΔx, dAdΛyy, dAdΛyx, dAdΛxy, dAdΛxx],
        [dAdB, dBdB, dBdΔy, dBdΔx, dBdΛyy, dBdΛyx, dBdΛxy, dBdΛxx],
        [dAdΔy, dBdΔy, dΔydΔy, dΔydΔx, dΔydΛyy, dΔydΛyx, dΔydΛxy, dΔydΛxx],
        [dAdΔx, dBdΔx, dΔydΔx, dΔxdΔx, dΔxdΛyy, dΔxdΛyx, dΔxdΛxy, dΔxdΛxx],
        [dAdΛyy, dBdΛyy, dΔydΛyy, dΔxdΛyy, dΛyydΛyy, dΛyydΛyx, dΛyydΛxy, dΛyydΛxx],
        [dAdΛyx, dBdΛyx, dΔydΛyx, dΔxdΛyx, dΛyxdΛyy, dΛyxdΛyx, dΛyxdΛxy, dΛyxdΛxx],
        [dAdΛxy, dBdΛxy, dΔydΛxy, dΔxdΛxy, dΛxydΛyy, dΛxydΛyx, dΛxydΛxy, dΛxydΛxx],
        [dAdΛxx, dBdΛxx, dΔydΛxx, dΔxdΛxx, dΛxxdΛyy, dΛxxdΛyx, dΛxxdΛxy, dΛxxdΛxx],
    ])
    
    return L, J, H


# Simple test.
def test_affine_spectral_fields_and_grad_hess(framenum=75):
    # Identity-ish parameters
    #params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    #params = 
    # Lets select the existing affine parameter vector.
    #params = delta_coords_affine[framenum].copy()

    img = np.asarray(sector_data[framenum], dtype=np.float64)
    med = np.asarray(sector_median, dtype=np.float64)

    # Moving-image spectral moment fields
    f0, F0, Fx, Fy, Fxx, Fxy, Fyy = spectral_moment_fields(img)

    # Reference/template spectrum
    #Mhat = np.fft.fftshift(np.fft.fft2(med))
    Mhat = np.fft.fft2(med)
    m0 = Mhat[0,0].real
    Mhat = np.fft.fftshift(Mhat)

    # Centered frequency grids for fftshifted arrays
    ky, kx = centered_k(img.shape)

    L, J, H = affine_loss_grad_hess(
        params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky
    )

    # Gauss newton step
    try:
        step = -0.5*np.linalg.solve(H, J)
    except np.linalg.LinAlgError:
        step = np.zeros_like(params)

    globals().update(locals())

def validate_affine_spectral_fields_and_grad_hess(framenum=75):
    img = np.asarray(sector_data[framenum], dtype=np.float64)
    med = np.asarray(sector_median, dtype=np.float64)

    # Moving-image spectral moment fields
    f0, F0, Fx, Fy, Fxx, Fxy, Fyy = spectral_moment_fields(img)

    # Reference/template spectrum
    #Mhat = np.fft.fftshift(np.fft.fft2(med))
    Mhat = np.fft.fft2(med)
    m0 = Mhat[0,0].real
    Mhat = np.fft.fftshift(Mhat)

    # Centered frequency grids for fftshifted arrays
    ky, kx = centered_k(img.shape)

    # Gather different points to validate that the gradient is what we expect
    params = np.copy(delta_coords_affine[framenum])

    L, J, H = affine_loss_grad_hess(
        params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky
    )
    
    Ln = np.zeros_like(J)
    ε = 1e-6
    for i in range(len(params)):
        dp = np.zeros_like(params)
        dp[i] = ε
        pn = params + dp
        Lp, J_, H_ = affine_loss_grad_hess(pn, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky)
        Ln[i] = (Lp - L) / ε
    globals().update(locals())
    

# Step is behaving. Lets try an optimizer!
def test_affine_spectral_fields_and_grad_hess_optimizer(framenum=75, epochs=10, rate=1.0, LMDamping=1e-12):
    # Identity-ish parameters
    #params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
    #params =
    # Lets select the existing affine parameter vector.
    params = delta_coords_affine[framenum].copy()
    
    img = np.asarray(sector_data[framenum], dtype=np.float64)
    med = np.asarray(sector_median, dtype=np.float64)
    
    # Moving-image spectral moment fields
    f0, F0, Fx, Fy, Fxx, Fxy, Fyy = spectral_moment_fields(img)
    
    # Reference/template spectrum
    #Mhat = np.fft.fftshift(np.fft.fft2(med))
    Mhat = np.fft.fft2(med)
    m0 = Mhat[0,0].real
    Mhat = np.fft.fftshift(Mhat)
    
    # Centered frequency grids for fftshifted arrays
    ky, kx = centered_k(img.shape)
    kx0 = np.fft.ifftshift(kx) # Uncenter K for the visualization below
    ky0 = np.fft.ifftshift(ky)
    
    pm = np.zeros_like(params)  # momentum for Adam
    pv = np.zeros_like(params)   # velocity for Adam
    hm = np.zeros_like(params)
    hv = np.zeros_like(params)
    β1, β2 = 0.9, 0.999
    print(params)
    
    for epoch in range(epochs):
        L, J, H = affine_loss_grad_hess(
            params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky
        )
        
        # Whatever we're doing, its unhappy with the affine terms. I suspect that's where the problem lies. Lets force the gradient on their terms to always be zero.
        #J[4:] = 0.0
        #H[4:, :] = 0.0
        #H[:, 4:] = 0.0
        #np.fill_diagonal(H, 1.0)  # Identity Hessian on affine terms to prevent singularity
        # Lets generalize this for arbitrary coordinates we want to zero out.
        #ignore = [4,5,6,7]  # affine parameters
        ignore = [5,6] # off-diagonal affine parameters b,c which are near zero and causing instability
        J[ignore] = 0.0
        H[ignore, :] = 0.0
        H[:, ignore] = 0.0
        for i in ignore:
            H[i, i] = 1.0
        
        # Gauss newton step
        try:
            # Lavenberg-Marquardt style damping to improve stability: add a small multiple of the identity to the Hessian.
            H = H + np.outer(J,J)*(LMDamping if epoch == 0 else 0.0)
            
            #step = -0.5*np.linalg.solve(H, J)
            step = -np.linalg.solve(H, J)
        except np.linalg.LinAlgError:
            print("Hessian is singular, cannot take Newton step.")
            break
        params += step*rate
        
        # Lets try a gradient rescale based on the diagonal of H.
        #grad_rescale = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-30))
        #params -= rate * grad_rescale * J
        
        # It REALLY didn't like that. Lets try adam.
        #pm[:] = β1 * pm + (1 - β1) * J
        #pv[:] = β2 * pv + (1 - β2) * (J ** 2)
        #
        #hm[:] = pm / (1 - β1 ** (epoch + 1))
        #hv[:] = pv / (1 - β2 ** (epoch + 1))
        #
        #grad = hm / (np.sqrt(hv) + 1e-8)
        #params -= rate * grad
        
        # def eval_loss(params): # Defined above
        A_s, B_s, dy_s, dx_s, a_s, b_s, c_s, d_s = params
        Lambda = np.array([[a_s, b_s], [c_s, d_s]])
        Lambda_inv = np.linalg.inv(Lambda)
        rotated = hc.affine_resample(img, Lambda_inv)
        phasor = np.exp(-2j * np.pi * (ky0 * dy_s + kx0 * dx_s))
        result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A_s + B_s
        med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))
        loss = np.mean((result - med_thru_fltr)**2)
        
        cv2_imshow(cv2.resize(result - med_thru_fltr, (512, 512), interpolation=cv2.INTER_LANCZOS4),
            #title=f'[{framenum:04d}] loss={loss:.2e} @ epoch={epoch}')
            title=f'[{framenum:04d}] @ epoch={epoch}, loss={loss:.16e}')
    print(params)
    
    globals().update(locals())

# Copy that I've been working on to refine bugs.
if True:
    from scipy.ndimage import map_coordinates

    def centered_xy(shape):
        H, W = shape
        y = np.arange(H, dtype=np.float64) - H / 2.0
        x = np.arange(W, dtype=np.float64) - W / 2.0
        Y, X = np.meshgrid(y, x, indexing='ij')
        return Y, X

    def centered_k(shape):
        H, W = shape
        ky1 = np.fft.fftshift(np.fft.fftfreq(H))
        kx1 = np.fft.fftshift(np.fft.fftfreq(W))
        ky, kx = np.meshgrid(ky1, kx1, indexing='ij')
        return ky, kx

    def spectral_moment_fields(img):
        """
        Build the FFT fields needed by affine_loss_grad_hess.

        F0   = FFT(L)
        Fx   = FFT(x * L)
        Fy   = FFT(y * L)
        Fxx  = FFT(x^2 * L)
        Fxy  = FFT(x y * L)
        Fyy  = FFT(y^2 * L)

        Everything is fftshifted so interpolation happens on a centered frequency plane.

        In addition, we isolate the unshifted DC component f0 = F0[0,0].
        """
        img = np.asarray(img, dtype=np.float64)
        Y, X = centered_xy(img.shape)

        #F0  = np.fft.fftshift(np.fft.fft2(img))
        F0 = np.fft.fft2(img)
        f0 = F0[0,0].real

        F0  = np.fft.fftshift(F0)
        Fy  = np.fft.fftshift(np.fft.fft2(Y * img))
        Fx  = np.fft.fftshift(np.fft.fft2(X * img))
        Fyy = np.fft.fftshift(np.fft.fft2((Y * Y) * img))
        Fxy = np.fft.fftshift(np.fft.fft2((X * Y) * img))
        Fxx = np.fft.fftshift(np.fft.fft2((X * X) * img))
        return f0, F0, Fx, Fy, Fxx, Fxy, Fyy

    def interp_conj_cpu(F, qy, qx, order=1):
        """
        F   : complex array on fftshifted frequency grid
        qx,qy : frequency coordinates in fftfreq units, same shape as target grid
                expected range roughly [-0.5, 0.5)
        order : 1=bilinear, 3=cubic

        Returns conj(F)(qy,qx) sampled with periodic wrap.
        """
        H, W = F.shape

        # map fftfreq units to fftshifted pixel coordinates
        # qx = -0.5 maps near 0, qx = 0 maps near W/2, qx -> +0.5 maps near W
        y = qy * H + H / 2.0
        x = qx * W + W / 2.0
        
        coords = np.array([y, x])  # ndimage expects row,col
        #coords = np.array([x, y])  # ndimage expects row,col

        real = map_coordinates(F.real, coords, order=order, mode='wrap')
        imag = map_coordinates(F.imag, coords, order=order, mode='wrap')
        return real - 1j * imag   # conjugate after interpolation

    def affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1):
        ky, kx = centered_k(F0.shape)

        # q = Λᵀ·k  in (row, col) convention:  Λ = [[a,b],[c,d]]
        # Λᵀ = [[a,c],[b,d]],  k = [ky, kx]ᵀ
        qy = a * ky + c * kx    # row-frequency component
        qx = b * ky + d * kx    # col-frequency component

        Lc  = interp_conj_cpu(F0,  qy, qx, order=order)
        Lx  = interp_conj_cpu(Fx,  qy, qx, order=order)
        Ly  = interp_conj_cpu(Fy,  qy, qx, order=order)
        Lxx = interp_conj_cpu(Fxx, qy, qx, order=order)
        Lxy = interp_conj_cpu(Fxy, qy, qx, order=order)
        Lyy = interp_conj_cpu(Fyy, qy, qx, order=order)
        
        im = 2j*np.pi

        return Lc, im*Lx, im*Ly, im**2*Lxx, im**2*Lxy, im**2*Lyy, kx, ky

    def affine_loss_grad_hess_original(params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky):#, interp_conj):
        A, B, dy, dx, a, b, c, d = params
        det = a*d - b*c

        # q = Λᵀk in (row,col) convention:  Λ = [[a,b],[c,d]], k = [ky,kx]ᵀ
        # qy = a*ky + c*kx (row-freq),  qx = b*ky + d*kx (col-freq)
        #
        # Lx = ∂L̂†/∂q_col (from Fx = FFT(X·img), X = col coord)
        # Ly = ∂L̂†/∂q_row (from Fy = FFT(Y·img), Y = row coord)
        #
        # Chain rule for Λ elements (row,col convention):
        #   a → ∂q_row/∂a = ky  →  Ka uses ky·Ly
        #   b → ∂q_col/∂b = ky  →  Kb uses ky·Lx
        #   c → ∂q_row/∂c = kx  →  Kc uses kx·Ly
        #   d → ∂q_col/∂d = kx  →  Kd uses kx·Lx

        #affine_spectral_fields
        Lc, Lx, Ly, Lxx, Lxy, Lyy, kx, ky = affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1)

        ph = np.exp(-2j*np.pi*(kx*dx + ky*dy))
        Z  = ph * Lc * Mhat

        Dx = -2j*np.pi*kx
        Dy = -2j*np.pi*ky

        # Base moments
        K   = np.sum(Z).real
        Kx  = np.sum(Dx*Z).real
        Ky  = np.sum(Dy*Z).real
        Kxx = np.sum(Dx*Dx*Z).real
        Kxy = np.sum(Dx*Dy*Z).real
        Kyy = np.sum(Dy*Dy*Z).real

        # First affine moments  (row,col convention: a→ky·Ly, b→ky·Lx, c→kx·Ly, d→kx·Lx)
        Ka = np.sum(ph * (ky*Ly) * Mhat).real
        Kb = np.sum(ph * (ky*Lx) * Mhat).real
        Kc = np.sum(ph * (kx*Ly) * Mhat).real
        Kd = np.sum(ph * (kx*Lx) * Mhat).real

        # Translation-affine mixed moments
        Kxa = np.sum(Dx * ph * (ky*Ly) * Mhat).real
        Kxb = np.sum(Dx * ph * (ky*Lx) * Mhat).real
        Kxc = np.sum(Dx * ph * (kx*Ly) * Mhat).real
        Kxd = np.sum(Dx * ph * (kx*Lx) * Mhat).real

        Kya = np.sum(Dy * ph * (ky*Ly) * Mhat).real
        Kyb = np.sum(Dy * ph * (ky*Lx) * Mhat).real
        Kyc = np.sum(Dy * ph * (kx*Ly) * Mhat).real
        Kyd = np.sum(Dy * ph * (kx*Lx) * Mhat).real

        # Second affine moments  (same row,col pattern applied to L̂† second derivatives)
        Kaa = np.sum(ph * (ky*ky*Lyy) * Mhat).real
        Kab = np.sum(ph * (ky*ky*Lxy) * Mhat).real
        Kac = np.sum(ph * (ky*kx*Lyy) * Mhat).real
        Kad = np.sum(ph * (ky*kx*Lxy) * Mhat).real

        Kbb = np.sum(ph * (ky*ky*Lxx) * Mhat).real
        Kbc = np.sum(ph * (ky*kx*Lxy) * Mhat).real
        Kbd = np.sum(ph * (ky*kx*Lxx) * Mhat).real

        Kcc = np.sum(ph * (kx*kx*Lyy) * Mhat).real
        Kcd = np.sum(ph * (kx*kx*Lxy) * Mhat).real
        Kdd = np.sum(ph * (kx*kx*Lxx) * Mhat).real
        # These sure feel like an object from general relativity! I suppose since we started from a "distance" that it makes sense that the curvature of the loss landscape would be described by a metric-like object. Very cool.

        # Constants
        EL = np.sum(np.abs(F0)**2).real / F0.size
        #l0 = F0[0, 0].real
        #m0 = Mhat[0, 0].real
        l0 = f0
        Om = F0.size

        S0 = 0.5*A*A*EL + A*B*l0 - A*K
        SA = A*EL + B*l0 - K

        # Compute the reduced loss (not including the constant term from Mhat, which doesn't affect optimization)
        L = det * S0 + 0.5*Om*B*B - m0*B
        # Add the Mhat term back
        L = L + m0 

        # det first derivatives
        ja, jb, jc, jd = d, -c, -b, a

        # Gradient
        J = np.zeros(8, dtype=np.float64)
        J[0] = det * SA
        J[1] = Om*B + (A*det*l0 - m0)
        J[2] = -A*det*Ky   # dy
        J[3] = -A*det*Kx   # dx
        J[4] = ja*S0 - A*det*Ka
        J[5] = jb*S0 - A*det*Kb
        J[6] = jc*S0 - A*det*Kc
        J[7] = jd*S0 - A*det*Kd

        # Hessian
        H = np.zeros((8, 8), dtype=np.float64)

        # A/B block
        H[0,0] = det * EL
        H[0,1] = det * l0
        H[1,1] = Om

        # A / translation
        H[0,2] = -det * Ky
        H[0,3] = -det * Kx

        # translation / translation
        H[2,2] = -A * det * Kyy
        H[2,3] = -A * det * Kxy
        H[3,3] = -A * det * Kxx

        # A / affine
        H[0,4] = ja*SA - det*Ka
        H[0,5] = jb*SA - det*Kb
        H[0,6] = jc*SA - det*Kc
        H[0,7] = jd*SA - det*Kd

        # B / affine
        H[1,4] = A * ja * l0
        H[1,5] = A * jb * l0
        H[1,6] = A * jc * l0
        H[1,7] = A * jd * l0

        # translation / affine
        H[2,4] = -A * (ja*Ky + det*Kya)
        H[2,5] = -A * (jb*Ky + det*Kyb)
        H[2,6] = -A * (jc*Ky + det*Kyc)
        H[2,7] = -A * (jd*Ky + det*Kyd)

        H[3,4] = -A * (ja*Kx + det*Kxa)
        H[3,5] = -A * (jb*Kx + det*Kxb)
        H[3,6] = -A * (jc*Kx + det*Kxc)
        H[3,7] = -A * (jd*Kx + det*Kxd)

        # affine / affine second derivatives of det
        # only h_ad = h_da = +1, h_bc = h_cb = -1
        h = np.zeros((4,4), dtype=np.float64)
        h[0,3] = h[3,0] = 1.0   # a,d
        h[1,2] = h[2,1] = -1.0  # b,c

        j = np.array([ja, jb, jc, jd], dtype=np.float64)
        K1 = np.array([Ka, Kb, Kc, Kd], dtype=np.float64)
        K2 = np.array([
            [Kaa, Kab, Kac, Kad],
            [Kab, Kbb, Kbc, Kbd],
            [Kac, Kbc, Kcc, Kcd],
            [Kad, Kbd, Kcd, Kdd],
        ], dtype=np.float64)

        for r in range(4):
            for s in range(r, 4):
                #val = h[r,s]*S0 + j[r]*j[s]*S0 - A*det*(j[r]*K1[s] + j[s]*K1[r] + K2[r,s])
                val = h[r,s]*S0 - A*(j[r]*K1[s] + j[s]*K1[r] + det*K2[r,s])
                H[4+r, 4+s] = val
                #H[4+s, 4+r] = val

        # Symmetrize
        H = H + H.T - np.diag(np.diag(H))
        return L, J, H

    def affine_loss_grad_hess(params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky):#, interp_conj):
        A, B, dy, dx, a, b, c, d = params
        det = a*d - b*c
        Λ = np.array([
            [a, b],
            [c, d],
        ])
        iΛ = np.array([
            [d, -b],
            [-c, a],
        ]) / det
        N = F0.size

        # q = Λᵀk in (row,col) convention:  Λ = [[a,b],[c,d]], k = [ky,kx]ᵀ
        # qy = a*ky + c*kx (row-freq),  qx = b*ky + d*kx (col-freq)
        #
        # Lx = ∂L̂†/∂q_col (from Fx = FFT(X·img), X = col coord)
        # Ly = ∂L̂†/∂q_row (from Fy = FFT(Y·img), Y = row coord)
        #
        # Chain rule for Λ elements (row,col convention):
        #   a → ∂q_row/∂a = ky  →  Ka uses ky·Ly
        #   b → ∂q_col/∂b = ky  →  Kb uses ky·Lx
        #   c → ∂q_row/∂c = kx  →  Kc uses kx·Ly
        #   d → ∂q_col/∂d = kx  →  Kd uses kx·Lx

        #affine_spectral_fields
        Lc, Lx, Ly, Lxx, Lxy, Lyy, kx, ky = affine_spectral_fields(F0, Fx, Fy, Fxx, Fxy, Fyy, a, b, c, d, order=1)
        
        #im = 1j # why does python do this
        im = 2j*np.pi
        
        # Phase correlator arguments
        ph = np.exp(-im*(kx*dx + ky*dy))
        Z  = ph * Mhat
        
        C =  (Lc*Z).real.mean()/N
        Cy = ((-im*ky)*Lc*Z).real.mean()/N
        Cx = ((-im*kx)*Lc*Z).real.mean()/N
        
        Cy_y = ((-im*ky)*Ly*Z).real.mean()/N
        Cy_x = ((-im*ky)*Lx*Z).real.mean()/N
        Cx_y = ((-im*kx)*Ly*Z).real.mean()/N
        Cx_x = ((-im*kx)*Lx*Z).real.mean()/N
        
        Cyy = ((-im*ky)*(-im*ky)*Lc*Z).real.mean()/N
        Cyx = ((-im*ky)*(-im*kx)*Lc*Z).real.mean()/N # redundant?
        Cxy = ((-im*kx)*(-im*ky)*Lc*Z).real.mean()/N
        Cxx = ((-im*kx)*(-im*kx)*Lc*Z).real.mean()/N
        
        Cyy_y = ((-im*ky)*(-im*ky)*Ly*Z).real.mean()/N
        Cyy_x = ((-im*ky)*(-im*ky)*Lx*Z).real.mean()/N
        Cyx_y = ((-im*ky)*(-im*kx)*Ly*Z).real.mean()/N
        Cyx_x = ((-im*ky)*(-im*kx)*Lx*Z).real.mean()/N
        Cxy_y = ((-im*kx)*(-im*ky)*Ly*Z).real.mean()/N # redundant?
        Cxy_x = ((-im*kx)*(-im*ky)*Lx*Z).real.mean()/N # redundant?
        Cxx_y = ((-im*kx)*(-im*kx)*Ly*Z).real.mean()/N
        Cxx_x = ((-im*kx)*(-im*kx)*Lx*Z).real.mean()/N
        
        Cyy_yy = ((-im*ky)*(-im*ky)*Lyy*Z).real.mean()/N
        Cyy_yx = ((-im*ky)*(-im*ky)*Lxy*Z).real.mean()/N
        Cyy_xy = ((-im*ky)*(-im*ky)*Lxy*Z).real.mean()/N # redundant? # Cyy_xy = Cyy_yx
        Cyy_xx = ((-im*ky)*(-im*ky)*Lxx*Z).real.mean()/N
        Cyx_yy = ((-im*ky)*(-im*kx)*Lyy*Z).real.mean()/N
        Cyx_yx = ((-im*ky)*(-im*kx)*Lxy*Z).real.mean()/N
        Cyx_xy = ((-im*ky)*(-im*kx)*Lxy*Z).real.mean()/N # redundant? # Cyx_xy = Cyx_yx
        Cyx_xx = ((-im*ky)*(-im*kx)*Lxx*Z).real.mean()/N
        Cxy_yy = ((-im*kx)*(-im*ky)*Lyy*Z).real.mean()/N
        Cxy_yx = ((-im*kx)*(-im*ky)*Lxy*Z).real.mean()/N
        Cxy_xy = ((-im*kx)*(-im*ky)*Lxy*Z).real.mean()/N # redundant? # Cxy_xy = Cxy_yx
        Cxy_xx = ((-im*kx)*(-im*ky)*Lxx*Z).real.mean()/N
        Cxx_yy = ((-im*kx)*(-im*kx)*Lyy*Z).real.mean()/N
        Cxx_yx = ((-im*kx)*(-im*kx)*Lxy*Z).real.mean()/N
        Cxx_xy = ((-im*kx)*(-im*kx)*Lxy*Z).real.mean()/N # redundant? # Cxx_xy = Cxx_yx
        Cxx_xx = ((-im*kx)*(-im*kx)*Lxx*Z).real.mean()/N
        
        
        
        # Constants
        #Om = F0.size
        LL = (np.abs(F0)**2).real.mean()/N
        MM = (np.abs(Mhat)**2).real.mean()/N
        #l0 = F0[0, 0].real
        #m0 = Mhat[0, 0].real
        l0 = f0/N
        m0 = m0/N
        #Om = F0.size

        # Compute the Loss here
        L = 0.5*(det*A**2*LL + B**2 + MM + 2.0*B*(l0*A*det - m0) - 2.0*A*det*C)
        #globals().update(locals())
        
        # Compute the Gradient
        dA = det*(A*LL + B*l0 - C)
        dB = B + l0*A*det - m0
        dΔy = -A*det*Cy
        dΔx = -A*det*Cx
        dΛyy = iΛ.T[0,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cy_y
        dΛyx = iΛ.T[0,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cy_x
        dΛxy = iΛ.T[1,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cx_y
        dΛxx = iΛ.T[1,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Cx_x
        J = np.array([dA, dB, dΔy, dΔx, dΛyy, dΛyx, dΛxy, dΛxx], dtype=np.float64)
        
        # Compute the Hessian
        dAdA = det*LL
        dAdB = det*l0
        dAdΔy = -det*Cy
        dAdΔx = -det*Cx
        dAdΛyy = iΛ.T[0,0]*det*(A*LL + B*l0 - C) - det*Cy_y
        dAdΛyx = iΛ.T[0,1]*det*(A*LL + B*l0 - C) - det*Cy_x
        dAdΛxy = iΛ.T[1,0]*det*(A*LL + B*l0 - C) - det*Cx_y
        dAdΛxx = iΛ.T[1,1]*det*(A*LL + B*l0 - C) - det*Cx_x
        
        dBdB = 1.0
        dBdΔy = 0.0
        dBdΔx = 0.0
        dBdΛyy = iΛ.T[0,0]*det*l0
        dBdΛyx = iΛ.T[0,1]*det*l0
        dBdΛxy = iΛ.T[1,0]*det*l0
        dBdΛxx = iΛ.T[1,1]*det*l0
        
        dΔydΔy = -A*det*Cyy
        dΔydΔx = -A*det*Cxy
        dΔydΛyy = -A*iΛ.T[0,0]*det*Cy - A*det*Cyy_y
        dΔydΛyx = -A*iΛ.T[0,1]*det*Cy - A*det*Cyy_x
        dΔydΛxy = -A*iΛ.T[1,0]*det*Cy - A*det*Cyx_y
        dΔydΛxx = -A*iΛ.T[1,1]*det*Cy - A*det*Cyx_x
        
        dΔxdΔx = -A*det*Cxx
        dΔxdΛyy = -A*iΛ.T[0,0]*det*Cx - A*det*Cxy_y
        dΔxdΛyx = -A*iΛ.T[0,1]*det*Cx - A*det*Cxy_x
        dΔxdΛxy = -A*iΛ.T[1,0]*det*Cx - A*det*Cxx_y
        dΔxdΛxx = -A*iΛ.T[1,1]*det*Cx - A*det*Cxx_x
        
        #dΛijdΛyz = det*((iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(A**2*LL/2 + A*B*l0
        # - A*det*(
        # (iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*C
        # + iΛ[j,y]*C_yz
        # + iΛ[z,y]*C_ij
        # + C_yijz
        # )
        
        Ci_j = np.array([
            [Cyy, Cxy],
            [Cyx, Cxx],
        ], dtype=np.float64)
        
        Cij_yz = np.array([
            Cyy_yy, Cyy_yx, Cyy_xy, Cyy_xx,
            Cyx_yy, Cyx_yx, Cyx_xy, Cyx_xx,
            Cxy_yy, Cxy_yx, Cxy_xy, Cxy_xx,
            Cxx_yy, Cxx_yx, Cxx_xy, Cxx_xx,
        ], dtype=np.float64).reshape((2,2,2,2))
        
        dΛijdΛyz = np.zeros((2,2,2,2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for y in range(2):
                    for z in range(2):
                        dΛijdΛyz[i,j,y,z] = det*(iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(0.5*A**2*LL + A*B*l0 - A*C)
                        dΛijdΛyz[i,j,y,z] -= A*det*(iΛ[z,y]*Ci_j[i,j] + iΛ[j,i]*Ci_j[y,z] + Cij_yz[y,i,j,z])
                        
        dΛyydΛyy = dΛijdΛyz[0,0,0,0]
        dΛyydΛyx = dΛijdΛyz[0,0,0,1]
        dΛyydΛxy = dΛijdΛyz[0,0,1,0]
        dΛyydΛxx = dΛijdΛyz[0,0,1,1]
        dΛyxdΛyy = dΛijdΛyz[0,1,0,0]
        dΛyxdΛyx = dΛijdΛyz[0,1,0,1]
        dΛyxdΛxy = dΛijdΛyz[0,1,1,0]
        dΛyxdΛxx = dΛijdΛyz[0,1,1,1]
        dΛxydΛyy = dΛijdΛyz[1,0,0,0]
        dΛxydΛyx = dΛijdΛyz[1,0,0,1]
        dΛxydΛxy = dΛijdΛyz[1,0,1,0]
        dΛxydΛxx = dΛijdΛyz[1,0,1,1]
        dΛxxdΛyy = dΛijdΛyz[1,1,0,0]
        dΛxxdΛyx = dΛijdΛyz[1,1,0,1]
        dΛxxdΛxy = dΛijdΛyz[1,1,1,0]
        dΛxxdΛxx = dΛijdΛyz[1,1,1,1]
        
        H = np.array([
            [dAdA, dAdB, dAdΔy, dAdΔx, dAdΛyy, dAdΛyx, dAdΛxy, dAdΛxx],
            [dAdB, dBdB, dBdΔy, dBdΔx, dBdΛyy, dBdΛyx, dBdΛxy, dBdΛxx],
            [dAdΔy, dBdΔy, dΔydΔy, dΔydΔx, dΔydΛyy, dΔydΛyx, dΔydΛxy, dΔydΛxx],
            [dAdΔx, dBdΔx, dΔydΔx, dΔxdΔx, dΔxdΛyy, dΔxdΛyx, dΔxdΛxy, dΔxdΛxx],
            [dAdΛyy, dBdΛyy, dΔydΛyy, dΔxdΛyy, dΛyydΛyy, dΛyydΛyx, dΛyydΛxy, dΛyydΛxx],
            [dAdΛyx, dBdΛyx, dΔydΛyx, dΔxdΛyx, dΛyxdΛyy, dΛyxdΛyx, dΛyxdΛxy, dΛyxdΛxx],
            [dAdΛxy, dBdΛxy, dΔydΛxy, dΔxdΛxy, dΛxydΛyy, dΛxydΛyx, dΛxydΛxy, dΛxydΛxx],
            [dAdΛxx, dBdΛxx, dΔydΛxx, dΔxdΛxx, dΛxxdΛyy, dΛxxdΛyx, dΛxxdΛxy, dΛxxdΛxx],
        ])
        
        return L, J, H

    def validate_affine_spectral_fields_and_grad_hess(framenum=75):
        img = np.asarray(sector_data[framenum], dtype=np.float64)
        med = np.asarray(sector_median, dtype=np.float64)

        # Moving-image spectral moment fields
        f0, F0, Fx, Fy, Fxx, Fxy, Fyy = spectral_moment_fields(img)

        # Reference/template spectrum
        #Mhat = np.fft.fftshift(np.fft.fft2(med))
        Mhat = np.fft.fft2(med)
        m0 = Mhat[0,0].real
        Mhat = np.fft.fftshift(Mhat)

        # Centered frequency grids for fftshifted arrays
        ky, kx = centered_k(img.shape)

        # Gather different points to validate that the gradient is what we expect
        params = np.copy(delta_coords_affine[framenum])
        #params = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)

        L, J, H = affine_loss_grad_hess(
            params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky
        )

        Ln = np.zeros_like(J)
        ε0, ε1, ε2 = 1e-6, 1e-8, 1e-8
        for i,ε in enumerate([ε0, ε0, ε1, ε1, ε2, ε2, ε2, ε2]):
            dp = np.zeros_like(params)
            dp[i] = ε
            pn = params + dp
            Lp, J_, H_ = affine_loss_grad_hess(pn, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky)
            
            dp[i] = -ε
            pn = params + dp
            Lm, J_, H_ = affine_loss_grad_hess(pn, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky)
            #Ln[i] = (Lp - L) / ε
            Ln[i] = (Lp - Lm) / (2*ε)
        globals().update(locals())
        
        # Compute the losses non spectrally
        A, B, dy, dx, a, b, c, d = params
        Λ = np.array([
            [a, b],
            [c, d]
        ], dtype=np.float64)
        iΛ = np.linalg.inv(Λ)
        rotated = hc.affine_resample(img, iΛ)
        phasor = np.exp(-2j * np.pi * (ky * dy + kx * dx))
        result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A + B
        #med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))
        med_thru_fltr = med.copy() 
        loss = np.mean((result - med_thru_fltr)**2)/2
        
        rel_error = (np.float64(L)-np.float64(loss))/np.float64(L)
        print(f"Non-spectral loss: {loss:.6f}, Spectral loss: {L:.6f}")
        print(f"Relative Error: {rel_error:e}, Ratio: {L/loss:.6f}")
        
        
        
        
    validate_affine_spectral_fields_and_grad_hess()
    # Lets get numpy to print as wide as possible instead of doing "word wrap"
    with np.printoptions(linewidth=2000):
        print(f"      {Ln = }")
        print(f"       {J = }")
        print(f"{(Ln-J)/J = }")
        

def iteratively_cull_samples(frame, num_samples=1, mult=16):
    frame0 = frame.copy()
    frame = frame.copy()
    
    ybound = (0, frame.shape[0])
    xbound = (0, frame.shape[1])
    
    # Higher resolution coordinates for subpixel sampling
    yaxis = np.linspace(0, frame.shape[0]-1, frame.shape[0]*mult, endpoint=False)
    xaxis = np.linspace(0, frame.shape[1]-1, frame.shape[1]*mult, endpoint=False)
    Y,X = np.meshgrid(yaxis, xaxis, indexing='ij')
    
    sample_pts = []
    for n in range(num_samples):
        bf = cv2.resize(frame, (frame.shape[1]*mult, frame.shape[0]*mult), interpolation=cv2.INTER_LANCZOS4)
        bfmax = np.unravel_index(np.argmax(bf), bf.shape)
        
        # Build a local quadratic fit from the 16x16 neighborhood
        # Lets use the max min max pattern to prevent spilling from the edges of the neighborhood. 
        y0 = max(0, bfmax[0]-mult//2)
        y1 = min(bf.shape[0], y0+mult)
        y0 = max(0, y1-mult)
        
        x0 = max(0, bfmax[1]-mult//2)
        x1 = min(bf.shape[1], x0+mult)
        x0 = max(0, x1-mult)
        
        # Fit a quadratic to the neighborhood
        nbh_f = bf[y0:y1, x0:x1].ravel()
        nbh_y = Y[y0:y1, x0:x1].ravel()
        nbh_x = X[y0:y1, x0:x1].ravel()
        nbh_u = np.ones_like(nbh_x)
        XX = np.array([nbh_u, nbh_y, nbh_x, nbh_y*nbh_y, nbh_y*nbh_x, nbh_x*nbh_x]).T
        args = np.linalg.lstsq(XX, nbh_f, rcond=None)[0]
        a,b,c,d,e,f = args
        # a + b*y + c*x + d*y² + e*yx + f*x²
        
        # iterate a local quadratic fit via gauss newton of the local gradient and hessian
        params = np.array([Y[bfmax], X[bfmax]], dtype=np.float64)
        #print(f"Iter 0: params = {params}")
        for k in range(3):
            #grad = np.array([b + 2*d*nbh_y + e*nbh_x, c + e*nbh_y + 2*f*nbh_x])
            grad = np.array([b + 2*d*params[0] + e*params[1], c + e*params[0] + 2*f*params[1]])
            hess = np.array([[2*d, e], [e, 2*f]])
            try:
                delta = np.linalg.solve(hess, grad)
                params -= delta
                #print(f"Iter {k+1}: params = {params}, grad = {grad}, delta = {delta}")
            except np.linalg.LinAlgError:
                break
        sample_pts.append(params)
        
        # Zero out a neighborhood around the found point to prevent resampling it
        # We used a 1x1 pixel domain to find this, lets zero out a 5x5 pixel domain to be safe.
        y0 = max(0, int(params[0])-2)
        y1 = min(bf.shape[0], y0+mult)
        y0 = max(0, y1-mult)
        
        x0 = max(0, int(params[1])-2)
        x1 = min(bf.shape[1], x0+mult)
        x0 = max(0, x1-mult)
        
        frame[y0:y1, x0:x1] = 0.0
        cv2_imshow(cv2.resize(frame, (2048,2048), interpolation=cv2.INTER_LANCZOS4))
        
    sample_pts = np.array(sample_pts)
    fig, ax = plt.subplots()
    ax.imshow(slog(frame0-frame0.min()), interpolation='lanczos')
    ax.scatter(*sample_pts.T[::-1], color='red')
    plt.show()
    
    return sample_pts

# Lets commit all of this to a new repo.
# Description:
# If I write "Oelkers et al's" is it better than "Oelkers et. al.'s"?
# Copilot: The correct form is "Oelkers et al.'s". The "et al." is a Latin abbreviation for "et alia", meaning "and others". When using it in possessive form, you should add the apostrophe and "s" after "al." to indicate possession. So, it should be "Oelkers et al.'s".
# Refactor of Oelkers et. al.'s DIA project with computer graphics optimizations & and a focus on maintainability. https://github.com/ryanoelkers/DIA
# me: How does that description sound? What would you do to improve it?
# copilot: The description is clear and concise, but it could be improved by providing more specific details about the optimizations and maintainability improvements you plan to implement. Here's a revised version:
# "Refactor of Oelkers et al.'s DIA project with a focus on computer graphics optimizations, including efficient algorithms for image processing and data handling, as well as a commitment to maintainable code structure and documentation. This project aims to enhance the performance and usability of the original DIA implementation while ensuring that the codebase is easy to understand and extend for future development."

# I've made some changes to a new repo. I'd like to overwrite the history of the repo's initial commit with the subtle changes. What command do I need? 
# Here's my state:
# PS C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\DIA_GPU> git status
# On branch main
# Your branch is up to date with 'origin/main'.

# Changes not staged for commit:
#   (use "git add <file>..." to update what will be committed)
#   (use "git restore <file>..." to discard changes in working directory)
#         modified:   LICENSE
#         modified:   README.md

# no changes added to commit (use "git add" and/or "git commit -a")
# PS C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\DIA_GPU> git log -3
# commit 458abe207c4896b00a0cc9b09b862e2792004137 (HEAD -> main, origin/main, origin/HEAD)
# Author: Copperbotte <copperbotte@gmail.com>
# Date:   Fri Apr 17 17:52:42 2026 -0700

#     Initial commit
# PS C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\DIA_GPU>
# 
# Copilot: To overwrite the history of the repo's initial commit with your changes, you can use the following commands:
# 1. Stage the changes you want to include in the new commit:
#    git add LICENSE README.md
# 2. Commit the changes with a new message:
#    git commit --amend -m "Updated LICENSE and README.md with new information"



def gather_all_samples(sector_data):
    all_samples = []
    for idx,sd in enumerate(sector_data):
        all_samples.append(iteratively_cull_samples_fourier(slog(sd-sd.min()), num_samples=100, rad=8))
        idx_ = idx+1
        if idx_ % 100 == 0:
            print(idx_)
    all_samples = np.array(all_samples)
    
    # Sort by spatial hash
    # range of 0-255, lets hash by rescaling to 4096 and indexing coordinates.
    # def hash_coord(coord, hrad=256):
    #     y, x = coord
    #     y = int(y/sector_data[0].shape[0]*hrad)
    #     x = int(x/sector_data[0].shape[1]*hrad)
    #     return (y, x)
    
    # Hash by voronoi cell of the first set of samples.
    s0 = all_samples[0]
    def hash_coord(coord, samples=s0):
        y, x = coord
        dists = np.linalg.norm(samples - coord, axis=1)
        idx = np.argmin(dists)
        return idx
    
    d = dict()
    order = []
    for frame_samples in all_samples:
        for sample in frame_samples:
            h = hash_coord(sample)
            if h not in d:
                d[h] = []
                order.append(h)
            d[h].append(sample)

    # Cull all samples that don't persist through the entire dataset.
    order = [h for h in order if len(d[h]) == len(all_samples)]
    all_samples = np.array([d[h] for h in order])
    
    fig, ax = plt.subplots()
    #for h in order:
    #    shush = ax.plot(*d[h].T[::-1], '.', label=str(h))
    for frame_samples in all_samples:
        shush = ax.plot(*frame_samples.T[::-1], '.', label=str(frame_samples))
    plt.show()
    
    # Best fit an affine transform between each set of samples and the first set of samples
    affines = []
    for n in range(len(all_samples)):
        src = all_samples[0]
        dst = all_samples[n]
        # src = Λ(dst + Δ) + ε, where Λ is a 2x2 matrix, Δ is a 2D translation, and ε is noise.
        # ℒ = (1/2)Σₙ(Λⁱʲ(dₙʲ+Δʲ) - sₙⁱ)²/N = (1/2)Σₙ(Rₙⁱ)²/N
        # ∂ℒ/∂Λⁱʲ = (1/N)Σₙ(dₙʲ+Δʲ)(Rₙⁱ)
        # ∂ℒ/∂Δʲ = (1/N)ΣₙΛⁱʲ(Rₙⁱ)
        # ∂ℒ/∂Λⁱʲ = 0ⁱʲ = (1/N)Σₙ(dₙʲ+Δʲ)(Rₙⁱ)
        # ∂ℒ/∂Δʲ = 0ʲ = (1/N)ΣₙΛⁱʲ(Rₙⁱ)
        # 0ⁱʲ = Σₙ(dₙʲ+Δʲ)(Λⁱᵏ(dₙᵏ+Δᵏ) - sₙⁱ)
        # 0ʲ = ΣₙΛⁱʲ(Λⁱʲ(dₙʲ+Δʲ) - sₙⁱ)
        # Σₙ(dₙʲ+Δʲ)sₙⁱ = Σₙ(dₙʲ+Δʲ)(Λⁱʲ(dₙᵏ+Δᵏ))
        # Σₙ(Λⁱʲ sₙⁱ) = Σₙ(dₙʲ+Δʲ)(Λⁱʲ(dₙʲ+Δʲ))
        
        
        # ℒ = (1/2)Σₙ(Λⁱʲ(dₙʲ+Δʲ) - sₙⁱ)²/N = (1/2)Σₙ(Rₙⁱ)²/N
        # ∂ℒ = ΣₙRₙⁱ(∂Λⁱʲ(dₙʲ+Δʲ) + Λⁱʲ∂Δʲ)
        
        params = np.array([0.0,0.0,  1.0,0.0,  0.0,1.0], dtype=np.float64) # Δy, Δx, a, b, c, d
        # dst: (N, 2), src: (N, 2)
        # params: [Δy, Δx, a, b, c, d]
        mask = np.ones(src.shape[0], dtype=bool)
        for epoch in range(5):
            Δ = params[:2]
            Λ = params[2:].reshape(2, 2)
            
            # 1. Compute Residuals (N, 2)
            diff = dst + Δ # (N, 2)
            res = (diff @ Λ.T) - src 
            var = np.einsum('ni,ni->n', res, res)
            
            # Sigma mask the output residual to suppress huge outliers
            # cnum, clow, chigh, cstd = sigma_clip_mask_numba(sc_cimg, sc_mask, 2.5, 2.5)
            # cimg1 = sc_cimg[sc_mask]  # apply the mask to get the clipped values
            sigma_clip_mask_numba(var, mask, 2.5, 2.5)
            #print(params, np.linalg.norm(res))
            res[~mask] = 0.0
            
            # 2. Build Jacobian (N, 2, 6)
            N = dst.shape[0]
            J = np.zeros((N, 2, 6))
            
            # Derivatives w.r.t translation (Δy, Δx)
            J[:, :, 0] = Λ[:, 0] # ∂R/∂Δy
            J[:, :, 1] = Λ[:, 1] # ∂R/∂Δx
            
            # Derivatives w.r.t Λ (a, b, c, d)
            J[:, 0, 2] = diff[:, 0] # ∂Ry/∂a
            J[:, 0, 3] = diff[:, 1] # ∂Ry/∂b
            J[:, 1, 4] = diff[:, 0] # ∂Rx/∂c
            J[:, 1, 5] = diff[:, 1] # ∂Rx/∂d
            
            # 3. Gradient (6,)
            # ∇ℒ = (1/N) * Σ (J_n.T @ res_n)
            grad = np.einsum('nij,ni->j', J, res) / N
            
            # J has shape (N, 2, 6). We need to flatten the observation dimensions.
            J_flat = J.reshape(-1, 6)  # Shape (2N, 6)
            res_flat = res.reshape(-1) # Shape (2N,)
            
            # Compute Approximate Hessian (6x6) and Gradient (6,)
            H = J_flat.T @ J_flat
            g = J_flat.T @ res_flat
            
            # Solve the linear system
            try:
                step = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                # Add a little damping (Levenberg-Marquardt style) if H is singular
                step = np.linalg.solve(H + 1e-6 * np.eye(6), -g)
            
            # Update parameters
            params += step
        affines.append(params)
    
    affines = np.array(affines)
    # Apply the transforms and plot
    all_samples_transformed = []
    for n in range(len(all_samples)):
        params = affines[n]
        Δ = params[:2]
        Λ = params[2:].reshape(2, 2)
        transformed = (all_samples[n] + Δ) @ Λ.T
        all_samples_transformed.append(transformed)
    
    # all_samples.shape (83, 1146, 2)
    fig, ax = plt.subplots()
    for sample in all_samples_transformed:
        ax.plot(*sample.T[::-1], '.')
    plt.show()
    
        
    globals().update(locals())
    
    #return d


# Matched filter. Fits two additional variables to the data along with the dxy coordinate as above:
# Residual = L1(x+Δx)*A + B - L0(x).
# Given a FIXED L0 background, we phase correlate AND fit A,B simultaneously. This should be the optimal linear filter for this problem, and should give us the best possible SNR for detecting changes. The only question is whether it will be stable enough to converge, and whether it will be fast enough to run in a reasonable time.
# Let 𝕃₁ = ∫|L₁|²dk, 𝕃₀ = ∫|L₀|²dk, z(k, Δx) = exp(-𝒾k•Δx)L₁†(k) L₀(k), so
# ℱ = -Re{∫z dk}
# ℱₓ = -Re{∫(-𝒾 kₓ) z dk}
# ℱᵧ = -Re{∫(-𝒾 kᵧ) z dk}
# ℱₓₓ = -Re{∫-kₓ kₓ z dk}
# ℱₓᵧ = -Re{∫-kₓ kᵧ z dk}
# ℱᵧᵧ = -Re{∫-kᵧ kᵧ z dk} 
# Or in general, ℱⁱ = -Re{∫-kⁱ z dk}, and ℱⁱʲ = -Re{∫-kⁱ kʲ z dk}
# And finally our loss:
# ℒ = (1/2)∫(img₁(x+Δx)•A + B - img₀(x))² dx
# This can be rearranged to a spectral form:
# ℒ = (1/2)A²𝕃₁ + (1/2)B² + (1/2)𝕃₀ + B(A L₁(k=0) - L₀(k=0)) - Aℱ
# Lets omit the k=0 for L₁ and L₀ for brevity. They'll never be used except for within an fft, or as their DC term.
# 
# It has a familiar Jacobian:
# ∂ℒ/∂A = A𝕃₁ + B L₁ - ℱ
# ∂ℒ/∂B = B + A L₁ - L₀
# ∂ℒ/∂Δxⁱ = A ℱⁱ
# 
# The Hessian is constructed similarly:
# ∂²ℒ/∂A² = 𝕃₁, ∂²ℒ/∂A∂B = L₁, ∂²ℒ/∂A∂Δxʲ = ℱʲ
# ∂²ℒ/∂B∂A = L₁, ∂²ℒ/∂B² = 1, ∂²ℒ/∂B∂Δxʲ = 0ʲ
# ∂²ℒ/∂Δxⁱ∂A = ℱⁱ, ∂²ℒ/∂Δxⁱ∂B = 0ⁱ, ∂²ℒ/∂Δxⁱ∂Δxʲ = A ℱⁱʲ
# Which I REALLY hope is PSD around the solution!!
# 
# From mathematica:
# H = {
#   {𝕃₁, L₁, -ℱy, -ℱx},
#   {L₁, 1, 0, 0},
#   {-ℱy, 0, -ℱyy, -ℱxy},
#   {-ℱx, 0, -ℱxy, -ℱxx}
# }
# (Inverse[M]*(-2 ℱx ℱxy ℱy + ℱxx ℱy² + ℱx² ℱyy + (ℱxy² - ℱxx ℱyy) (L₁² - 𝕃₁))) // Simplify
# 
# {
#  {-ℱxy² + ℱxx ℱyy, (ℱxy² - ℱxx ℱyy) L₁, ℱx ℱxy - ℱxx ℱy, ℱxy ℱy - ℱx ℱyy},
#  {(ℱxy² - ℱxx ℱyy) L₁, -2 ℱx ℱxy ℱy + ℱxx ℱy² + ℱx² ℱyy + (-ℱxy² + ℱxx ℱyy) 𝕃₁, (-ℱx ℱxy + ℱxx ℱy) L₁, (-ℱxy ℱy + ℱx ℱyy) L₁},
#  {ℱx ℱxy - ℱxx ℱy, (-ℱx ℱxy + ℱxx ℱy) L₁, -ℱx² + ℱxx L₁² -  ℱxx 𝕃₁, ℱx ℱy - ℱxy L₁² +  ℱxy 𝕃₁},
#  {ℱxy ℱy - ℱx ℱyy, (-ℱxy ℱy + ℱx ℱyy) L₁, ℱx ℱy - ℱxy L₁² + ℱxy 𝕃₁, -ℱy² + ℱyy L₁² - ℱyy 𝕃₁}
#  }
# That seems sufficient.
#
def matched_filter_phase_corr_iterator(img0, img1, init=None, tol=None, spectral_blur=None, iters=100):
    fft0 = np.fft.fft2(img0)
    fft1 = np.fft.fft2(img1)
    z = fft0*fft1.conj()
    zph = np.nan_to_num(z / np.abs(z), nan=0.0, posinf=0.0, neginf=0.0)
    ky,kx = np.meshgrid(np.fft.fftfreq(img0.shape[0]), np.fft.fftfreq(img0.shape[1]), indexing='ij')
    tau = np.pi*2
    itau = tau*1j

    # Spectral weight W(k) — e.g. Gaussian for blur-based phase locking.
    # When W is present, the loss becomes L = (1/2N) sum_k W(k)|...|^2,
    # so every spectral sum gets W(k) multiplied in.
    # A Gaussian W(k) = exp(-0.5*(kr/sigma_k)^2) is equivalent to
    # blurring both images by sigma_pix = 1/(2*pi*sigma_k) before fitting.
    if spectral_blur is not None:
        #W = spectral_weight
        #10 pixel blur -> sigma_k = 1/(2*pi*10) ~ 0.016, so W(k) = exp(-0.5*(k/0.016)^2)
        sigma = 1/(2*np.pi*spectral_blur)#32.0)
        W = np.exp(-0.5*(np.sqrt(kx**2 + ky**2)/sigma)**2)
    else:
        W = 1.0  # no-op: scalar 1 broadcasts, zero cost
    z_w = z * W       # weighted cross-spectrum for F, Fx, Fxx, ...
    zph_w = zph * W   # weighted whitened cross-spectrum for seed

    if init is None:
        # Seed the phase correlator as usual (unweighted for argmax — want the sharp peak).
        dyx = np.array(np.unravel_index(np.argmax(np.fft.ifft2(zph).real), img0.shape), np.float64)

        # Warm start shift with one weighted Newton step.
        phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1])) * zph_w
        F = (phasor).sum().real
        Fx = (-itau*kx*phasor).sum().real
        Fy = (-itau*ky*phasor).sum().real
        Fxx = (-tau**2*kx*kx*phasor).sum().real
        Fxy = (-tau**2*kx*ky*phasor).sum().real
        Fyy = (-tau**2*ky*ky*phasor).sum().real
        Hdet = Fxx*Fyy - Fxy*Fxy
        Hinv = np.array([[Fxx,-Fxy],[-Fxy,Fyy]])/Hdet
        step = Hinv @ np.array([Fy, Fx])
        dyx -= step

        # Resample image A:
        phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1]))
        shifted = np.fft.ifft2(fft1*phasor).real
        # Best fit img0 ~ A*shifted + B!
        A_matrix = np.column_stack([shifted.ravel(), np.ones_like(shifted.ravel())])
        sol, residuals, rank, s = np.linalg.lstsq(A_matrix, img0.ravel(), rcond=None)
        A, B = sol
    else:
        # Extract A,B,dyx from init, assuming it has the right form. Lets skip checks for performance.
        A, B = init[:2]
        dyx = init[2:]

    args = np.array([A, B, dyx[0], dyx[1]], dtype=np.float64) # A, B, Δy, Δx

    N_pix = img0.shape[0] * img0.shape[1]
    # Weighted norms: W enters every spectral sum.
    # LL1 = (1/N) sum_k W(k)|F1(k)|^2,  L1 = W(0)*F1(0,0),  L0 = W(0)*F0(0,0)
    W0 = W[0,0] if hasattr(W, '__getitem__') else W  # W at DC
    W_sum = W.sum().real if hasattr(W, 'sum') else N_pix * W  # sum_k W(k) = weighted N_pix
    LL1 = (W * np.abs(fft1)**2).sum().real / N_pix
    L0 = (fft0[0,0] * W0).real
    L1 = (fft1[0,0] * W0).real
    # The DC terms should be purely real, as our inputs are guaranteed to be purely real.
    # I hope they don't have to be absolute valued.

    #for n in range(10):
    step = np.inf*np.ones_like(args)
    n = -1
    while True:
        n += 1 # Start with -1 so the first iteration is 0.
        # This lets us dynamically select a format depending on if tol is present.
        if tol is None:
            if iters < n:
                break
        else:
            if (iters < n) or (np.linalg.norm(step) < tol):
                break

        A, B, dy, dx = args

        phasor = np.exp(-itau*(ky*dy + kx*dx)) * z_w  # weighted cross-spectrum
        F = (phasor).sum().real / N_pix          # sum_k W(k) s0·s1(Δ)
        Fx = (-itau*kx*phasor).sum().real / N_pix
        Fy = (-itau*ky*phasor).sum().real / N_pix
        Fxx = (-tau**2*kx*kx*phasor).sum().real / N_pix
        Fxy = (-tau**2*kx*ky*phasor).sum().real / N_pix
        Fyy = (-tau**2*ky*ky*phasor).sum().real / N_pix

        # Assemble the Jacobian
        Ja = A*LL1 + B*L1 - F
        Jb = W_sum*B + A*L1 - L0
        Jy = -A*Fy
        Jx = -A*Fx
        J = np.array([Ja, Jb, Jy, Jx])

        # Assemble the Hessian
        Haa, Hab, Hay, Hax = LL1, L1, -Fy, -Fx
        Hbb, Hby, Hbx = W_sum, 0.0, 0.0
        Hyy, Hxy = -A*Fyy, -A*Fxy
        Hxx = -A*Fxx

        H = np.array([
            [Haa, Hab, Hay, Hax],
            [Hab, Hbb, Hby, Hbx],
            [Hay, Hby, Hyy, Hxy],
            [Hax, Hbx, Hxy, Hxx]
        ])

        # Assemble useful intermediate structures
        xyxy_xx_yy = Fxy**2 - Fxx*Fyy # ℱxy² - ℱxx ℱyy  | -(ℱxx ℱyy - ℱxy²)  | ℱxy² - ℱxx ℱyy   | -ℱxx ℱyy  + ℱxy²
        x_yy__xy_y = Fx*Fyy - Fxy*Fy  # ℱx ℱyy - ℱxy ℱy | -(ℱxy ℱy - ℱx ℱyy) | ℱx ℱyy - ℱxy ℱy | -ℱxy ℱy + ℱx ℱyy
        xx_y__x_xy = Fxx*Fy - Fxy*Fx  # ℱxx ℱy - ℱx ℱxy | -(ℱx ℱxy - ℱxx ℱy) | ℱxx ℱy - ℱx ℱxy | -ℱx ℱxy + ℱxx ℱy
        # Hey! It sure looks like a GR construct. Neat!
        # I suppose the Hessian is the same form as a metric tensor. I think.


        # Assemble the determinant
        # (Inverse[M]*(-2 ℱx ℱxy ℱy + ℱxx ℱy² + ℱx² ℱyy + (ℱxy² - ℱxx ℱyy) (L₁² - 𝕃₁))) // Simplify
        #det = -2*x_xy*y_xy + Fxx*Fy**2 + Fx**2*Fyy + xy_xxyy*(L1**2 - LL1)
        # (-2 ℱx ℱxy ℱy + ℱxx ℱy² + ℱx² ℱyy + (ℱxy² - ℱxx ℱyy) (L₁² - 𝕃₁))) // Simplify
        # (-2 ℱx ℱxy ℱy + (ℱxx ℱy) ℱy + ℱx (ℱx ℱyy) + (ℱxy² - ℱxx ℱyy) (L₁² - 𝕃₁))) // Simplify
        # (ℱx ℱyy - ℱxy ℱy) ℱx + (ℱxx ℱy - ℱx ℱxy) ℱy + (ℱxy² - ℱxx ℱyy) (L₁² - 𝕃₁)) // Simplify
        det = x_yy__xy_y*Fx + xx_y__x_xy*Fy + xyxy_xx_yy*(L1**2 - LL1)

# Lets leave this comment here for reference, and edit the one below.
# {
#  {-ℱxy² + ℱxx ℱyy, (ℱxy² - ℱxx ℱyy) L₁, ℱx ℱxy - ℱxx ℱy, ℱxy ℱy - ℱx ℱyy},
#  {(ℱxy² - ℱxx ℱyy) L₁, -2 ℱx ℱxy ℱy + ℱxx ℱy² + ℱx² ℱyy + (-ℱxy² + ℱxx ℱyy) 𝕃₁, (-ℱx ℱxy + ℱxx ℱy) L₁, (-ℱxy ℱy + ℱx ℱyy) L₁},
#  {ℱx ℱxy - ℱxx ℱy, (-ℱx ℱxy + ℱxx ℱy) L₁, -ℱx² + ℱxx L₁² -  ℱxx 𝕃₁, ℱx ℱy - ℱxy L₁² +  ℱxy 𝕃₁},
#  {ℱxy ℱy - ℱx ℱyy, (-ℱxy ℱy + ℱx ℱyy) L₁, ℱx ℱy - ℱxy L₁² + ℱxy 𝕃₁, -ℱy² + ℱyy L₁² - ℱyy 𝕃₁}
#  }

        # Assemble the inverse Hessian
        # NOTE: This was derived in Mathematica with H_BB=1. Now that H_BB=N_pix,
        # this analytic inverse is STALE. Using np.linalg.inv(Hadj) below instead.
        # {
        #  {-xyxy_xx_yy, xyxy_xx_yy L₁, -xx_y__x_xy, -x_yy__xy_y},
        #  {xyxy_xx_yy L₁, -(x_yy__xy_y*Fx + xx_y__x_xy*Fy) + (-xyxy_xx_yy) 𝕃₁, (xx_y__x_xy) L₁, (x_yy__xy_y) L₁},
        #  {-xx_y__x_xy, (-xx_y__x_xy) L₁, -Fx**2 + Fxx L₁**2 -  Fxx 𝕃₁, Fx Fy - Fxy L₁**2 +  Fxy 𝕃₁},
        #  {-x_yy__xy_y, (-x_yy__xy_y) L₁, Fx Fy - Fxy L₁² + Fxy 𝕃₁, -Fy**2 + Fyy L₁**2 - Fyy 𝕃₁}
        #  }

        Hinv = np.array([
            [-xyxy_xx_yy, xyxy_xx_yy*L1, -xx_y__x_xy, -x_yy__xy_y],
            [xyxy_xx_yy*L1, -(x_yy__xy_y*Fx + xx_y__x_xy*Fy) + (-xyxy_xx_yy)*LL1, (xx_y__x_xy)*L1, (x_yy__xy_y)*L1],
            [-xx_y__x_xy, (-xx_y__x_xy)*L1, -Fx**2 + Fxx*L1**2 -  Fxx*LL1, Fx*Fy - Fxy*L1**2 +  Fxy*LL1],
            [-x_yy__xy_y, (-x_yy__xy_y)*L1, Fx*Fy - Fxy*L1**2 + Fxy*LL1, -Fy**2 + Fyy*L1**2 - Fyy*LL1]
        ]) / det

        # Its causing problems. Alas! LM time.
        H = np.array([
            [Haa, Hab, Hay, Hax],
            [Hab, Hbb, Hby, Hbx],
            [Hay, Hby, Hyy, Hxy],
            [Hax, Hbx, Hxy, Hxx]
        ])
        #Hinv = np.linalg.inv(H)

        Hadj = H + (J@J)*np.identity(4)*1e-8#*1e-15#*1e-3 # Damped least squares. This should help with convergence, and it should be fine as long as we get close enough that the Hessian is PSD, which I hope is always the case near the solution.
        Hinv = np.linalg.inv(Hadj)

        # Predict!
        step = Hinv @ J
        step[0] *= args[0] # force A to be "positive"
        args -= step
        args[0] = abs(args[0])
        #args[0] = np.abs(
        #print(f"  iter {n}: args={args}, |step|={np.linalg.norm(step):.12e}")
        #dyx -= H_inv @ np.array([Jy, Jx])
        #print(f"{dyx = }")
        #cv2_imshow(slog(np.abs(np.fft.ifft2(fft1*phasor).real-img0)))

    # Debug display of this loss.
    # phasor = np.exp(-itau*(ky*dyx[0] + kx*dyx[1]))
    # shifted = np.fft.ifft2(fft1*phasor).real
    # residual = A*shifted + B - img0
    # residual = cv2.resize(residual, (8*residual.shape[0], 8*residual.shape[1]), interpolation=cv2.INTER_LANCZOS4)
    # cv2_imshow(residual, title="Residual", prep=True)

    #return args[0], args[1], args[2:]
    return args


def animate_cv2_diff_with_subpixel_adj_phasecorr_V3(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    # Lets iterate through the entire sequence, and cache the resulting shifts in a map. Then we can apply the shifts with a single fft per frame, rather than iterating to convergence each frame. That should be much faster.
    print("Precomputing phase correlation shifts...")
    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[0] = np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    for idt in range(1, len(seq)):
        delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
        #phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
    print("Done precomputing phase correlation shifts!")

    # Now the potentially awful one. Lets iterate through the sequence, applying the subpixel shift, then cache a y axis slice to apply a median to get the median background.
    sector_median = np.zeros_like(ctrl_slice)
    
    # We should select our chunk size wisely. We're throwing out a LOT of data, but I cannot do the full spectral nuft for this without destroying the sector data.
    # sector_data.shape -> (1145, 2048, 2048)
    # at float64, that's 1145*2048*2048*8 bytes = 38.4 GB. Thats a lot, but maybe we can do it if we only keep one chunk in memory at a time. Lets say we do 128 pixel chunks, and we only keep the y axis slice for the median in memory. That would be 1145*128*2048*8 bytes = 2.3 GB, which is much more reasonable.
    # I can do 2.3GB! I'm at 73% usage of 64GB, which leaves ~17GB free. So our theoretical max chunk size is:
    # 17GB / (1145*2048*8 bytes) = 896 pixels. Lets go with 512. That grants us 4 iterations to compute the entire background.
    # Actually, this first test is a slice. Lets first validate that we wont kill the ram and throw an error if we will.

    chunksize = 512
    float64_sizeof = 8
    chunksize = min(chunksize, ctrl_slice.shape[0])
    num_chunks = (ctrl_slice.shape[0] + chunksize - 1) // chunksize

    est_nbytes = len(seq) * chunksize * ctrl_slice.shape[1] * float64_sizeof
    est_n_gb = est_nbytes / 1024**3
    print(f"Estimated memory usage for precomputing shifted frames: {est_n_gb:.2f} GB")
    lim_n_gb = 4.0
    print(f"Limiting malloc to {lim_n_gb:.2f} GB to avoid a scary pagefile.")
    if lim_n_gb < est_n_gb:
        raise MemoryError(f"Estimated memory usage of {est_n_gb:.2f} GB exceeds the limit of {lim_n_gb:.2f} GB. Aborting to avoid crashing the system.")
    
    tcx_chunk = np.zeros((len(seq), chunksize, ctrl_slice.shape[1]), dtype=np.float64)
    vmin, vmax = np.inf, -np.inf

    max_epochs = 1
    for epoch in range(max_epochs): # We can iterate this process to convergence if we want, but lets just do it twice for now and see how bad the residuals look.
        # icky O(n²) loop
        #for idy in range(len(seq)):
        print(f"[{epoch+1}] Applying subpixel shifts and computing median background in chunks...")
        for idc in range(num_chunks):
            y_start = idc*chunksize
            y_end = min((idc+1)*chunksize, ctrl_slice.shape[0])
            y_slice = slice(y_start, y_end)
            for idt in range(len(seq)):
                raw = fltr(seq[idt][idx_crop])
                A,B,dy,dx = delta_coords[idt]
                phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
                shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B
                tcx_chunk[idt, :y_end-y_start] = shifted[y_start:y_end]
                #tcx_chunk[idt, y_slice] = shifted[y_slice]
            # Now we have a chunk of the shifted frames in memory. We can compute the median across time for this chunk, and store it in the sector_median.
            #sector_median[y_slice] = np.median(tcx_chunk[:,:y_end-y_start], axis=0)
            # Lets try a sigma clip before doing the median. 
            for idy in range(y_start, y_end):
                for idx in range(sector_median.shape[1]):
                    pixel_stack = tcx_chunk[:, idy-y_start, idx]
                    med, mean, clow, chigh = sigclip_histogram(pixel_stack, 5.0, 5.0, returnmean=True)
                    sector_median[idy, idx] = mean
                    # this feels like it'll perform awfully. Lets see.
            
            print(f"[{epoch+1}] Computed median for chunk {idc+1}/{num_chunks}")
        #del tcx_chunk # Free memory
        print(f"[{epoch+1}] Done precomputing median background!")

        vmin, vmax = np.inf, -np.inf
        print(f"[{epoch+1}] Recomputing phase correlation shifts from the median background...")
        # Iterate again to recompute delta_coords from the median as our control!
        # Also get bounds.
        for idt in range(len(seq)):
            raw = fltr(seq[idt][idx_crop])
            #delta_coords[idt] = phase_corr_iterator(sector_median, raw)
            #delta_coords[idt] = matched_filter_phase_corr_iterator(sector_median, raw)
            trgt = fltr(seq[0][idx_crop]) if epoch+1 < max_epochs else sector_median
            delta_coords[idt] = matched_filter_phase_corr_iterator(trgt, raw)
            A,B,dy,dx = delta_coords[idt]
            phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
            shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B
            diff = shifted - sector_median
            med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
            vmin = min(vmin, clow)
            vmax = max(vmax, chigh)

        # Gauge fix: pin the 4-parameter degeneracy (shift, scale, offset)
        # so the median doesn't drift between epochs.
        # mean_dy = delta_coords[:, 2].mean()
        # mean_dx = delta_coords[:, 3].mean()
        # delta_coords[:, 2] -= mean_dy
        # delta_coords[:, 3] -= mean_dx
        # mean_A = delta_coords[:, 0].mean()
        # mean_B = delta_coords[:, 1].mean()
        # delta_coords[:, 0] /= mean_A
        # delta_coords[:, 1] = (delta_coords[:, 1] - mean_B) / mean_A

        print(f"[{epoch+1}] Done recomputing phase correlation shifts! vmin={vmin}, vmax={vmax}")

    del tcx_chunk # Free memory


    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum][idx_crop])

        #dyx = delta_coords[framenum] #phase_corr_iterator(ctrl_slice, raw)#[idx_slice])
        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B

        """
        diff = raw - ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 1: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale
        """

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)

        ema[:] = raw
        framenum += 1
        total_framenum += 1

def animate_cv2_diff_with_subpixel_adj_phasecorr_V4(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    # Lets iterate through the entire sequence, and cache the resulting shifts in a map. Then we can apply the shifts with a single fft per frame, rather than iterating to convergence each frame. That should be much faster.
    print("Precomputing phase correlation shifts...")
    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[0] = np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    for idt in range(1, len(seq)):
        delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
        #phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))
    print("Done precomputing phase correlation shifts!")

    # Now the potentially awful one. Lets iterate through the sequence, applying the subpixel shift, then cache a y axis slice to apply a median to get the median background.
    sector_median = np.zeros_like(ctrl_slice)
    
    # We should select our chunk size wisely. We're throwing out a LOT of data, but I cannot do the full spectral nuft for this without destroying the sector data.
    # sector_data.shape -> (1145, 2048, 2048)
    # at float64, that's 1145*2048*2048*8 bytes = 38.4 GB. Thats a lot, but maybe we can do it if we only keep one chunk in memory at a time. Lets say we do 128 pixel chunks, and we only keep the y axis slice for the median in memory. That would be 1145*128*2048*8 bytes = 2.3 GB, which is much more reasonable.
    # I can do 2.3GB! I'm at 73% usage of 64GB, which leaves ~17GB free. So our theoretical max chunk size is:
    # 17GB / (1145*2048*8 bytes) = 896 pixels. Lets go with 512. That grants us 4 iterations to compute the entire background.
    # Actually, this first test is a slice. Lets first validate that we wont kill the ram and throw an error if we will.

    chunksize = 512
    float64_sizeof = 8
    chunksize = min(chunksize, ctrl_slice.shape[0])
    num_chunks = (ctrl_slice.shape[0] + chunksize - 1) // chunksize

    est_nbytes = len(seq) * chunksize * ctrl_slice.shape[1] * float64_sizeof
    est_n_gb = est_nbytes / 1024**3
    print(f"Estimated memory usage for precomputing shifted frames: {est_n_gb:.2f} GB")
    lim_n_gb = 4.0
    print(f"Limiting malloc to {lim_n_gb:.2f} GB to avoid a scary pagefile.")
    if lim_n_gb < est_n_gb:
        raise MemoryError(f"Estimated memory usage of {est_n_gb:.2f} GB exceeds the limit of {lim_n_gb:.2f} GB. Aborting to avoid crashing the system.")
    
    tcx_chunk = np.zeros((len(seq), chunksize, ctrl_slice.shape[1]), dtype=np.float64)
    vmin, vmax = np.inf, -np.inf

    max_epochs = 1
    for epoch in range(max_epochs): # We can iterate this process to convergence if we want, but lets just do it twice for now and see how bad the residuals look.
        # icky O(n²) loop
        #for idy in range(len(seq)):
        # print(f"[{epoch+1}] Applying subpixel shifts and computing median background in chunks...")
        # for idc in range(num_chunks):
        #     y_start = idc*chunksize
        #     y_end = min((idc+1)*chunksize, ctrl_slice.shape[0])
        #     y_slice = slice(y_start, y_end)
        #     for idt in range(len(seq)):
        #         raw = fltr(seq[idt][idx_crop])
        #         A,B,dy,dx = delta_coords[idt]
        #         phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        #         shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B
        #         tcx_chunk[idt, :y_end-y_start] = shifted[y_start:y_end]
        #         #tcx_chunk[idt, y_slice] = shifted[y_slice]
        #     # Now we have a chunk of the shifted frames in memory. We can compute the median across time for this chunk, and store it in the sector_median.
        #     #sector_median[y_slice] = np.median(tcx_chunk[:,:y_end-y_start], axis=0)
        #     # Lets try a sigma clip before doing the median. 
        #     for idy in range(y_start, y_end):
        #         for idx in range(sector_median.shape[1]):
        #             pixel_stack = tcx_chunk[:, idy-y_start, idx]
        #             med, mean, clow, chigh = sigclip_histogram(pixel_stack, 5.0, 5.0, returnmean=True)
        #             sector_median[idy, idx] = mean
        #             # this feels like it'll perform awfully. Lets see.
            
        #     print(f"[{epoch+1}] Computed median for chunk {idc+1}/{num_chunks}")
        #del tcx_chunk # Free memory
        #print(f"[{epoch+1}] Done precomputing median background!")

        print(f"[{epoch+1}] Computing median background with coherent spectral mean...")
        # Lets try the coherent spectral mean to compute the median. 
        # fft_med = (Σₜ Aₜ phasorₜ imgₜ(k)) / (Σₜ Aₜ²)
        fft_sector_med = np.zeros(seq[0][idx_crop].shape, dtype=np.complex128)
        fft_sector_A2 = np.zeros(seq[0][idx_crop].shape, dtype=np.float64)
        for idt in range(len(seq)):
            raw = fltr(seq[idt][idx_crop])
            A,B,dy,dx = delta_coords[idt]
            phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
            #shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B
            fft_sector_med += A * phasor * np.fft.fft2(raw)
            fft_sector_A2 += A**2
        fft_sector_med /= fft_sector_A2
        sector_median = np.fft.ifft2(fft_sector_med).real
        # I hope this works!
        print(f"[{epoch+1}] Done computing median background with coherent spectral mean!")

        vmin, vmax = np.inf, -np.inf
        print(f"[{epoch+1}] Recomputing phase correlation shifts from the median background...")
        # Iterate again to recompute delta_coords from the median as our control!
        # Also get bounds.
        for idt in range(len(seq)):
            raw = fltr(seq[idt][idx_crop])
            #delta_coords[idt] = phase_corr_iterator(sector_median, raw)
            #delta_coords[idt] = matched_filter_phase_corr_iterator(sector_median, raw)
            trgt = fltr(seq[0][idx_crop]) if epoch+1 < max_epochs else sector_median
            delta_coords[idt] = matched_filter_phase_corr_iterator(trgt, raw)
            A,B,dy,dx = delta_coords[idt]
            phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
            shifted = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B
            diff = shifted - sector_median
            med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
            vmin = min(vmin, clow)
            vmax = max(vmax, chigh)



        print(f"[{epoch+1}] Done recomputing phase correlation shifts! vmin={vmin}, vmax={vmax}")

    del tcx_chunk # Free memory

    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum][idx_crop])

        #dyx = delta_coords[framenum] #phase_corr_iterator(ctrl_slice, raw)#[idx_slice])
        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B

        """
        diff = raw - ema
        med, clow, chigh = sigclip_histogram(diff, 5.0,5.0)
        emamin = (1.0-ema_alpha)*emamin + ema_alpha*clow#diff.min()
        emamax = (1.0-ema_alpha)*emamax + ema_alpha*chigh#diff.max()
        if total_framenum == 1: # try to fix the flashbang
            emamin = clow
            emamax = chigh
        vscale = 1.0/(1.0-ema_alpha**(total_framenum+1))
        vmin = emamin*vscale
        vmax = emamax*vscale
        """

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin, Max=vmax)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


# This variant skips all precomputation for both median and background, and instead iteratively computes it as we animate.
def animate_cv2_diff_with_subpixel_adj_phasecorr_V5_progressive(seq, cplx=False, EMA=False, ema_alpha=0.1, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0



    px0, py0 = 1426, 1616
    # Crop the diff to +32 and -32 around this pixel, zoom in with lanczos. Let me get the code for that rq.
    #crop = diff[py0-32:py0+32, px0-32:px0+32]
    #idx_crop = (slice(py0-64,py0+64), slice(px0-64,px0+64))
    #crop_width = 64//2
    crop_width = 128//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    idx_slice = idx_crop

    ctrl_slice = seq[0][idx_slice]
    ema = ema[idx_crop]

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[:,0] = 1.0 # A = 1 #np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    #for idt in range(1, len(seq)):
    #    delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, fltr(seq[idt][idx_crop]))

    sector_median = fltr(ctrl_slice.copy()) #np.zeros_like(ctrl_slice)
    # seed the sector median with the usual for loop over x slices.
    for xx in range(sector_median.shape[1]):
        sector_median[:,xx] = np.median(fltr(seq[:, *idx_crop][:, :, xx]), axis=0)
        # me: actually, before I continue, I've got a question. This kind of array slicing, will it copy the array when I perform the slice? That is, if I do [:, idx_crop], this gives a chunk through the volume I'm processing. Next, I want to select a given column from this volume, so normally i'd do seq[:, idx_crop][:, :, xx]. But, this *iterated* slice is concerning, as I can *never* copy this dataset, as its over half my ram. Will this copy, or is merely a reference?
        # Copilot: In Python, slicing an array typically creates a view (reference) rather than a copy, especially when using NumPy arrays. However, the behavior can depend on how the slicing is done and the underlying memory layout of the array. In your case, when you do seq[:, idx_crop], it will create a view of the original array without copying the data. This means that when you further slice this view with [:, :, xx], it will still reference the original data without creating a new copy. So, you should be able to perform these operations without running into memory issues due to copying, as long as you are working with NumPy arrays and not performing any operations that would force a copy (like certain types of indexing or modifications).


    fft_sector_med = np.zeros(ctrl_slice.shape, dtype=np.complex128)
    n_accumulated = 0
    vmin0, vmax0 = None, None  #np.inf, -np.inf
    vmin1, vmax1 = np.inf, -np.inf

    framenum = 0
    total_framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = fltr(seq[framenum][idx_crop])

        # Use the previous frame's shift as the initial guess for this frame's shift, to speed up convergence. This should be fine as long as the shifts don't change too much between frames, which is probably true given the cadence of these observations.
        init = delta_coords[framenum] if len(seq) <= total_framenum  else delta_coords[max(0,framenum-1)]
        delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw, init=init, tol=1e-12)
        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        N_pix = raw.shape[0] * raw.shape[1]
        raw_fft = np.fft.fft2(raw)*phasor*A
        raw_fft[0,0] += B * N_pix  # B is per-pixel offset → DC contribution is B·N_pix
        raw = np.fft.ifft2(raw_fft).real

        # Update the sector median accumulators.
        # The WLS scene estimate is: ŝ(k) = (1/N) Σ_n [A_n e^{-ikΔ_n} F̂_n(k)]
        # with B·N_pix added at k=0. Denominator is just N_frames, not Σ A².
        fft_sector_med += raw_fft
        n_accumulated += 1

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        big_img = cv2.resize(crop, (8*crop.shape[0], 8*crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, prep=prep, fps=fps, Min=vmin0, Max=vmax0)

        # sigclip big_img
        med, clow, chigh = sigclip_histogram(crop, 5.0, 5.0)
        vmin1 = min(vmin1, clow)
        vmax1 = max(vmax1, chigh)

        ema[:] = raw
        framenum += 1
        total_framenum += 1

        # Update the sector median from the accumulators
        if framenum == len(seq): # This is wrapped AFTER this step.
            sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
            fft_sector_med.fill(0.0)
            n_accumulated = 0
            #vmin0, vmax0 = vmin1, vmax1
            #vmin1, vmax1 = np.inf, -np.inf
            # Gauge fix: pin all 4 degenerate directions
            mean_dy = delta_coords[:, 2].mean()
            mean_dx = delta_coords[:, 3].mean()
            delta_coords[:, 2] -= mean_dy
            delta_coords[:, 3] -= mean_dx
            mean_A = delta_coords[:, 0].mean()
            mean_B = delta_coords[:, 1].mean()
            delta_coords[:, 0] /= mean_A
            delta_coords[:, 1] = (delta_coords[:, 1] - mean_B) / mean_A

def animate_cv2_diff_with_subpixel_adj_phasecorr_V6_progressive():

    seq = slog_data#[:32]
    ctrl_slice = np.median(seq, axis=0)#[:32], axis=0)#slog_median
    sector_median = ctrl_slice.copy()

    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    sector_cache = np.zeros((len(seq),) + ctrl_slice.shape, dtype=np.float64)
    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[:,0] = 1.0 # A = 1 #np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    #for idt in range(1, len(seq)):
    for idt in range(len(seq)):
        delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, seq[idt], init=delta_coords[max(0,idt-1)], iters=10, spectral_blur=4)
        # Fill sector cache with the initially aligned frames, so we have something to display while we compute the median.
        A,B,dy,dx = delta_coords[idt]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(seq[idt])*phasor).real*A + B
        sector_cache[idt] = raw

    print(f"{delta_coords[0] = }")

    #sector_median = fltr(ctrl_slice.copy()) #np.zeros_like(ctrl_slice)
    fft_sector_med = np.zeros(ctrl_slice.shape, dtype=np.complex128)
    n_accumulated = 0
    vmin0, vmax0 = None, None  #np.inf, -np.inf
    vmin1, vmax1 = np.inf, -np.inf



    framenum = 0
    total_framenum = 0
    chunksize = 32#128#32
    #chunksize = 2
    chunklim = 0 + chunksize*2
    blursize = 16.0

    # Lets make it fit to local medians that grow over time, to emulate a pyramid structure.
    # For each 32 frames we fit a local median, then in the next iteration we fit every 64 frames, 128 frames, etc.

    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        #raw = fltr(seq[framenum][idx_crop])
        raw = seq[framenum]#[idx_crop]

        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[max(0,framenum-1)], spectral_weight=True)#, tol=1e-16)

        # Lets set the spectral blur to be related to the distance from the chunksize.
        # That is, if we start at chunklim=64, with chunksize=32, then:
        # I'd like the blur to be 0px at 32, and 32px at 64.
        #spectral_blur = chunksize * max(0.0, min(1.0, (framenum-chunklim+chunksize)/chunksize))
        #spectral_blur = min(chunksize, spectral_blur)
        #spectral_blur = None if spectral_blur == 0.0 else spectral_blur
        spectral_blur = None if blursize < 1.0 else blursize
        #spectral_blur = None
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[framenum], iters=10, spectral_blur=spectral_blur)#tol=1e-12)#, spectral_weight=True)#, tol=1e-16)
        delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw, init=delta_coords[framenum], iters=5, spectral_blur=spectral_blur)#, tol=1e-12)#, spectral_weight=True)#, tol=1e-16)

        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        N_pix = raw.shape[0] * raw.shape[1]
        raw_fft = np.fft.fft2(raw)*phasor*A
        raw_fft[0,0] += B * N_pix  # B is per-pixel offset → DC contribution is B·N_pix
        raw = np.fft.ifft2(raw_fft).real
        sector_cache[framenum] = raw

        # Update delta coords afterwards. I think coupling this with the sector_median changes is causing it to not converge.
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw)#, init=init)#, tol=1e-16)

        # Update the sector median accumulators.
        # The WLS scene estimate is: ŝ(k) = (1/N) Σ_n [A_n e^{-ikΔ_n} F̂_n(k)]
        # with B·N_pix added at k=0. Denominator is just N_frames, not Σ A².
        if spectral_blur is None or True:
            fft_sector_med += raw_fft
            n_accumulated += 1

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        #crop = raw - ctrl_slice#sector_median
        #big_img = cv2.resize(crop, (2048,2048), interpolation=cv2.INTER_LANCZOS4)
        big_img = cv2.resize(crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, Min=vmin0, Max=vmax0)
        #cv2_imshow(crop, title=title, Min=vmin0, Max=vmax0)

        # sigclip big_img
        if spectral_blur is None or True:
            med, clow, chigh = sigclip_histogram(crop, 5.0, 5.0)
            vmin1 = min(vmin1, clow)
            vmax1 = max(vmax1, chigh)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


        # Update the sector median from the accumulators
        if framenum == len(seq): # This is wrapped AFTER this step.
        #if framenum == len(seq) or framenum == chunklim:
            #sector_median = np.median(sector_cache[:chunklim-chunksize], axis=0)
            #sector_median = np.median(sector_cache, axis=0)
            blursize /= 2.0#-= 1

            #framenum = 0
            #chunklim += chunksize*2
            #chunklim *= 2
            sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
            fft_sector_med.fill(0.0)
            n_accumulated = 0
            #vmin0, vmax0 = np.nanmin(sector_cache), np.nanmax(sector_cache)
            vmin0, vmax0 = vmin1, vmax1
            vmin1, vmax1 = np.inf, -np.inf

            #vlim = min(-vmin0, vmax0)
            #vmin0, vmax0 = -vlim, vlim

            #sector_median = np.median(sector_cache[], axis=0)
            #beta = 0.5
            #sector_median = (1.0-beta)*sector_median + beta*np.median(sector_cache, axis=0)

            #print(f"{delta_coords.mean(axis=0) = }, {delta_coords.std(axis=0) = }")

        globals().update(locals())

def animate_cv2_diff_with_subpixel_adj_phasecorr_V7_progressive():

    seq = slog_data#[:32]
    ctrl_slice = np.median(seq, axis=0)#[:32], axis=0)#slog_median
    sector_median = ctrl_slice.copy()

    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    sector_cache = np.zeros((len(seq),) + ctrl_slice.shape, dtype=np.float64)
    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[:,0] = 1.0 # A = 1 #np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    #for idt in range(1, len(seq)):
    for idt in range(len(seq)):
        delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, seq[idt], init=delta_coords[max(0,idt-1)], iters=10, spectral_blur=4)
        # Fill sector cache with the initially aligned frames, so we have something to display while we compute the median.
        A,B,dy,dx = delta_coords[idt]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(seq[idt])*phasor).real*A + B
        sector_cache[idt] = raw

    print(f"{delta_coords[0] = }")

    #sector_median = fltr(ctrl_slice.copy()) #np.zeros_like(ctrl_slice)
    fft_sector_med = np.zeros(ctrl_slice.shape, dtype=np.complex128)
    n_accumulated = 0
    vmin0, vmax0 = None, None  #np.inf, -np.inf
    vmin1, vmax1 = np.inf, -np.inf



    framenum = 0
    total_framenum = 0
    chunksize = 32#128#32
    #chunksize = 2
    chunklim = 0 + chunksize*2
    blursize = 16.0

    # Lets make it fit to local medians that grow over time, to emulate a pyramid structure.
    # For each 32 frames we fit a local median, then in the next iteration we fit every 64 frames, 128 frames, etc.

    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        #raw = fltr(seq[framenum][idx_crop])
        raw = seq[framenum]#[idx_crop]

        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[max(0,framenum-1)], spectral_weight=True)#, tol=1e-16)

        # Lets set the spectral blur to be related to the distance from the chunksize.
        # That is, if we start at chunklim=64, with chunksize=32, then:
        # I'd like the blur to be 0px at 32, and 32px at 64.
        #spectral_blur = chunksize * max(0.0, min(1.0, (framenum-chunklim+chunksize)/chunksize))
        #spectral_blur = min(chunksize, spectral_blur)
        #spectral_blur = None if spectral_blur == 0.0 else spectral_blur
        spectral_blur = None if blursize < 1.0 else blursize
        #spectral_blur = None
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[framenum], iters=10, spectral_blur=spectral_blur)#tol=1e-12)#, spectral_weight=True)#, tol=1e-16)
        delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw, init=delta_coords[framenum], iters=5, spectral_blur=spectral_blur)#, tol=1e-12)#, spectral_weight=True)#, tol=1e-16)

        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        N_pix = raw.shape[0] * raw.shape[1]
        raw_fft = np.fft.fft2(raw)*phasor*A
        raw_fft[0,0] += B * N_pix  # B is per-pixel offset → DC contribution is B·N_pix
        raw = np.fft.ifft2(raw_fft).real
        sector_cache[framenum] = raw

        # Update delta coords afterwards. I think coupling this with the sector_median changes is causing it to not converge.
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw)#, init=init)#, tol=1e-16)

        # Update the sector median accumulators.
        # The WLS scene estimate is: ŝ(k) = (1/N) Σ_n [A_n e^{-ikΔ_n} F̂_n(k)]
        # with B·N_pix added at k=0. Denominator is just N_frames, not Σ A².
        if spectral_blur is None or True:
            fft_sector_med += raw_fft
            n_accumulated += 1

        title = f'tess s4 c1 ccd4 f{framenum}'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        #crop = raw - ctrl_slice#sector_median
        #big_img = cv2.resize(crop, (2048,2048), interpolation=cv2.INTER_LANCZOS4)
        big_img = cv2.resize(crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
        cv2_imshow(big_img, title=title, Min=vmin0, Max=vmax0)
        #cv2_imshow(crop, title=title, Min=vmin0, Max=vmax0)

        # sigclip big_img
        if spectral_blur is None or True:
            med, clow, chigh = sigclip_histogram(crop, 5.0, 5.0)
            vmin1 = min(vmin1, clow)
            vmax1 = max(vmax1, chigh)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


        # Update the sector median from the accumulators
        if framenum == len(seq): # This is wrapped AFTER this step.
        #if framenum == len(seq) or framenum == chunklim:
            #sector_median = np.median(sector_cache[:chunklim-chunksize], axis=0)
            #sector_median = np.median(sector_cache, axis=0)
            blursize /= 2.0#-= 1

            #framenum = 0
            #chunklim += chunksize*2
            #chunklim *= 2
            sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
            fft_sector_med.fill(0.0)
            n_accumulated = 0
            #vmin0, vmax0 = np.nanmin(sector_cache), np.nanmax(sector_cache)
            vmin0, vmax0 = vmin1, vmax1
            vmin1, vmax1 = np.inf, -np.inf

            #vlim = min(-vmin0, vmax0)
            #vmin0, vmax0 = -vlim, vlim

            #sector_median = np.median(sector_cache[], axis=0)
            #beta = 0.5
            #sector_median = (1.0-beta)*sector_median + beta*np.median(sector_cache, axis=0)

            #print(f"{delta_coords.mean(axis=0) = }, {delta_coords.std(axis=0) = }")

        globals().update(locals())


def animate_cv2_diff_with_subpixel_adj_phasecorr_V9_pyramid():
    """
    Progressive refinement with both spectral blur annealing AND gated scene growth.
    
    Two complementary mechanisms:
      1. Spectral blur (coarse→fine in frequency): controls the optimizer's basin
         of attraction. Heavy blur = only low-freq alignment, wide basin.
      2. Growing window (coarse→fine in time): controls scene quality. Only frames
         that have been fit WITHOUT blur (fully converged) enter the coadd.
    
    The growing window prevents the failure mode where poorly-fit frames from
    early epochs contaminate the scene, creating a positive feedback loop.
    This is the same principle as RANSAC: build the model from inliers first,
    then gradually expand.
    """
    seq = slog_data
    N = len(seq)
    shape = seq[0].shape
    N_pix = shape[0] * shape[1]

    ky, kx = np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.fftfreq(shape[1]), indexing='ij')

    # Working state
    delta_coords = np.zeros((N, 4), dtype=np.float64)
    delta_coords[:, 0] = 1.0
    sector_cache = np.zeros((N,) + shape, dtype=np.float64)

    def apply_delta(frame, delta):
        A, B, dy, dx = delta
        phasor = np.exp(-2j * np.pi * (ky * dy + kx * dx))
        fft_out = np.fft.fft2(frame) * phasor * A
        fft_out[0, 0] += B * N_pix
        return np.fft.ifft2(fft_out).real

    # ── Seed pass: fit all frames against the global median with heavy blur ──
    ctrl = np.median(seq, axis=0)
    for i in range(N):
        delta_coords[i] = matched_filter_phase_corr_iterator(
            ctrl, seq[i], init=delta_coords[max(0, i - 1)], iters=10, spectral_blur=8)
        sector_cache[i] = apply_delta(seq[i], delta_coords[i])
    print(f"Seed done.  shifts: mean={delta_coords[:, 2:].mean(0)}, std={delta_coords[:, 2:].std(0)}")

    # ── Growing-window progressive refinement ──
    chunksize = 16                     # frames added to the window each cycle
    chunklim = chunksize * 2           # current window upper bound
    blursize = 8.0                     # spectral blur in pixels; anneals per-frame within window

    fft_sector_med = np.zeros(shape, dtype=np.complex128)
    n_accumulated = 0
    # Bootstrap scene from the first chunk of seeded frames
    for i in range(min(chunksize, N)):
        fft_sector_med += np.fft.fft2(sector_cache[i])
        n_accumulated += 1
    sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
    fft_sector_med.fill(0.0)
    n_accumulated = 0

    vmin0, vmax0 = None, None
    vmin1, vmax1 = np.inf, -np.inf
    framenum = 0

    while cv2.waitKey(1) & 0xFF != ord('q'):
        raw = seq[framenum]

        # ── Per-frame blur: ramp from blursize at the window edge down to None
        # for frames well inside the converged zone.
        # Frames < (chunklim - chunksize) are in the converged zone → no blur.
        # Frames in [chunklim-chunksize, chunklim) get decreasing blur.
        dist_from_edge = chunklim - chunksize - framenum  # positive = inside converged zone
        if dist_from_edge > 0 or blursize < 0.5:
            spectral_blur = None
        else:
            frac = max(0.0, min(1.0, (framenum - (chunklim - chunksize)) / chunksize))
            spectral_blur = blursize * frac
            spectral_blur = None if spectral_blur < 0.5 else spectral_blur

        tol = 1e-6 if chunklim < N else 1e-12
        delta_coords[framenum] = matched_filter_phase_corr_iterator(
            sector_median, raw, init=delta_coords[framenum],
            iters=50, spectral_blur=spectral_blur, tol=tol)

        # Apply registration
        A, B, dy, dx = delta_coords[framenum]
        phasor = np.exp(-2j * np.pi * (ky * dy + kx * dx))
        raw_fft = np.fft.fft2(raw) * phasor * A
        raw_fft[0, 0] += B * N_pix
        sector_cache[framenum] = np.fft.ifft2(raw_fft).real

        # ── Gated accumulation: only fully-converged (unblurred) frames enter the scene ──
        if spectral_blur is None:
            fft_sector_med += raw_fft
            n_accumulated += 1

        # Display
        crop = sector_cache[framenum] - sector_median
        big_img = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)
        title = f'f{framenum}/{N} [:{chunklim - chunksize}] blur={spectral_blur}'
        if spectral_blur is None:
            med, clow, chigh = sigclip_histogram(crop, 5.0, 5.0)
            vmin1 = min(vmin1, clow)
            vmax1 = max(vmax1, chigh)
        cv2_imshow(big_img, title=title, Min=vmin1, Max=vmax1)

        framenum += 1

        # ── Window boundary: rebuild scene, grow window, anneal blur ──
        if framenum >= N or framenum >= chunklim:
            if n_accumulated > 0:
                sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
                fft_sector_med.fill(0.0)
                n_accumulated = 0

            # Grow window
            framenum = 0
            chunklim = min(chunklim + chunksize, N + chunksize)  # let it exceed N to enter full-data mode

            # Once the window covers all frames, start annealing blur for subsequent passes
            if chunklim > N:
                blursize /= 2.0

            vmin0, vmax0 = vmin1, vmax1
            vmin1, vmax1 = np.inf, -np.inf

            print(f"Window→{min(chunklim - chunksize, N)}/{N}, blur={blursize:.2f}, "
                  f"A={delta_coords[:,0].mean():.6f}±{delta_coords[:,0].std():.6f}, "
                  f"B={delta_coords[:,1].mean():.6f}±{delta_coords[:,1].std():.6f}, "
                  f"dy={delta_coords[:,2].mean():.4f}±{delta_coords[:,2].std():.4f}, "
                  f"dx={delta_coords[:,3].mean():.4f}±{delta_coords[:,3].std():.4f}")

    globals().update(locals())


def animate_cv2_diff_with_subpixel_adj_phasecorr_V10_progressive():

    seq = slog_data#[:32]
    ctrl_slice = np.median(seq, axis=0)#[:32], axis=0)#slog_median
    sector_median = ctrl_slice.copy()

    ema = seq[0].copy()
    emamin = 0.0
    emamax = 0.0

    ky,kx = np.meshgrid(np.fft.fftfreq(ema.shape[0]), np.fft.fftfreq(ema.shape[1]), indexing='ij')
    #phasor = np.exp(-2j*np.pi*(ky*dyx[0] + kx*dyx[1]))

    sector_cache = np.zeros((len(seq),) + ctrl_slice.shape, dtype=np.float64)
    delta_coords = np.zeros((len(seq), 4), dtype=np.float64)
    delta_coords[:,0] = 1.0 # A = 1 #np.array([1.0, 0.0, 0.0, 0.0]) # A=1, B=0, Δy=0, Δx=0 for the control frame
    #for idt in range(1, len(seq)):
    for idt in range(len(seq)):
        delta_coords[idt] = matched_filter_phase_corr_iterator(ctrl_slice, seq[idt], init=delta_coords[max(0,idt-1)], iters=10, spectral_blur=4)
        # Fill sector cache with the initially aligned frames, so we have something to display while we compute the median.
        A,B,dy,dx = delta_coords[idt]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(seq[idt])*phasor).real*A + B
        sector_cache[idt] = raw

    print(f"{delta_coords[0] = }")

    #sector_median = fltr(ctrl_slice.copy()) #np.zeros_like(ctrl_slice)
    fft_sector_med = np.zeros(ctrl_slice.shape, dtype=np.complex128)
    n_accumulated = 0
    vmin0, vmax0 = None, None  #np.inf, -np.inf
    vmin1, vmax1 = np.inf, -np.inf



    framenum = 0
    total_framenum = 0
    chunksize = 16#32#128#32
    #chunksize = 2
    chunklim = 0 + chunksize*2
    blursize = 16.0

    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        #raw = fltr(seq[framenum][idx_crop])
        raw = seq[framenum]#[idx_crop]

        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[max(0,framenum-1)], spectral_weight=True)#, tol=1e-16)

        # Lets set the spectral blur to be related to the distance from the chunksize.
        # That is, if we start at chunklim=64, with chunksize=32, then:
        # I'd like the blur to be 0px at 32, and 32px at 64.
        spectral_blur = chunksize * max(0.0, min(1.0, (framenum-chunklim+chunksize)/chunksize))
        spectral_blur = min(chunksize, spectral_blur)
        spectral_blur = None if spectral_blur == 0.0 else spectral_blur
        #spectral_blur = None if blursize < 1.0 else blursize
        #spectral_blur = None
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(ctrl_slice, raw, init=delta_coords[framenum], iters=10, spectral_blur=spectral_blur)#tol=1e-12)#, spectral_weight=True)#, tol=1e-16)

        tol = 1e-6 if chunklim < len(seq) else 1e-12
        delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw, init=delta_coords[framenum], iters=50, spectral_blur=spectral_blur, tol=tol)#tol=1e-12)#, spectral_weight=True)#, tol=1e-16)

        A,B,dy,dx = delta_coords[framenum]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        N_pix = raw.shape[0] * raw.shape[1]
        raw_fft = np.fft.fft2(raw)*phasor*A
        raw_fft[0,0] += B * N_pix  # B is per-pixel offset → DC contribution is B·N_pix
        raw = np.fft.ifft2(raw_fft).real
        sector_cache[framenum] = raw

        # Update delta coords afterwards. I think coupling this with the sector_median changes is causing it to not converge.
        #delta_coords[framenum] = matched_filter_phase_corr_iterator(sector_median, raw)#, init=init)#, tol=1e-16)

        # Update the sector median accumulators.
        # The WLS scene estimate is: ŝ(k) = (1/N) Σ_n [A_n e^{-ikΔ_n} F̂_n(k)]
        # with B·N_pix added at k=0. Denominator is just N_frames, not Σ A².
        if spectral_blur is None:
            fft_sector_med += raw_fft
            n_accumulated += 1

        title = f'tess s4 c1 ccd4 f{framenum} [:{chunklim-chunksize}]'# + f' shift=({Xmean:.2f},{Ymean:.2f}) a={a_coeff:.1f} b={b_coeff:.4f}'
        #crop = diff#[idx_crop]
        crop = raw - sector_median
        #crop = raw - ctrl_slice#sector_median
        #big_img = cv2.resize(crop, (2048,2048), interpolation=cv2.INTER_LANCZOS4)
        big_img = cv2.resize(crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
        #cv2_imshow(big_img, title=title, Min=vmin0, Max=vmax0)
        #cv2_imshow(crop, title=title, Min=vmin0, Max=vmax0)

        # sigclip big_img
        if spectral_blur is None:# or True:
            med, clow, chigh = sigclip_histogram(crop, 5.0, 5.0)
            vmin1 = min(vmin1, clow)
            vmax1 = max(vmax1, chigh)
        cv2_imshow(big_img, title=title, Min=vmin1, Max=vmax1)

        ema[:] = raw
        framenum += 1
        total_framenum += 1


        # Update the sector median from the accumulators
        #if framenum == len(seq): # This is wrapped AFTER this step.
        if framenum == len(seq) or framenum == chunklim:
            #sector_median = np.median(sector_cache[:chunklim-chunksize], axis=0)
            #sector_median = np.median(sector_cache, axis=0)
            #blursize /= 2.0#-= 1

            framenum = 0
            chunklim += chunksize
            #chunklim *= 2
            sector_median = np.fft.ifft2(fft_sector_med / n_accumulated).real
            fft_sector_med.fill(0.0)
            n_accumulated = 0
            #vmin0, vmax0 = np.nanmin(sector_cache), np.nanmax(sector_cache)
            vmin0, vmax0 = vmin1, vmax1
            vmin1, vmax1 = np.inf, -np.inf

            #vlim = min(-vmin0, vmax0)
            #vmin0, vmax0 = -vlim, vlim

            #sector_median = np.median(sector_cache[], axis=0)
            #beta = 0.5
            #sector_median = (1.0-beta)*sector_median + beta*np.median(sector_cache, axis=0)

            #print(f"{delta_coords.mean(axis=0) = }, {delta_coords.std(axis=0) = }")

        globals().update(locals())

# I've run an output of this and I've collected a set of delta coords:
# delta_coords[:, n] = [A, B, dy, dx].
# Plotting the subpixel drift is *fascinating* and shows spacecraft internal state:
# plt.plot(delta_coords[:,3], delta_coords[:,2])
# Lets make a far more refined plot of this data.
def make_delta_coords_plot(delta_coords):
    fig, ax = plt.subplots(figsize=(8,8))
    A, B, dy, dx = delta_coords.T
    #sc = ax.scatter(dx, dy, c=np.arange(len(dx)), cmap='viridis', s=10)
    #sc = ax.plot(dx, dy, '-o', markersize=2)
    #dy, dx = dy*1e3, dx*1e3  # convert to millipixels for better readability
    dy -= np.mean(dy) # I accidentally mutated this. I hope this corrects for the error...
    dx -= np.mean(dx)
    ax.set_xlabel('dx (pixels)')
    ax.set_ylabel('dy (pixels)')
    #ax.set_title('Subpixel Drift Over Time')
    # Sector 19, Camera 1, CCD 1
    ax.set_title('TESS Sector 19 Camera 1 CCD 1 Subpixel Drift')
    ax.grid()
    # Lets force the grid to be at 1.0 pixel intervals!
    #ax.set_xticks(np.arange(np.floor(dx.min()), np.ceil(dx.max()) + 1, 1.0))
    #ax.set_yticks(np.arange(np.floor(dy.min()), np.ceil(dy.max()) + 1, 1.0))
    ax.set_aspect('equal', adjustable='box')

    from matplotlib.collections import LineCollection
    points = np.array([dx, dy]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(0, len(dx)), capstyle='round')
    lc.set_array(np.arange(len(dx)))
    lc.set_linewidth(2)
    ax.add_collection(lc, autolim=True)
    fig.colorbar(lc, ax=ax, label='Frame Index')
    ax.autoscale_view()

    from matplotlib.ticker import EngFormatter
    formatter = EngFormatter(unit='px', sep='')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    plt.show()
    # Now lets save the figure at high dpi!
    #fig.savefig(r'C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\sector19_cam1_ccd1_subpixel_drift.png', dpi=400)
    # Clear the figure so we dont accidentally show it.
    #plt.close(fig)


def offset_2d_medians_inplace(sector_data):
    #img_seq_diff -= np.median(img_seq_diff, axis=(0,1))[None,None,:]
    #img_seq_diff -= np.median(img_seq_diff, axis=(1,2))[:,None,None]
    #img_seq_diff -= np.median(img_seq_diff, axis=(0,2))[None,:,None]

    # x axis offsets
    t0 = time_ns()
    for xx in range(sector_data.shape[2]):
        sector_data[:,:,xx] -= np.median(sector_data[:,:,xx])[None,None]
        t1 = time_ns()
         # I should have the screen real estate for the full temporal slices too!
        cv2_imshow(sector_data[:,:,xx], title=f"x={xx}/{sector_data.shape[2]}")
        if (t1-t0) > 60e9:
            print(f"Processed x={xx} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("x medians done")

    # I believe the order mattered here? But I'm not sure really. I suspect it shouldn't..?
    # t axis offsets
    t0 = time_ns()
    for tt in range(sector_data.shape[0]):
        sector_data[tt,:,:] -= np.median(sector_data[tt,:,:])[None,None]
        t1 = time_ns()
        cv2_imshow(sector_data[tt,:,:], title=f"t={tt}/{sector_data.shape[0]}")
        if (t1-t0) > 60e9:
            print(f"Processed t={tt} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("t medians done")

    # y axis offsets
    t0 = time_ns()
    for yy in range(sector_data.shape[1]):
        sector_data[:,yy,:] -= np.median(sector_data[:,yy,:])[None,None]
        t1 = time_ns()
        cv2_imshow(sector_data[:,yy,:], title=f"y={yy}/{sector_data.shape[1]}")
        if (t1-t0) > 60e9:
            print(f"Processed y={yy} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("y medians done")

def kill_ram_chunk_process():
    kill_ram_chunk()
    globals().update(locals())
    #px0, py0 = 1426, 1616
    #crop_width = 512//2
    #idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    #sector_data = sector_data[:, *idx_crop]
    offset_2d_medians_inplace(sector_data)
    globals().update(locals())
    # delta_coords is already precomputed, lets find the global sector median and then apply the phase correction to each frame, then we can animate the results.
    # Due to the extreme ram costs, we cannot find the global median. Lets first do an in-place transform of the data via this spectral process.
    ky,kx = np.meshgrid(np.fft.fftfreq(sector_data.shape[1]), np.fft.fftfreq(sector_data.shape[2]), indexing='ij')
    for tt in range(sector_data.shape[0]):
        A, B, dy, dx = delta_coords[tt]
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        sector_data[tt] = np.fft.ifft2(np.fft.fft2(sector_data[tt])*phasor)*A + B
        tt_ = tt+1
        if tt_ % 100 == 0:
            print(f"Processed t={tt_}/{sector_data.shape[0]}")
    globals().update(locals())
    
    # Now compute the sector median by going by every x slice.
    sector_median = np.zeros(sector_data.shape[1:], dtype=np.float64)
    for xx in range(sector_data.shape[2]):
        sector_median[:,xx] = np.median(sector_data[:,:,xx], axis=0)
        if (xx+1) % 100 == 0:
            print(f"Processed x={xx+1}/{sector_data.shape[2]}")

    globals().update(locals())
    # animate!

    # Before animating, estimate v bounds.
    vlow = np.min(delta_coords[:,1])  # B offset min
    vhigh = np.max(delta_coords[:,1]) # B offset max

    framenum = 0
    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        raw = sector_data[framenum]
        diff = raw - sector_median
        title = f'tess s4 c1 ccd4 f{framenum} B={delta_coords[framenum,1]:.2f}'
        cv2_imshow(diff, title=title, Min=vlow, Max=vhigh)
        framenum += 1
        globals().update(locals())

def progressive_refine_bilinear():
    # Since the actual pixels are "cells," their "masses" are transferred linearly to neighboring ones with subpixel adjustments.
    # The combination of which is proportional to how much each pixel overlaps with its 4 neighbors, which is by definition,
    # a bilinear interpolation. 
    pass


# More terribly logged data... I need to really organize this file. :(
def extract_bigdiff_wavelet_error():
    crop = raw - sector_median
    big_img = cv2.resize(crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
    big_raw = cv2.resize(raw, (512,512), interpolation=cv2.INTER_LANCZOS4)
    big_med = cv2.resize(sector_median, (512,512), interpolation=cv2.INTER_LANCZOS4)
    fig, ax = plt.subplots()

    big_src = (big_raw-B)/A
    big_diff = big_src - big_med

    #ax.plot(big_raw[:,14])
    #ax.plot(big_med[:,14])
    #ax.plot(big_src[:,14])
    
    ax.plot(big_img[:,14])
    ax.plot(big_diff[:,14])
    plt.show()

# Lets make a variant of the above to use an interactive matplotlib plot with two sliders for A and B.
def extract_bigdiff_wavelet_error_manipulate(A, B):
    crop = raw - sector_median
    big_img = cv2.resize(crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
    big_raw = cv2.resize(raw, (512,512), interpolation=cv2.INTER_LANCZOS4)
    big_med = cv2.resize(sector_median, (512,512), interpolation=cv2.INTER_LANCZOS4)

    big_src = (big_raw-B)/A
    big_diff = big_src - big_med


    # Lets leave the above fixed and refer to everything below as our data.
    # We'll need to import the mpl sliders.
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.widgets

    fig, ax = plt.subplots()

    state = np.array([A, B]).copy() # to prevent us overwriting A,B
    x = np.arange(big_img.shape[0])
    # Now lets cache the plots to modify in the update callbacks
    line_img, = ax.plot(x, big_img[:,14])
    line_diff, = ax.plot(x, big_diff[:,14])

    # Lets analyze how the callbacks effect the data, as I'm seeing something strange.
    # diff = src*A + B - med
    # The default mean signal of diff is near zero, with the std dominating:
    # μ=0.0006754864784071501, σ²=0.010786352238717656
    # Lets suppose we fix this value at precisely 0 for the sake of argument. Then:
    # ∫ (src*A + B - med) pdf(x)dx = 0
    # A∫ src pdf(x)dx + B∫ pdf(x)dx - ∫ med pdf(x)dx = 0
    # A E[src] + B - E[med] = 0
    # B = E[med] - A E[src]
    # Thus, there is one parameter.
    # Lets solve for both to keep things interesting.
    # A = (E[med] - B) / E[src]
    mean_median = big_med[:,14].mean()
    mean_src = big_src[:,14].mean()


    # Add vertical sliders for both A and B
    ax_A = fig.add_axes([0.25, 0.01, 0.65, 0.03])  # [left, bottom, width, height]
    ax_B = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    slider_A = matplotlib.widgets.Slider(ax_A, 'A', 0.5, 1.5, valinit=A)
    #slider_B = matplotlib.widgets.Slider(ax_B, 'B', -50, 50, valinit=B)
    slider_B = matplotlib.widgets.Slider(ax_B, 'B', -50e-3, 50e-3, valinit=0.0)

    # backup procedures
    # def update_A(val):
    #     state[0] = val
    #     A, B = state
    # 
    #     big_raw = big_src * A + B
    #     big_diff = big_raw - big_med
    #     line_diff.set_ydata(big_diff[:,14])
    #     fig.canvas.draw_idle()
    # 
    # def update_B(val):
    #     state[1] = val
    #     A, B = state
    # 
    #     big_raw = big_src * A + B
    #     big_diff = big_raw - big_med
    #     line_diff.set_ydata(big_diff[:,14])
    #     fig.canvas.draw_idle()

    # def update_A(val):
    #     state[0] = val
    #     A, B = state
    #     B = mean_median - A * mean_src  # enforce zero-mean constraint
    #     state[1] = B  # update B in the state to reflect the constraint
    
    #     big_raw = big_src * A + B
    #     big_diff = big_raw - big_med
    #     line_diff.set_ydata(big_diff[:,14])
    #     #slider_B.set_val(B)  # update the slider to reflect the constraint
    #     # We need a non blocking update, were getting a recursion limit.
    #     slider_B.eventson = False
    #     slider_B.set_val(B)
    #     slider_B.eventson = True
    #     fig.canvas.draw_idle()

    # def update_B(val):
    #     state[1] = val
    #     A, B = state
    #     A = (mean_median - B) / mean_src  # enforce zero-mean constraint
    #     state[0] = A  # update A in the state to reflect the constraint

    #     big_raw = big_src * A + B
    #     big_diff = big_raw - big_med
    #     line_diff.set_ydata(big_diff[:,14])
    #     slider_A.eventson = False
    #     slider_A.set_val(A)  # update the slider to reflect the constraint
    #     slider_A.eventson = True
    #     fig.canvas.draw_idle()

    # The wavelets I'm seeing look like phase alignment issues. Lets enforce the B constraint and let the B slider refer to a dy constraint.
    # The bound of [-50,50] should be remapped to [-50e-3, 50e-3] to refer to millipixel drifts.
    ky = np.fft.fftfreq(big_src.shape[0])
    def update_A(val):
        state[0] = val
        A = state[0]
        dy = state[1]

        B = mean_median - A * mean_src  # enforce zero-mean constraint

        phasor = np.exp(-2j * np.pi * ky * dy)
        raw_shifted = np.fft.ifft(np.fft.fft(big_src[:,14]) * phasor).real
        diff = raw_shifted * A + B - big_med[:,14]
        line_diff.set_ydata(diff)
        fig.canvas.draw_idle()

    def update_B(val):
        state[1] = val
        A = state[0]
        dy = state[1]

        B = mean_median - A * mean_src  # enforce zero-mean constraint

        phasor = np.exp(-2j * np.pi * ky * dy)
        raw_shifted = np.fft.ifft(np.fft.fft(big_src[:,14]) * phasor).real
        diff = raw_shifted * A + B - big_med[:,14]
        line_diff.set_ydata(diff)
        fig.canvas.draw_idle()

    slider_A.on_changed(update_A)
    slider_B.on_changed(update_B)

    plt.show()
    globals().update(locals())



def animate_cv2_phasecorr(seq, fltr=None, prep=True, fps=-1, Min=None,Max=None):
    framenum = 0
    ema = seq[0].copy()

    lbnd = [img[:512,:512].min() for img in seq]
    ctrl = fltr(seq[0].copy() - lbnd[0])

    ky,kx = np.meshgrid(np.fft.fftfreq(ctrl.shape[0]), np.fft.fftfreq(ctrl.shape[1]), indexing='ij')

    while cv2.waitKey(1) & 0xFF != ord("q"):
        framenum = framenum % len(seq) # Handles negatives gracefully
        #if EMA:
        #    ema[:] = ema*(1-ema_alpha) + ema_alpha*seq[framenum]
        #else:
        #    ema[:] = seq[framenum]

        raw = fltr(seq[framenum] - lbnd[framenum])

        # Lets apply the above phase correlator per-frame, instead of precaching.
        A,B,dy,dx = matched_filter_phase_corr_iterator(ctrl[:512,:512], raw[:512,:512]) # Phase correlate with a section
        phasor = np.exp(-2j*np.pi*(ky*dy + kx*dx))
        raw = np.fft.ifft2(np.fft.fft2(raw)*phasor).real*A + B

        diff = raw - ema
        ema[:] = raw

        lo, hi = diff[:512,:512].min(), diff[:512,:512].max()
        Min, Max = lo, hi

        cv2_imshow(diff, title=f'tess s4 c1 ccd4 f{framenum}', prep=prep, fps=fps, Min=Min, Max=Max)
        framenum += 1





# In mathematica, how do I assign assumptions globally for the file I'm in, instead of doing:
# Assuming[{x,k,x0,t,t0,t1} \[Element] PositiveReals]?
# Copilot: In Mathematica, you can set global assumptions using the $Assumptions variable. For example, you can assign:
# $Assumptions = {x, k, x0, t, t0, t1} \[Element] PositiveReals;
# This will apply the assumptions to all subsequent computations in that session. You can also use the Assuming function to temporarily set assumptions for a specific block of code, but setting $Assumptions is the way to make them global for the entire file or session.

# for ii in range(len(nightmare_sector_spectral_fft_txy_img)):
#     src = r'C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\nightmare_volume_32' + '\\'
#     src = src + 'nightmare_img_{:02d}.png'.format(ii)
#     plt.imsave(src, nightmare_sector_spectral_fft_txy_img[ii])

def spectral_denoise(sector_data, sector_headers):

    times = np.array([header["TSTART"] for header in sector_headers])
    N = len(times)
    dt = 0.0165
    frame_times = times[0] + dt*np.arange(1570)
    w = np.fft.fftfreq(1570, dt)
    ph = np.exp(2j * np.pi * w[:,None] * times[None,:])
    F = (ph @ sector_data.reshape(1042, -1)).reshape((-1,512,512))
    print("F done")
    Q = np.sum(ph**2, axis=1)
    den = N * N - np.abs(Q) ** 2
    P_ls = 2.0 * (N * np.abs(F) ** 2 - np.real(Q[:,None,None] * np.conj(F) ** 2)) / den[:,None,None]
    print("P_ls done")
    alpha = 2.0 * (N * F - Q[:,None,None] * np.conj(F)) / den[:,None,None]
    print("alpha done")

    phasor = np.exp(-2j * np.pi * w[:,None] * frame_times[None,:])
    #phasor0 = np.exp(-2j * np.pi * w * frame_times[0])
    phasor0 = phasor[:,0]
    
    #recon_alpha = (phasor @ alpha.reshape((len(alpha), -1))).reshape((-1, 512,512))
    recon_alpha = np.fft.fft(alpha * phasor0[:,None,None], axis=0).real
    print("recon alpha done")
    #recon_nuft = (F @ alpha.reshape((len(alpha), -1))).reshape((-1, 512,512))
    recon_nuft = np.fft.fft(F * phasor0[:,None,None], axis=0).real
    print("recon nuft done")

    
    

    # Resample image to undistort
    w_tan = WCS(naxis=2)
    w_tan.wcs.crpix = w_sip.wcs.crpix.copy()
    w_tan.wcs.crval = w_sip.wcs.crval.copy()
    w_tan.wcs.cd    = w_sip.wcs.cd.copy()
    w_tan.wcs.ctype = ['RA---TAN', 'DEC--TAN']  # no SIP!
    yy, xx = np.mgrid[0:512, 0:512].astype(np.float64)

    px_out = np.column_stack([xx.ravel(), yy.ravel()])
    sky = w_tan.all_pix2world(px_out, 0)        # ideal grid → (RA, Dec)
    px_src = w_sip.all_world2pix(sky, 0)         # (RA, Dec) → where to sample in distorted frame
    map_x = px_src[:, 0].reshape(512, 512)       # source x per output pixel
    map_y = px_src[:, 1].reshape(512, 512)       # source y per output pixel

    cx, cy = 256, 256  # center of your 512×512 chunk
    shift_x = map_x[cy, cx] - cx
    shift_y = map_y[cy, cx] - cy
    map_x -= shift_x
    map_y -= shift_y
    
    n_frames = recon_nuft.shape[0]
    undistorted = np.empty((n_frames, 512, 512), dtype=np.float64)
    for i in range(n_frames):
        undistorted[i] = map_coordinates(
            recon_nuft[i],   # full frame (or at least enough to cover the map)
            [map_y, map_x],    # scipy convention: [row, col] = [y, x]
            order=3, mode='constant', cval=0.0
        )
    globals().update(locals())

"""
# Lets crop the spectra and fft back to see what happens.
yy, xx = np.indices(F_xy.shape[1:])
# Adjust so the zero index is in the center
yy = yy - F_xy.shape[1]//2
xx = xx - F_xy.shape[2]//2
rr = np.sqrt(xx**2 + yy**2)
mask = rr < 50 # Keep only low frequencies within a radius of 50 pixels
mask = np.fft.ifftshift(mask) # Shift back the mask so it aligns with the unshifted FFT
F_xy_masked = F_xy * mask[None,:,:]

while cv2.waitKey(1) & 0xFF != ord("q"):
    #ema[:] = ema*0.9 + 0.1*rcn[framenum]
    cv2_imshow(tonemap(np.fft.fftshift(np.abs(F_xy[framenum]))), title=f'tess s4 c1 ccd4 f{framenum}')
    framenum += 1
    if framenum >= len(rcn):
        framenum -= len(rcn)
"""



def test_jpl_horizons():
    from astroquery.jplhorizons import Horizons
    obj = Horizons(id='-95', location='@0',
                epochs={'start': '2018-10-19', 'stop': '2018-11-15', 'step': '30m'})
    vec = obj.vectors()
    import numpy as np
    import matplotlib.pyplot as plt
    xyzt = np.ascontiguousarray(np.array([vec['x'], vec['y'], vec['z'], vec['datetime_jd']]).T)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*xyzt.T[:3], 'o-')
    ax.set_aspect('equal')
    plt.show()

# What is ephemeris?
# Copilot: An ephemeris is a table or data file that provides the positions of astronomical objects in the sky at specific times. It typically includes information such as the object's right ascension, declination, distance from Earth, and other relevant parameters. Ephemerides are used by astronomers to track the movement of celestial bodies and to plan observations. They can be generated for planets, moons, asteroids, comets, and other objects in the solar system, as well as for stars and galaxies.
# me: Isn't this just the same as the code above? what does eph do differently to vec?
# Copilot: The `ephemerides()` method in the `Horizons` class provides a more detailed and comprehensive set of data compared to the `vectors()` method. While `vectors()` typically returns the position and velocity vectors of the object at specific times, `ephemerides()` can include additional information such as the object's apparent magnitude, phase angle, elongation, and other observational parameters. The ephemeris data is often used for planning observations and understanding the visibility of celestial objects from Earth, while the vector data is more focused on the object's trajectory and motion in space.
def test_jpl_horizons_ephemeris():
    from astroquery.jplhorizons import Horizons
    obj = Horizons(id='-95', location='@0',
                epochs={'start': '2018-10-19', 'stop': '2018-11-15', 'step': '30m'})
    eph = obj.ephemerides()
    import numpy as np
    import matplotlib.pyplot as plt
    xyzt = np.ascontiguousarray(np.array([eph['x'], eph['y'], eph['z'], eph['datetime_jd']]).T)
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*xyzt.T[:3], 'o-')
    ax.set_aspect('equal')
    plt.show()

# times = np.array([header["TSTART"] for header in sector_headers])
def query_jpl_horizons_with_given_BTJD(times, bjdrefi=2457000):
    from astroquery.jplhorizons import Horizons
    from astropy.table import vstack
    import numpy as np

    jd_times = times + bjdrefi  # BTJD → JD
    chunk_size = 50  # Horizons safe limit per request

    def _query_batched(body_id, label):
        tables = []
        for i in range(0, len(jd_times), chunk_size):
            chunk = jd_times[i:i+chunk_size].tolist()
            tables.append(Horizons(id=body_id, location='@0', epochs=chunk).vectors())
            print(f"  {label}: {min(i+chunk_size, len(jd_times))}/{len(jd_times)}")
        vec = vstack(tables)
        return np.array([vec['x'], vec['y'], vec['z'], vec['datetime_jd']]).T

    xyzt_TESS  = _query_batched('-95',  'TESS')
    xyzt_Earth = _query_batched('399',  'Earth')   # geocenter
    xyzt_Moon  = _query_batched('301',  'Moon')

    return xyzt_TESS, xyzt_Earth, xyzt_Moon

def plot_jpl_horizons_with_given_BTJD(xyzt_TESS, xyzt_Earth, xyzt_Moon):
    import matplotlib.pyplot as plt

    # ── Panel 1: Heliocentric overview (AU) ──
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*xyzt_Earth[:,:3].T, '.-', markersize=1, label='Earth')
    ax.plot(*xyzt_Moon[:,:3].T, '.-', markersize=1, label='Moon')
    ax.plot(*xyzt_TESS[:,:3].T, '.-', markersize=1, label='TESS')
    ax.set_xlabel('X (AU)'); ax.set_ylabel('Y (AU)'); ax.set_zlabel('Z (AU)')
    ax.set_aspect('equal'); ax.legend(); ax.set_title('SSB frame (AU)')

    # ── Panel 2: Geocentric (Earth radii) ──
    AU_to_Re = 23455.0  # 1 AU ≈ 23,455 R⊕
    tess_geo  = (xyzt_TESS[:,:3]  - xyzt_Earth[:,:3]) * AU_to_Re
    moon_geo  = (xyzt_Moon[:,:3]  - xyzt_Earth[:,:3]) * AU_to_Re

    fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d'))
    ax2.plot(*moon_geo.T, '.-', markersize=1, label='Moon')
    ax2.plot(*tess_geo.T, '.-', markersize=1, label='TESS')
    ax2.plot([0], [0], [0], 'o', color='blue', markersize=8, label='Earth')
    ax2.set_xlabel('X (R⊕)'); ax2.set_ylabel('Y (R⊕)'); ax2.set_zlabel('Z (R⊕)')
    ax2.set_aspect('equal'); ax2.legend(); ax2.set_title('Geocentric frame (R⊕)')

    plt.show()
    return tess_geo, moon_geo

# Meanpool an image to 256x256:
# img = sector_data[100]
# img = img.reshape(256, img.shape[0]//256, 256, img.shape[1]//256).mean(axis=(1,3))

#w_sip = WCS(sector_headers[0])
def plot_jpl_horizons_with_given_BTJD_Frustrums(xyzt_TESS, xyzt_Earth, xyzt_Moon,
                                                 crval1=33.92206207187626,
                                                 crval2=1.3757010398066702,
                                                 w_sip=None, arrow_len=15.0,
                                                 every=20, billboard_img=None,
                                                 billboard_frame=0):
    """Geocentric orbit plot with TESS view frustum quivers.
    
    crval1, crval2 : boresight RA, Dec in degrees (from FITS header)
    w_sip          : WCS object — if given, also draws 4 corner rays + billboard
    arrow_len      : quiver length in R⊕
    every          : draw a quiver every N-th epoch (keeps plot readable)
    billboard_img  : 2D array (downsampled frame) to paste as a textured quad
                     at the end of the frustum at billboard_frame.
                     e.g. tonemap(sector_data[0])[::8, ::8]
    billboard_frame: which epoch index to attach the billboard to
    """
    import matplotlib.pyplot as plt

    AU_to_Re = 23455.0
    tess_geo = (xyzt_TESS[:,:3] - xyzt_Earth[:,:3]) * AU_to_Re
    moon_geo = (xyzt_Moon[:,:3] - xyzt_Earth[:,:3]) * AU_to_Re

    # ── Boresight unit vector (ICRS Cartesian) ──
    ra  = np.deg2rad(crval1)
    dec = np.deg2rad(crval2)
    bore = np.array([np.cos(dec)*np.cos(ra),
                     np.cos(dec)*np.sin(ra),
                     np.sin(dec)])

    # ── Tangent-plane basis vectors (sky "right" and "up" in ICRS) ──
    # East  = -sin(RA), cos(RA), 0       (increasing RA direction)
    # North = -sin(Dec)cos(RA), -sin(Dec)sin(RA), cos(Dec)
    e_east  = np.array([-np.sin(ra), np.cos(ra), 0.0])
    e_north = np.array([-np.sin(dec)*np.cos(ra),
                        -np.sin(dec)*np.sin(ra),
                         np.cos(dec)])

    # ── Corner rays (if WCS provided) ──
    corners = None
    if w_sip is not None:
        nx, ny = w_sip.pixel_shape  # (NAXIS1, NAXIS2)
        corner_px = np.array([[0, 0], [nx-1, 0], [nx-1, ny-1], [0, ny-1]], dtype=np.float64)
        sky = w_sip.all_pix2world(corner_px, 0)  # (4, 2) RA, Dec in degrees
        ra_c  = np.deg2rad(sky[:, 0])
        dec_c = np.deg2rad(sky[:, 1])
        corners = np.column_stack([np.cos(dec_c)*np.cos(ra_c),
                                   np.cos(dec_c)*np.sin(ra_c),
                                   np.sin(dec_c)])  # (4, 3)

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    ax.plot(*moon_geo.T, '.-', markersize=1, label='Moon', alpha=0.4)
    ax.plot(*tess_geo.T, '.-', markersize=1, label='TESS')
    #ax.plot([0], [0], [0], 'o', color='blue', markersize=8, label='Earth')

    # ── Earth sphere (radius = 1 R⊕) ──
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0, np.pi, 16)
    ex = np.outer(np.cos(u), np.sin(v))
    ey = np.outer(np.sin(u), np.sin(v))
    ez = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(ex, ey, ez, color='dodgerblue', alpha=0.6, label='Earth')

    # ── Quiver: boresight ──
    idx = np.arange(0, len(tess_geo), every)
    pos = tess_geo[idx]
    ax.quiver(pos[:,0], pos[:,1], pos[:,2],
              bore[0]*np.ones(len(idx)),
              bore[1]*np.ones(len(idx)),
              bore[2]*np.ones(len(idx)),
              length=arrow_len, color='red', alpha=0.6, arrow_length_ratio=0.15,
              label='boresight')

    # ── Quiver: corner rays ──
    if corners is not None:
        colors = ['orange', 'gold', 'lime', 'cyan']
        for ci in range(4):
            ax.quiver(pos[:,0], pos[:,1], pos[:,2],
                      corners[ci,0]*np.ones(len(idx)),
                      corners[ci,1]*np.ones(len(idx)),
                      corners[ci,2]*np.ones(len(idx)),
                      length=arrow_len*0.7, color=colors[ci], alpha=0.3,
                      arrow_length_ratio=0)

    # ── Billboard: textured image quad at one frustum position ──
    if billboard_img is not None and w_sip is not None:
        # Image orientation from the CD matrix
        cd = w_sip.wcs.cd  # [[dRA/dpx, dRA/dpy], [dDec/dpx, dDec/dpy]]
        # Pixel-x direction in 3D = CD column 0 mapped onto tangent plane
        right_sky = cd[0, 0] * e_east + cd[1, 0] * e_north  # per-pixel-x
        up_sky    = cd[0, 1] * e_east + cd[1, 1] * e_north  # per-pixel-y
        # These are in deg/pixel — normalize to unit vectors, keep magnitude for FOV scale
        right_norm = np.linalg.norm(right_sky)
        up_norm    = np.linalg.norm(up_sky)
        right_hat  = right_sky / right_norm
        up_hat     = up_sky / up_norm

        # Billboard size: at distance arrow_len (R⊕), angular extent = npix * deg/pix
        h, w_img = billboard_img.shape[:2]
        # Full detector size in pixels (use original, not downsampled)
        nx, ny = w_sip.pixel_shape
        ds_x = nx / w_img  # downsample factor
        ds_y = ny / h
        # Half-width/height in R⊕ at arrow_len distance
        half_w = arrow_len * np.tan(np.deg2rad(nx * right_norm / 2))
        half_h = arrow_len * np.tan(np.deg2rad(ny * up_norm / 2))

        # Quad center
        origin = tess_geo[billboard_frame] + bore * arrow_len
        # Mesh grid for the quad
        ss = np.linspace(-1, 1, w_img)
        tt = np.linspace(-1, 1, h)
        S, T = np.meshgrid(ss, tt)
        QX = origin[0] + S * half_w * right_hat[0] + T * half_h * up_hat[0]
        QY = origin[1] + S * half_w * right_hat[1] + T * half_h * up_hat[1]
        QZ = origin[2] + S * half_w * right_hat[2] + T * half_h * up_hat[2]

        # Colormap the image for facecolors (needs RGBA, shape h-1 x w-1 x 4)
        if billboard_img.ndim == 2:
            rgba = plt.get_cmap('gray')(billboard_img / billboard_img.max())
        else:
            rgba = billboard_img.astype(np.float64)
            if rgba.max() > 1.5:  # uint8 0-255 → normalize to [0,1]
                rgba = rgba / 255.0
        # facecolors needs (M-1, N-1, 3or4) — one color per face, not per vertex
        fc = rgba[:-1, :-1]
        if fc.shape[-1] == 3:
            fc = np.concatenate([fc, np.ones((*fc.shape[:2], 1))], axis=-1)

        ax.plot_surface(QX, QY, QZ, facecolors=fc, shade=False, alpha=0.9,
                        rstride=1, cstride=1)

    ax.set_xlabel('X (R⊕)'); ax.set_ylabel('Y (R⊕)'); ax.set_zlabel('Z (R⊕)')
    ax.set_aspect('equal'); ax.legend(); ax.set_title('Geocentric + TESS FOV')
    plt.show()
    return tess_geo, moon_geo


def identify_streak(w_sip, tess_geo_0, tess_geo_1, tess_vel_0, tess_vel_1,
                    t0, t1, bjdrefi=2457000,
                    px_start_0=None, px_end_0=None,
                    px_start_1=None, px_end_1=None):
    """Constrain orbital parameters of a streak seen in two consecutive TESS frames.
    
    Parameters
    ----------
    w_sip        : WCS with SIP distortion
    tess_geo_0/1 : (3,) geocentric position of TESS at frames 0 and 1, in R⊕
    tess_vel_0/1 : (3,) geocentric velocity of TESS at frames 0 and 1, in R⊕/day
    t0, t1       : BTJD times of the two frames
    px_start_0   : [x, y] pixel coords of streak start in frame 0 (or None)
    px_end_0     : [x, y] pixel coords of streak end in frame 0
    px_start_1   : [x, y] pixel coords of streak start in frame 1
    px_end_1     : [x, y] pixel coords of streak end in frame 1 (or None)
    """
    Re_km = 6371.0  # km per R⊕

    dt_days = t1 - t0
    dt_sec  = dt_days * 86400.0
    print(f"Δt = {dt_days:.6f} days = {dt_sec:.1f} s = {dt_sec/60:.1f} min")

    # ── TESS state ──
    r_tess = (tess_geo_0 + tess_geo_1) / 2  # mean position R⊕
    v_tess = (tess_vel_0 + tess_vel_1) / 2  # mean velocity R⊕/day
    dist_tess = np.linalg.norm(r_tess)
    speed_tess = np.linalg.norm(v_tess) * Re_km / 86400  # km/s
    print(f"TESS geocentric distance: {dist_tess:.1f} R⊕ = {dist_tess*Re_km:.0f} km")
    print(f"TESS speed: {speed_tess:.2f} km/s")

    # ── Convert pixel endpoints to RA/Dec ──
    endpoints = {}
    for label, px in [('start_0', px_start_0), ('end_0', px_end_0),
                      ('start_1', px_start_1), ('end_1', px_end_1)]:
        if px is not None:
            rd = w_sip.all_pix2world([px], 0)[0]  # [RA, Dec] in degrees
            ra_r, dec_r = np.deg2rad(rd)
            uvec = np.array([np.cos(dec_r)*np.cos(ra_r),
                             np.cos(dec_r)*np.sin(ra_r),
                             np.sin(dec_r)])
            endpoints[label] = {'px': px, 'radec': rd, 'uvec': uvec}
            print(f"  {label}: px={px}  →  RA={rd[0]:.4f}° Dec={rd[1]:.4f}°")

    # ── Intra-frame angular rate (streak length / exposure time) ──
    for fi, (s, e) in [(0, ('start_0', 'end_0')), (1, ('start_1', 'end_1'))]:
        if s in endpoints and e in endpoints:
            dot = np.clip(np.dot(endpoints[s]['uvec'], endpoints[e]['uvec']), -1, 1)
            ang_deg = np.rad2deg(np.arccos(dot))
            rate = ang_deg / (dt_sec / 3600)  # deg/hr (exposure ≈ dt)
            print(f"  Frame {fi} streak: {ang_deg:.3f}° → {rate:.2f}°/hr apparent")

    # ── Inter-frame motion (midpoint to midpoint) ──
    mid0, mid1 = None, None
    if 'start_0' in endpoints and 'end_0' in endpoints:
        mid0 = (endpoints['start_0']['uvec'] + endpoints['end_0']['uvec'])
        mid0 /= np.linalg.norm(mid0)
    elif 'end_0' in endpoints:
        mid0 = endpoints['end_0']['uvec']
    if 'start_1' in endpoints and 'end_1' in endpoints:
        mid1 = (endpoints['start_1']['uvec'] + endpoints['end_1']['uvec'])
        mid1 /= np.linalg.norm(mid1)
    elif 'start_1' in endpoints:
        mid1 = endpoints['start_1']['uvec']

    if mid0 is not None and mid1 is not None:
        dot = np.clip(np.dot(mid0, mid1), -1, 1)
        inter_ang = np.rad2deg(np.arccos(dot))
        inter_rate = inter_ang / (dt_days * 24)  # deg/hr
        print(f"  Inter-frame midpoint shift: {inter_ang:.3f}° → {inter_rate:.2f}°/hr")

    # ── Angular rate from TESS's own motion (parallax on a static object) ──
    # For an object at distance d from TESS, TESS's own transverse velocity
    # creates an apparent angular rate of v_TESS_perp / d
    los = mid0 if mid0 is not None else mid1
    if los is not None:
        v_tess_kms = v_tess * Re_km / 86400  # km/s
        v_perp = v_tess_kms - np.dot(v_tess_kms, los) * los
        v_perp_mag = np.linalg.norm(v_perp)
        print(f"  TESS transverse velocity (wrt LOS): {v_perp_mag:.2f} km/s")

    # ── Distance estimates for different orbital regimes ──
    print("\n── Orbital shell analysis ──")
    shells = [
        ("LEO (400km)",     6771,  7.67),
        ("MEO (20200km)",  26571,  3.87),
        ("GEO (35786km)",  42164,  3.07),
        ("Cislunar (L1)",  326000, 0.90),
        ("Lunar orbit",    384400, 1.02),
    ]

    # Closest approach of LOS to Earth
    d_min_tess = -np.dot(r_tess, los)  # distance along LOS to closest approach
    closest_Re = np.sqrt(np.dot(r_tess, r_tess) - d_min_tess**2) if d_min_tess > 0 else np.linalg.norm(r_tess)
    print(f"  LOS closest approach to Earth: {closest_Re:.1f} R⊕ = {closest_Re*Re_km:.0f} km")

    for name, r_obj_km, v_obj_kms in shells:
        r_obj_Re = r_obj_km / Re_km
        # Solve |r_tess + d*los|² = r_obj² for d
        A = 1.0
        B = 2.0 * np.dot(r_tess, los)
        C = np.dot(r_tess, r_tess) - r_obj_Re**2
        disc = B**2 - 4*A*C
        if disc < 0:
            print(f"  {name:20s}: LOS misses by {(closest_Re - r_obj_Re)*Re_km:.0f} km")
            continue
        d1 = (-B - np.sqrt(disc)) / 2
        d2 = (-B + np.sqrt(disc)) / 2
        dists = [d for d in [d1, d2] if d > 0]
        if not dists:
            print(f"  {name:20s}: shell is behind TESS")
            continue
        for d_Re in dists:
            d_km = d_Re * Re_km
            # v/d [km/s / km] = rad/s; × (180/π × 3600) = × 206265 → deg/hr
            expected_rate = (v_obj_kms + v_perp_mag) / d_km * 206265  # deg/hr
            print(f"  {name:20s}: d={d_km:>10.0f} km,  "
                  f"expected ~{expected_rate:.2f}°/hr")

    # ── Reverse solve: what distance from TESS matches the observed rate? ──
    # Measure from intra-frame streak
    measured_rates = []
    for fi, (s, e) in [(0, ('start_0', 'end_0')), (1, ('start_1', 'end_1'))]:
        if s in endpoints and e in endpoints:
            dot = np.clip(np.dot(endpoints[s]['uvec'], endpoints[e]['uvec']), -1, 1)
            ang_deg = np.rad2deg(np.arccos(dot))
            rate = ang_deg / (dt_sec / 3600)  # deg/hr
            measured_rates.append(rate)

    if measured_rates and v_perp_mag > 0:
        obs_rate = np.mean(measured_rates)
        # observed_rate [deg/hr] = v_rel / d * 206265
        # → d = v_rel / (obs_rate / 206265)
        # Lower bound: just TESS parallax (object stationary)
        d_parallax_km = v_perp_mag / (obs_rate / 206265)
        d_parallax_Re = d_parallax_km / Re_km
        obj_pos = r_tess + (d_parallax_Re) * los
        obj_geocentric_Re = np.linalg.norm(obj_pos)
        obj_alt_km = (obj_geocentric_Re - 1) * Re_km
        print(f"\n── Reverse solve (observed {obs_rate:.2f}°/hr) ──")
        print(f"  If object is STATIONARY (parallax-only):")
        print(f"    d from TESS: {d_parallax_km:.0f} km")
        print(f"    Geocentric r: {obj_geocentric_Re:.1f} R⊕ = {obj_geocentric_Re*Re_km:.0f} km")
        print(f"    Altitude: {obj_alt_km:.0f} km")
        # Try a few orbital velocities
        for v_guess, label in [(3.07, "GEO-like ~3 km/s"), (7.0, "LEO-like ~7 km/s"),
                                (1.0, "slow ~1 km/s"), (20.0, "asteroid ~20 km/s")]:
            d_km = (v_perp_mag + v_guess) / (obs_rate / 206265)
            d_Re = d_km / Re_km
            op = r_tess + d_Re * los
            r_geo = np.linalg.norm(op)
            print(f"  If v_obj≈{v_guess} km/s ({label}):")
            print(f"    d from TESS: {d_km:.0f} km,  geocentric r: {r_geo:.1f} R⊕ = {r_geo*Re_km:.0f} km")

    # ── JD for database lookup ──
    jd0 = t0 + bjdrefi
    jd1 = t1 + bjdrefi
    print(f"\n── For Space-Track / Celestrak lookup ──")
    print(f"  JD range: {jd0:.6f} – {jd1:.6f}")
    from datetime import datetime, timedelta
    epoch = datetime(2000, 1, 1, 12, 0, 0)  # J2000.0 = JD 2451545.0
    dt0 = epoch + timedelta(days=jd0 - 2451545.0)
    dt1 = epoch + timedelta(days=jd1 - 2451545.0)
    print(f"  UTC range: {dt0.strftime('%Y-%m-%d %H:%M:%S')} – {dt1.strftime('%Y-%m-%d %H:%M:%S')}")
    if los is not None:
        ra_los  = np.rad2deg(np.arctan2(los[1], los[0])) % 360
        dec_los = np.rad2deg(np.arcsin(los[2]))
        print(f"  Streak direction: RA≈{ra_los:.2f}° Dec≈{dec_los:.2f}°")

# LEO_scale = (400+6371)/6371
# MEO_scale = (20200+6371)/6371
# GEO_scale = (35786+6371)/6371
# ax.plot_surface(ex*LEO_scale, ey*LEO_scale, ez*LEO_scale, color='dodgerblue', alpha=0.6, label='LEO')
# ax.plot_surface(ex*MEO_scale, ey*MEO_scale, ez*MEO_scale, color='dodgerblue', alpha=0.6, label='MEO')
# ax.plot_surface(ex*GEO_scale, ey*GEO_scale, ez*GEO_scale, color='dodgerblue', alpha=0.6, label='GEO')











cu_code += r'''

// Lets find a way to do median sorts on the GPU to accelerate:
// group_median = np.median(sector_data[max(0,i-3):][:8], axis=0)
// sector_data is an array of size (N_frames, 2048, 2048). Our median is applied per-pixel, so this feels like something that can be done via gpu compute, especially since this is only 8 frames at a time.
// Lets assume each frame is stored in its own 1d array

// Encode one frame into the given place in the median buffer to make the filter fast.
__global__
void encode_frame_into_median_buffer(
    double* __restrict__ median_frame,
    const double* __restrict__ frame,
    int N_frames, int height, int width, int frame_idx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;
    if(total_pixels <= idx) return;

    // Encodes pixels into contiguous memory. I hope this is faster than a huge stride.
    median_frame[idx*8 + frame_idx] = frame[idx];
}

__device__ __forceinline__
void SWAP(double* arr, int a, int b) {
    if(arr[b] < arr[a]) {
        double temp = arr[a];
        arr[a] = arr[b];
        arr[b] = temp;
    }
}

__global__
void median_filter_3d(
    double* __restrict__ median_frame,
    const double* __restrict__ frames,
    int N_frames, int height, int width
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = height * width;
    if(total_pixels <= idx) return;

    // Load the 8 values for this pixel across the frames
    double pixel_values[8];
    for(int i=0; i<8; i++)
        pixel_values[i] = median_frame[idx*8 + i];

    // Sort the 8 values to find the median
    // Sorting networks / Bose-Nelson Algorithm since the number of elements is tiny.
    double *px = pixel_values; // alias for readability
    SWAP(px, 0, 1); SWAP(px, 2, 3); SWAP(px, 4, 5); SWAP(px, 6, 7);
    SWAP(px, 0, 2); SWAP(px, 1, 3); SWAP(px, 4, 6); SWAP(px, 5, 7);

    SWAP(px, 1, 2); SWAP(px, 5, 6); SWAP(px, 0, 4); SWAP(px, 3, 7);
    SWAP(px, 1, 5); SWAP(px, 2, 6); SWAP(px, 1, 2); SWAP(px, 5, 6);
    SWAP(px, 3, 5); SWAP(px, 2, 4); SWAP(px, 3, 4);

    // The median is the average of the 4th and 5th elements (0-indexed)
    median_frame[idx] = 0.5 * (px[3] + px[4]);
}

'''


































# Lets find a way to have python report what in its ram is consuming all my memory. I need to manually clean it.
# def report_ram():
#     import psutil
#     process = psutil.Process()
#     mem_info = process.memory_info()
#     print(f"RAM usage: {mem_info.rss / (1024**3):.2f} GB")
# Oh, I mean per global scope variable.
def report_ram_for_all_symbols():
    import psutil
    import sys
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Total RAM usage: {mem_info.rss / (1024**3):.2f} GB")
    print("RAM usage by global variables:")
    items = sorted(globals().items(), key=lambda item: sys.getsizeof(item[1]), reverse=True)
    # Cull items with "0.0MB"
    items = [(name, obj) for name, obj in items if sys.getsizeof(obj) > 0]

    for name, obj in items[:25]:
        size_mb = sys.getsizeof(obj) / (1024**2)
        print(f"  {name}: {size_mb:.2f} MB")
    total_size = sum([sys.getsizeof(obj) / (1024**2) for name, obj in items if sys.getsizeof(obj) > 0])
    print(f"{total_size = }")



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
    DISPLAY_prev_frame = None
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

                if DISPLAY_prev_frame is not None:
                    # sector, camera, ccd
                    delta = algn - DISPLAY_prev_frame
                    cv2_imshow(tonemap(delta), title=f'tess s{sector} c{camera} ccd{ccd} f{ii}')
                DISPLAY_prev_frame = algn.copy()

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

rawdir = ROOT / "TESS_sector_19"
clndir = ROOT / "DIA_TEMP" / "clean_s19"

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

    rawdir = ROOT / "TESS_sector_19"
    clndir = ROOT / "DIA_TEMP" / "clean_s19"

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
# def tonemap(img):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         #ld = np.log(img)
#         ld = slog(img)
#     ld_fltr = ld[~np.isnan(ld) & ~np.isinf(ld)]
#     ld_nan = np.nan_to_num(ld, nan=np.nanmax(ld_fltr), neginf=np.nanmin(ld_fltr), posinf=np.nanmax(ld_fltr))
#     histo, bin_edges = np.histogram(ld_nan, bins=4096)
#     bins = (bin_edges[:-1] + bin_edges[1:]) / 2
#     cs_histo = np.cumsum(histo)
#     cdf_histo = cs_histo.astype(np.float64) / cs_histo[-1].astype(np.float64)
#     ld_mapped = np.interp(ld_nan, bins, cdf_histo)
#     return ld_mapped

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


"""
Alright let’s see if we can actually build the model! For the sake of argument, let’s assume the psf *does not exist,* and instead it’s just weird trajectory flow. This leads us to an actual, actionable math object we can tinker with.

ℒ = ∫∫(L - img)²/2 dx 
This loss is the one I’ve been dancing around for a while. I have it written as an integral transform for simplicity, but it’s time we relax it. Also, it’s wrong. One second:

ℒ = ∫∫∫(L - img)²/2 dx dt
There we go! I suspect the full spectral form will emerge in the math. The math will tell us what we should choose here.

Let’s first substitute the integrals with their true forms, a triple sum of complex phasors. I’ll be a bit loose here with the spatial indices, but I'll be very picky with the temporal ones since they're non-uniform.

For any given index, lets use lower and uppercase to denote index and limit. So, time is encoded by t of T, and space by x of X, and y of Y. I'll omit T, X, and Y unless necissary for brevity. I may commit fully to einsums if its clear.

ℒ → ∑ₜ ∑ₓ ∑ᵧ (Lₜₓᵧ - imgₜₓᵧ)²/2 → (Lₜₓᵧ - imgₜₓᵧ)(Lᵗˣʸ - imgᵗˣʸ)/2
∂ℒ = ∂((Lₜₓᵧ - imgₜₓᵧ)(Lᵗˣʸ - imgᵗˣʸ))/2 = (Lᵗˣʸ - imgᵗˣʸ) ∂Lₜₓᵧ

Our atom that we developed earlier used k as our spatial frequency. Due to unicode limits, lets assign u,v as kx, ky. So our atom is:
exp(-𝒾([u, v]•([x₀,y₀] + [ẋ₀, ẏ₀]t) - ω₀t), which corresponded to the spatial fourier transform of a trajectory.  It should be recoverable as our model from fixed frequencies. Our input atom is:

Lₜₓᵧ = Fₜₓᵧ exp(𝒾ω₀tₜ) exp(-(1/2)(([xₓ,yᵧ] - ([x₀,y₀] + [ẋ₀, ẏ₀]t))²/σ))/(√(2π)σ)

Where in practice, we let σ → 0 to get a dirac delta distrbution. We'll leave it be for now, and see if the discrete math allows us to zero it out, or if it has a minima. Our underlying data is the *infinite series* of these atoms, which we don't have. This assumes an *input* of the latent space [x₀,y₀, ẋ₀, ẏ₀, ω₀, φ₀], which we use as our Radon-like MLE. φ₀ is the phase offset of exp(𝒾ω₀tₜ). It takes the form of τ in Lomb-Scargle.

Thinking about this more, I think we should sum k of K atoms per pixel per time to get a more accurate model by incorporating superpositioning:

Lₜₓᵧ = Σₖ Fₜₓᵧₖ exp(𝒾ωₖtₜ) exp(-(1/2)(([xₓ,yᵧ] - ([xₖ,yₖ] + [ẋₖ, ẏₖ]t))²/σ))/(√(2π)σ)
Lₜₓᵧ = Σₖ Fₜₓᵧₖ Zᵗˣʸᵏ, for brevity.

We only have one latent space variable here, the flux F.

∂Lₜₓᵧ = Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ
∂ℒ = (Lᵗˣʸ - imgᵗˣʸ) ∂Lₜₓᵧ = ∑ₜ ∑ₓ ∑ᵧ (Lᵗˣʸ - imgᵗˣʸ) Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ = 0
∑ₜ ∑ₓ ∑ᵧ Lₜₓᵧ Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ = ∑ₜ ∑ₓ ∑ᵧ imgₜₓᵧ Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ

This is normally the part where I get stuck, but lets take a proper fourier series along the x and y axes. Lets assign u of X, v of Y as our variables here, overloading X and Y as limits.

qᵤᵥˣʸ = exp(-𝒾2π(ux/X + vy/Y))
Σᵤ Σᵥ ∑ₜ ∑ₓ ∑ᵧ Lₜₓᵧ Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ = ∑ₜ ∑ₓ ∑ᵧ imgₜₓᵧ Σₖ ∂Fₜₓᵧₖ Zᵗˣʸᵏ





This corresponds to the pure 

Lₜₓᵧ = ∑ᵤ ∑ᵥ Aᵤᵥ exp(-𝒾([u, v]•([x₀,y₀] + [ẋ₀, ẏ₀]t) - ω₀t))

Scratch:
∂ ∇ ☐
ⁱʲᵏᵢⱼₖⁿᵐₙₘ
≈ ≡ ≅ ≠ ≤ ≥ ⊂ ⊆ ∈
± ∓ ∧ ∨
∫ dt ∈(-∞, ∞)
← → ⇒
• × ⊗ ⋆
ℕ ℤ ℝ ℂ ℍ
ℒ 𝓗 ℱ 𝒩
𝒾 𝒿 𝓀
₀₁₂₃₄₅₆₇₈₉ Lower
⁰¹²³⁴⁵⁶⁷⁸⁹ Upper
⁻¹ †
ₜₓᵧᵗˣʸ ᵤᵥ ᵘᵛ
ẋ ẏ
"""



if False:
    url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
    params = {
        "TIME": "2458410.922835747",
        "RA": 33.92206207187626,
        "DEC": 1.3757010398066702,
        "SR": 8.402405260871141,
        "RESPONSEFORMAT": "json",
        "VERB": 3,
        "-observer": "500",
        "-objFilter": "100",
    }

    r = requests.get(url, params=params, timeout=120)
    print(r.status_code)
    print(r.url)
    print(r.text[:4000])

    url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
    params = {
        "TIME": "2458410.922835747",
        "RA": 33.92206207187626,
        "DEC": 1.3757010398066702,
        "SR": 8.402405260871141,
        "RESPONSEFORMAT": "json",
        "VERB": 3,
        "-observer": "500",
        "-objFilter": "100",
    }

    r = requests.get(url, params=params, timeout=120)
    print(r.status_code)
    print(r.url)
    print(r.text[:4000])# Get data from skybotasteroid  for each frame

# Get asteroid data from skybot for each frame
import requests
import astropy.units
import io
import zipfile
from astropy.coordinates import SkyCoord


import requests

import astropy.units

def skybot_row_to_skycoord(row):
    """
    row is one dict from the SkyBoT JSON list.
    Uses the sexagesimal strings SkyBoT returns.
    """
    ra_str  = row["RA (hms)"]
    dec_str = row["DEC (dms)"]
    return SkyCoord(ra_str, dec_str, unit=(astropy.units.hourangle, astropy.units.deg), frame="icrs")

def skycoord_to_tess_pixel(wcs, coord, origin=0):
    """
    origin=0 gives NumPy-style pixel coords.
    origin=1 gives FITS-style pixel coords.
    """
    x, y = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, origin)
    return float(x), float(y)

def skybot_json_to_pixels(js, wcs, shape=None, origin=0):
    """
    js: list of dicts from SkyBoT
    wcs: astropy WCS
    shape: optional (ny, nx) to flag on-chip objects
    """
    rows = []

    ra_array = []
    de_array = []

    for row in js:
        coord = skybot_row_to_skycoord(row)
        ra_array.append(coord.ra.deg)
        de_array.append(coord.dec.deg)
    ra_array = np.array(ra_array)
    de_array = np.array(de_array)
    x_array, y_array = wcs.all_world2pix(ra_array, de_array, origin)

    for i, row in enumerate(js):
        #coord = skybot_row_to_skycoord(row)
        #x, y = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, origin)

        out = dict(row)
        out["ra_deg"] = ra_array[i]
        out["dec_deg"] = de_array[i]
        out["x_pix"] = float(x_array[i])
        out["y_pix"] = float(y_array[i])

        if shape is not None:
            ny, nx = shape
            out["on_chip"] = (0 <= x_array[i] < nx) and (0 <= y_array[i] < ny)
        else:
            out["on_chip"] = None

        rows.append(out)

    return rows

# all_rows = []
# total_size = 0
# t0 = time.time()
# t1s = []
# for n_file, in_file in enumerate(files):
#     data, header = fits.getdata(in_file, header=True)
#     wcs = WCS(header)

#     t_mid = 0.5*(header['TSTART']+header['TSTOP']) + 2457000

#     url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
#     params = {
#         "TIME": str(t_mid), #"2458410.922835747",
#         "RA": 33.92206207187626,
#         "DEC": 1.3757010398066702,
#         "SR": 8.402405260871141,
#         "RESPONSEFORMAT": "json",
#         "VERB": 3,
#         "-observer": "500",
#         "-objFilter": "100",
#     }

#     r = requests.get(url, params=params, timeout=120)
#     print(r.status_code)
#     #print(r.url)
#     #print(r.text[:4000])

#     js = json.loads(r.text)
#     idx = sorted(js, key=lambda x: x['VMag (mag)'])
#     rows = skybot_json_to_pixels(idx, w, shape=(ny, nx), origin=0)

#     # Write to a bytesio buffer
#     #output = io.BytesIO()
#     #with zipfile.ZipFile(output, 'w') as zf:
#     #    zf.writestr('skybot_results.json', json.dumps(rows, indent=2))

#     # Save the JSON to all_rows, write to disk later. 6gb of ram should be fine.
#     all_rows.append((t_mid, rows))
#     #all_rows.append((t_mid, js))
#     total_size += r.text.__sizeof__()
#     # Write an index like:
#     # [   1]
#     # [1007]
#     print(f"[{(n_file+1):4d}/{len(files):4d}] Got {len(rows)} objects from SkyBoT, total size so far: {total_size/1e6:.2f} MB")
# t0 = time.time())
    
    
#         t0 = time.time()1, time elapsed: {t1 - t0:.2f} s
    
# t1s = []]
#     t12s.    t1s.append(t1 - t0)
#     iueta = []""[]?N/A"
#     if len()t1s >2 <= :    if 2 <= len(t1s):
#         coeffs = np.polyfit(np.arange(len(t1s)), t1s, 1)
#         eta_seconds = np.polyval(coeffs, len(files)) - t1s[-1]
#         eta = humanize.precisedelta(eta_seconds, format="%0.3f")
#         eta = f"[ETA: {eta}]" {eta}

# Incomplete download:
#  ConnectionResetError: [WinError 10054] An existing connection was forcibly closed by the remote hostLets cache what we have: [ 275/1042] Got 9549 objects from SkyBoT, total size so far: 1798.53 MB, time elapsed: 5235.96 s [ETA: 3 hours, 57 minutes and 19 seconds]
# # zipfile!
# with zipfile.ZipFile('skybot_results.zip', 'w') as zf:
#     for i, (t_mid, rows) in enumerate(all_rows):
#         filename = f'skybot_results_{i:04d}.json'
#         zf.writestr(filename, json.dumps(rows, indent=2))
    
#         # Lets inbclude a metadata.json file too.
#     #that maps t_mid to an indexa  filename.
#     meta =    metadata = [{"t_mid": t_mid, "filename": f'skybot_results_{i:04d}.json'} for i, (t_mid, rows) in enumerate(all_rows)]    metadata = [{"t_mid": t_mid, "filename": f'skybot_results_{i:04d}.json'} for i, (t_mid, rows) in enumerate(all_rows)]
#     zf.writestr('metadata.json', json.dumps(metadata, indent=2))

# # Now lets sort theach js by their magnitude!!
# #all_rows = all_rows = [(t_mid, sorted(rows, key=lambda x: x['VMag (mag)'])) for t_mid, rows in all_rows]

# # Now lets update each rowrow to include the data we get from rows = skybot_json_to_pixels(idx, w, shape=(ny, nx), origin=0).


# for n_file, in_file in enumerate(files):
#     data, header = fits.getdata(in_file, header=True)
#     wcs = WCS(header)

#     t_mid = 0.5*(header['TSTART']+header['TSTOP']) + 2457000

#     url = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"
#     params = {
#         "TIME": str(t_mid), #"2458410.922835747",
#         "RA": 33.92206207187626,
#         "DEC": 1.3757010398066702,
#         "SR": 8.402405260871141,
#         "RESPONSEFORMAT": "json",
#         "VERB": 3,
#         "-observer": "500",
#         "-objFilter": "100",
#     }

#     r = requests.get(url, params=params, timeout=120)
#     print(r.status_code)
#     #print(r.url)
#     #print(r.text[:4000])

#     js = json.loads(r.text)
#     idx = sorted(js, key=lambda x: x['VMag (mag)'])
#     rows = skybot_json_to_pixels(idx, w, shape=(ny, nx), origin=0)
# t0 = time.time()
# t1s = []
    
#     t1 = time.time()
#     t1s.append(t1 - t0)
#     eta = "[N/A]"
#     if 2 <= len(t1s):
#         coeffs = np.polyfit(np.arange(len(t1s)), t1s, 1)
#         eta_seconds = np.polyval(coeffs, len(files)) - t1s[-1]
#         eta = humanize.precisedelta(eta_seconds, format="%0.3f")
#         eta = f"[ETA: {eta}]"

#     print(f"[{(n_file+1):4d}/{len(files):4d}] Got {len(rows)} objects from SkyBoT, total size so far: {total_size/1e6:.2f} MB, time elapsed: {t1 - t0:.2f} s {eta}")cs)data.shapejsjs
#     js =     js = all_rows[n_file][1] # This is the sorted list of dicts we got from skybot_json_to_pixels, but without the pixel coordinates. Lets add those in.js)((, t_m()t_mid, for n_file, (in_file, (t_mid, rows)) in enumerate(files):for n_file, (in_file, (t_mid, rows)) in enumerate(zip(files, all_rows)):rowsrows
#     all_ro    all_rows[n_file] = (t_mid, rows)rowsall_rows# Lets try something else.
# # Lets first get a list of all keys objectss that appear in all frames.


#every original dataset.
# dict_keys(['Num', 'Name', 'RA (hms)', 'DEC (dms)', 'Class', 'VMag (mag)', 'Err (arcsec)', 'd (arcsec)', 'dRA (arcsec/h)', 'dDEC (arcsec/h)', 'dg (ua)', 'dh (ua)', 'Phase (deg)', 'SunElong (deg)', 'position (au)', 'velocity (au/d)', 'ref_epoch', 'ssodnet', 'ra_deg', 'dec_deg', 'x_pix', 'y_pix', 'on_chip'])
# We should also append the t_mid to each row for easier access later.

if True:
    t0 = time.time()
    t1s = []
    all_names = set()
    for n_file, (t_mid, rows) in enumerate(all_rows):
        names = set()
        for row in rows:
            try:
                name = row['Name']
                if name in names:
                    print("Warning: Duplicate name found in the same file:", name)
                names.add(name)
            except KeyError as e:
                print("Warning: 'Name' key not found in row:", row)
                raise e
        all_names.update(names)

        t1 = time.time()
        t1s.append(t1 - t0)
        eta = "[N/A]"
        if 2 <= len(t1s):
            coeffs = np.polyfit(np.arange(len(t1s))[-10:], t1s[-10:], 1)
            eta_seconds = np.polyval(coeffs, len(files)) - t1s[-1]
            eta = humanize.precisedelta(eta_seconds, format="%0.3f")
            eta = f"[ETA: {eta}]"

        print(f"[{(n_file+1):4d}/{len(all_rows):4d}] time elapsed: {t1 - t0:.2f} s {eta}")


# Names were unique, nums were not.
# There appears to be 10705 unique names. 
# For each name, lets build a numpy array of a trajectory.
# dict_keys(['Num', 'Name', 'RA (hms)', 'DEC (dms)', 'Class', 'VMag (mag)', 'Err (arcsec)', 'd (arcsec)', 'dRA (arcsec/h)', 'dDEC (arcsec/h)', 'dg (ua)', 'dh (ua)', 'Phase (deg)', 'SunElong (deg)', 'position (au)', 'velocity (au/d)', 'ref_epoch', 'ssodnet', 'ra_deg', 'dec_deg', 'x_pix', 'y_pix', 'on_chip'])
# ->
# First, the const metadata.
# Name: Num, Class

# Next, the time series.
# Name: times, RA (deg), DEC (deg), VMag (mag), Err (arcsec), d (arcsec), dRA (arcsec/h), dDEC (arcsec/h), dg (ua), dh (ua), Phase (deg), SunElong (deg), position (au) (x,y,z), velocity (au/d) (vx, vy, vz), x_pix, y_pix, on_chip

space_rock_columns = ['times', 'x_pix', 'y_pix', 'ra_deg', 'dec_deg', 'VMag (mag)', 'Err (arcsec)', 'd (arcsec)', 'dRA (arcsec/d)', 'dDEC (arcsec/d)', 'dg (ua)', 'dh (ua)', 'Phase (deg)', 'SunElong (deg)', 'x_position (au)', 'y_position (au)', 'z_position (au)', 'vx_velocity (au/d)', 'vy_velocity (au/d)', 'vz_velocity (au/d)', 'on_chip']

metadata = {}
space_rocks = {}

for n_file, (t_mid, rows) in enumerate(all_rows):
    for row in rows:
        name = row['Name']
        if name not in metadata:
            metadata[name] = {
                "Num": row.get("Num", "n/a"),
                "Class": row.get("Class", "n/a"),
                "Count": 0, # Placeholder, will update later
            }
        metadata[name]['Count'] += 1

for name in list(metadata.keys())[:10]:
    print(name, metadata[name])

with open(r'C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\TESS_sector_4_asteroids\space_rock_metadata.txt', 'w') as f:
    for name, meta in metadata.items():
        f.write(f"{name}: {meta}\n")

@numba.njit(inline="always")
def char_to_float(c):
    return float(ord(c) - ord('0'))

@numba.njit(inline="always")
def char_to_int(c):
    return ord(c) - ord('0')

# '02 32 17.1114' -> 38.0712975 
@numba.njit
def parse_ra_array(ra):
    # Thankfully, the astronomers gave me a GREAT format of fixed length. It should be easy to write a c-like formatter. ra is a list.
    result = np.empty(len(ra), dtype=np.float64)
    for i in numba.prange(len(ra)):
        # h = float(ra[i][0:2])
        # m = float(ra[i][3:5])
        # s = float(ra[i][6:])
        # result[i] = 15*(h + m/60 + s/3600)
        # This is ideal, but numba is complaining it can't convert to float. Lets do it manually by matching keys to values.
        h = char_to_int(ra[i][0]) * 10 + char_to_int(ra[i][1])
        m = char_to_int(ra[i][3]) * 10 + char_to_int(ra[i][4])
        s = char_to_int(ra[i][6]) * 10000 + char_to_int(ra[i][7]) * 1000 + char_to_int(ra[i][8]) * 100 + char_to_int(ra[i][9]) * 10 + char_to_int(ra[i][10])
        result[i] = 15*(h + m/60 + s/360000)
    return result

@numba.njit
def parse_dec_array(dec):
    # Thankfully, the astronomers gave me a GREAT format of fixed length. It should be easy to write a c-like formatter. dec is a list.
    result = np.empty(len(dec), dtype=np.float64)
    for i in numba.prange(len(dec)):
        sign = -1 if dec[i][0] == '-' else 1
        # d = float(dec[i][1:3])
        # m = float(dec[i][4:6])
        # s = float(dec[i][7:])
        # result[i] = sign*(d + m/60 + s/3600)
        d = char_to_int(dec[i][1]) * 10 + char_to_int(dec[i][2])
        m = char_to_int(dec[i][4]) * 10 + char_to_int(dec[i][5])
        s = char_to_int(dec[i][7]) * 10000 + char_to_int(dec[i][8]) * 1000 + char_to_int(dec[i][9]) * 100 + char_to_int(dec[i][10]) * 10 + char_to_int(dec[i][11])
        result[i] = sign*(d + m/60 + s/360000)
    return result


if True:
    @numba.njit(inline='always')
    def _parse_unsigned_int_until(s, i, stop_char):
        val = 0
        n = len(s)
        while i < n and s[i] != stop_char:
            c = ord(s[i]) - 48
            val = val * 10 + c
            i += 1
        return val, i

    @numba.njit(inline='always')
    def _parse_frac_rest(s, i):
        frac = 0.0
        scale = 1.0
        n = len(s)
        while i < n:
            c = ord(s[i]) - 48
            frac = frac * 10.0 + c
            scale *= 10.0
            i += 1
        return frac / scale

    @numba.njit(inline='always')
    def parse_ra_hms_deg_scalar(s):
        # "HH MM SS.ssss"
        i = 0

        hh, i = _parse_unsigned_int_until(s, i, ' ')
        i += 1
        mm, i = _parse_unsigned_int_until(s, i, ' ')
        i += 1

        ss = 0
        n = len(s)
        while i < n and s[i] != '.':
            ss = ss * 10 + (ord(s[i]) - 48)
            i += 1

        frac = 0.0
        if i < n and s[i] == '.':
            i += 1
            frac = _parse_frac_rest(s, i)

        sec = ss + frac
        return 15.0 * (hh + mm / 60.0 + sec / 3600.0)

    @numba.njit(inline='always')
    def parse_dec_dms_deg_scalar(s):
        # "+DD MM SS.sss" or "-DD MM SS.sss"
        sign = 1.0
        i = 0
        if s[0] == '-':
            sign = -1.0
            i = 1
        elif s[0] == '+':
            i = 1

        dd, i = _parse_unsigned_int_until(s, i, ' ')
        i += 1
        mm, i = _parse_unsigned_int_until(s, i, ' ')
        i += 1

        ss = 0
        n = len(s)
        while i < n and s[i] != '.':
            ss = ss * 10 + (ord(s[i]) - 48)
            i += 1

        frac = 0.0
        if i < n and s[i] == '.':
            i += 1
            frac = _parse_frac_rest(s, i)

        sec = ss + frac
        return sign * (dd + mm / 60.0 + sec / 3600.0)

    @numba.njit(parallel=True, cache=True)
    def parse_ra_array_numba(ra_strs):
        out = np.empty(len(ra_strs), dtype=np.float64)
        for i in numba.prange(len(ra_strs)):
            out[i] = parse_ra_hms_deg_scalar(ra_strs[i])
        return out

    @numba.njit(parallel=True, cache=True)
    def parse_dec_array_numba(dec_strs):
        out = np.empty(len(dec_strs), dtype=np.float64)
        for i in numba.prange(len(dec_strs)):
            out[i] = parse_dec_dms_deg_scalar(dec_strs[i])
        return out



if True:
    t0 = time.time()
    t1s = []

    for name, meta in metadata.items():
        space_rocks[name] = np.empty((len(space_rock_columns), meta['Count']), dtype=np.float64)
        metadata[name]['Index'] = 0  # To keep track of where to insert the next data point

    timers = ProfileTimer()

    for n_file, (t_mid, rows) in enumerate(all_rows):
        with timers["load fits"]:
            data, header = fits.getdata(files[n_file], header=True)
        with timers["init WCS"]:
            wcs = WCS(header)
        
        with timers["convert to pixels"]:
            #rows = skybot_json_to_pixels(rows, wcs, shape=data.shape, origin=0)
            js = rows

            rows2 = []
            ra_array = []
            de_array = []
            origin=0
            shape = data.shape

            #with timers["extract skycoords"]:
                # for row in js:
                #     #coord = skybot_row_to_skycoord(row)

                #     ra_str  = row["RA (hms)"]
                #     dec_str = row["DEC (dms)"]
                #     coord = SkyCoord(ra_str, dec_str, unit=(astropy.units.hourangle, astropy.units.deg), frame="icrs")

                #     ra_array.append(coord.ra.deg)
                #     de_array.append(coord.dec.deg)
            with timers["extract skycoords"]:
                ra_array0 = [row["RA (hms)"] for row in js]
                de_array0 = [row["DEC (dms)"] for row in js]
            with timers["process skycoords"]:
                #ra_array = Angle(ra_array0, unit=astropy.units.hourangle).deg
                #de_array = Angle(de_array0, unit=astropy.units.deg).deg
                ra_array = parse_ra_array_numba(ra_array0)
                de_array = parse_dec_array_numba(de_array0)
                #raise Exception

            ra_array = np.array(ra_array)
            de_array = np.array(de_array)
            with timers["world2pix"]:
                x_array, y_array = wcs.all_world2pix(ra_array, de_array, origin)

            with timers["assemble rows"]:
                for i, row in enumerate(js):
                    #coord = skybot_row_to_skycoord(row)
                    #x, y = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, origin)

                    out = dict(row)
                    out["ra_deg"] = ra_array[i]
                    out["dec_deg"] = de_array[i]
                    out["x_pix"] = float(x_array[i])
                    out["y_pix"] = float(y_array[i])

                    if shape is not None:
                        ny, nx = shape
                        out["on_chip"] = (0 <= x_array[i] < nx) and (0 <= y_array[i] < ny)
                    else:
                        out["on_chip"] = None

                    rows2.append(out)



        with timers["populate space rocks"]:
            for row in rows2:
                name = row['Name']
                idx = metadata[name]['Index']

                # handle position and velocity
                pxyz = row.get('position (au)', {'x': np.nan, 'y': np.nan, 'z': np.nan})
                vxyz = row.get('velocity (au/d)', {'x': np.nan, 'y': np.nan, 'z': np.nan})

                space_rocks[name][:, idx] = [
                    t_mid,
                    row.get('x_pix', np.nan),
                    row.get('y_pix', np.nan),
                    row.get('ra_deg', np.nan),
                    row.get('dec_deg', np.nan),
                    row.get('VMag (mag)', np.nan),
                    row.get('Err (arcsec)', np.nan),
                    row.get('d (arcsec)', np.nan),
                    row.get('dRA (arcsec/d)', np.nan),
                    row.get('dDEC (arcsec/d)', np.nan),
                    row.get('dg (ua)', np.nan),
                    row.get('dh (ua)', np.nan),
                    row.get('Phase (deg)', np.nan),
                    row.get('SunElong (deg)', np.nan),
                    pxyz['x'], pxyz['y'], pxyz['z'],
                    vxyz['x'], vxyz['y'], vxyz['z'],
                    float(row.get('on_chip', False)),
                ]

                metadata[name]['Index'] += 1

        t1 = time.time()
        t1s.append(t1 - t0)
        eta = "[N/A]"
        if 2 <= len(t1s):
            coeffs = np.polyfit(np.arange(len(t1s))[-10:], t1s[-10:], 1)
            eta_seconds = np.polyval(coeffs, len(files)) - t1s[-1]
            eta = humanize.precisedelta(eta_seconds, format="%0.3f")
            eta = f"[ETA: {eta}]"

        print(f"[{(n_file+1):4d}/{len(all_rows):4d}] time elapsed: {t1 - t0:.2f} s {eta}")
    print(timers)

fig, ax = plt.subplots()
ax.imshow(tonemap(data), origin='lower')
for idx, (name, dset) in enumerate(space_rocks.items()):
    if idx >= 10:
        break
    x = dset[1, :]
    y = dset[2, :]
    ax.plot(x, y, marker='o', linestyle='-', label=name)
ax.legend()
plt.show()

times = space_rocks['Thalia'][0] - 2457000 # convert to btjd
xyzt_TESS, xyzt_Earth, xyzt_Moon = query_jpl_horizons_with_given_BTJD(times)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
ax.plot(*xyzt_TESS.T[:3], label='TESS')
ax.plot(*xyzt_Earth.T[:3], label='Earth')
ax.plot(*xyzt_Moon.T[:3], label='Moon')

for idx, (name, dset) in enumerate(space_rocks.items()):
    if idx >= 10:
        break

    x = dset[space_rock_columns.index('x_position (au)'), :]
    y = dset[space_rock_columns.index('y_position (au)'), :]
    z = dset[space_rock_columns.index('z_position (au)'), :]
    ax.plot(x, y, z, label=name)

ax.set_aspect('equal')
ax.legend()
plt.show()

list(space_rocks.keys())[:10]
['Thalia', 'Rachele', 'Nike', 'Magdalena', 'Nerthus', 'McFadden', 'Eleutheria', 'Bredichina', 'Forsius', 'Adriana']

# Lets find a sequence that sorts these by median magnitude.
seq = sorted(space_rocks.keys(), key=lambda name: np.nanmedian(space_rocks[name][space_rock_columns.index('VMag (mag)'), :]))
print(seq[:10])

if True:
    shush = []
    fig, ax = plt.subplots()
    ax.imshow(nightmare, origin='lower')
    for idx, name in enumerate(seq):
        if idx >= 100:
            break
        dset = space_rocks[name]
        t = dset[0, :]
        x = dset[1, :]
        y = dset[2, :]
        # Lets add circular endcaps!
        shush += ax.plot(x, y, '-', linewidth=5, label=name, solid_capstyle='round')
        # Lets add a marker at the end of each path with the name, oriented in the direciton of motion. We may need to estimate that from the last few values of x and y by a best fit vs time.
        # Best fit the last 10 points to get the trajectory vector
        vx = np.polyfit(t[-10:], x[-10:], 1)[0]
        vy = np.polyfit(t[-10:], y[-10:], 1)[0]
        angle = np.arctan2(vy, vx)

        txt = ax.text(x[-1], y[-1], name, color=f'C{idx}', fontsize=8, rotation=np.degrees(angle), rotation_mode='anchor', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7, edgecolor='none'), transform=ax.transData)
        # To shift the text above the line, we can add a small offset in the y-direction in display coordinates. For example, to shift it 10 pixels up:
        txt.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(0, 8, matplotlib.transforms.IdentityTransform()))


        shush.append(txt)

    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1, 1),
        ncol=2,
        fontsize='small',
    )
    ax.set_xlim(0, 2048)
    ax.set_ylim(0, 2048)
    plt.show()


# check with just Thalia
th_x = space_rocks['Thalia'][space_rock_columns.index('x_position (au)'), :]
th_y = space_rocks['Thalia'][space_rock_columns.index('y_position (au)'), :]
th_z = space_rocks['Thalia'][space_rock_columns.index('z_position (au)'), :]
xcol = space_rock_columns.index('x_position (au)')
th_n_xyz = space_rocks['Thalia'][xcol:xcol+3, :].T

# subtract thalia from tess
dX_n_xyz = th_n_xyz - xyzt_TESS[:, :3]
#dX_n_xyz = dX_n_xyz / np.linalg.norm(dX_n_xyz, axis=1, keepdims=True)
# "icrs"
TESS_Mapping_Variables = wcs_variables_load_nonan(header, wcs_variables_validate(header))
TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)
pxy_n_xy = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, dX_n_xyz)

fig, ax = plt.subplots()
ax.imshow(nightmare, origin='lower')
ax.plot(pxy_n_xy[:, 0], pxy_n_xy[:, 1], marker='o', linestyle='-', label='Thalia (from JPL Horizons via TESS)')
ax.legend()
plt.show()


thalia_tess_observer_rq[0]
{'Num': 23, 'Name': 'Thalia', 'RA (deg)': 38.09478422367955, 'DEC (deg)': 5.159340246380774, 'Class': 'MB>Middle', 'VMag (mag)': 10.4, 'Err (arcsec)': 0.033, 'dRA (arcsec/h)': -36.9006, 'dDEC (arcsec/h)': -7.2757, 'dg (ua)': 1.55737198828, 'dh (ua)': 2.53360089041, 'Phase (deg)': 5.82, 'SunElong (deg)': 14.93, 'position (au)': {'x': 2.125953158, 'y': 1.34776599, 'z': 0.308176946}, 'velocity (au/d)': {'x': -0.007706069, 'y': 0.006275506, 'z': 0.004680865}, 'ref_epoch': 2458410}
thalia_tess_observer_rq[1]
{'Num': 23, 'Name': 'Thalia', 'RA (deg)': 38.0896329641165, 'DEC (deg)': 5.158332947042845, 'Class': 'MB>Middle', 'VMag (mag)': 10.4, 'Err (arcsec)': 0.033, 'dRA (arcsec/h)': -36.9792, 'dDEC (arcsec/h)': -7.2311, 'dg (ua)': 1.55724818458, 'dh (ua)': 2.53354761571, 'Phase (deg)': 5.81, 'SunElong (deg)': 14.91, 'position (au)': {'x': 2.125953158, 'y': 1.34776599, 'z': 0.308176946}, 'velocity (au/d)': {'x': -0.007706069, 'y': 0.006275506, 'z': 0.004680865}, 'ref_epoch': 2458410}

space_rock_columns
['times', 'x_pix', 'y_pix', 'ra_deg', 'dec_deg', 'VMag (mag)', 'Err (arcsec)', 'd (arcsec)', 'dRA (arcsec/d)', 'dDEC (arcsec/d)', 'dg (ua)', 'dh (ua)', 'Phase (deg)', 'SunElong (deg)', 'x_position (au)', 'y_position (au)', 'z_position (au)', 'vx_velocity (au/d)', 'vy_velocity (au/d)', 'vz_velocity (au/d)', 'on_chip']

thalia_better = space_rocks['Thalia'].copy()
# fill this with the right data
for n_time in range(len(thalia_tess_observer_rq)):
    row = thalia_tess_observer_rq[n_time]
    thalia_better[space_rock_columns.index('ra_deg'), n_time] = row['RA (deg)']
    thalia_better[space_rock_columns.index('dec_deg'), n_time] = row['DEC (deg)']
    thalia_better[space_rock_columns.index('VMag (mag)'), n_time] = row['VMag (mag)']
    thalia_better[space_rock_columns.index('Err (arcsec)'), n_time] = row['Err (arcsec)']
    thalia_better[space_rock_columns.index('dRA (arcsec/d)'), n_time] = row['dRA (arcsec/h)'] * 24 # convert to arcsec/d
    thalia_better[space_rock_columns.index('dDEC (arcsec/d)'), n_time] = row['dDEC (arcsec/h)'] * 24 # convert to arcsec/d
    thalia_better[space_rock_columns.index('dg (ua)'), n_time] = row['dg (ua)']
    thalia_better[space_rock_columns.index('dh (ua)'), n_time] = row['dh (ua)']
    thalia_better[space_rock_columns.index('Phase (deg)'), n_time] = row['Phase (deg)']
    thalia_better[space_rock_columns.index('SunElong (deg)'), n_time] = row['SunElong (deg)']
    pos = row['position (au)']
    vel = row['velocity (au/d)']
    thalia_better[space_rock_columns.index('x_position (au)'), n_time] = pos['x']
    thalia_better[space_rock_columns.index('y_position (au)'), n_time] = pos['y']
    thalia_better[space_rock_columns.index('z_position (au)'), n_time] = pos['z']
    thalia_better[space_rock_columns.index('vx_velocity (au/d)'), n_time] = vel['x']
    thalia_better[space_rock_columns.index('vy_velocity (au/d)'), n_time] = vel['y']
    thalia_better[space_rock_columns.index('vz_velocity (au/d)'), n_time] = vel['z']

# Use the above solvers to get x_pix and y_pix
thalia_better[space_rock_columns.index('x_pix'), :] = wcs.world_to_pixel_values(thalia_better[space_rock_columns.index('ra_deg'), :], thalia_better[space_rock_columns.index('dec_deg'), :])[0]
thalia_better[space_rock_columns.index('y_pix'), :] = wcs.world_to_pixel_values(thalia_better[space_rock_columns.index('ra_deg'), :], thalia_better[space_rock_columns.index('dec_deg'), :])[1]

    
ra_array = thalia_better[space_rock_columns.index('ra_deg'), :]
de_array = thalia_better[space_rock_columns.index('dec_deg'), :]
x_array, y_array = wcs.all_world2pix(ra_array, de_array, 0)
thalia_better[space_rock_columns.index('x_pix'), :] = x_array
thalia_better[space_rock_columns.index('y_pix'), :] = y_array



if True:
    # Lets load and plot the first pic.
    file_idx = 0
    file_idx = len(times)-1
    data, header = fits.getdata(files[file_idx], header=True)

    name = "Thalia"
    dset = thalia_better
    #t = dset[0, :]
    t = times
    x = dset[1, :]
    y = dset[2, :]

    fig, ax = plt.subplots()
    img = slog(data) # Lets set vmax to be the value of where thalia should be at this pixel
    ax.imshow(img, origin='lower', vmin=np.nanmin(img), vmax=img[y[file_idx].astype(int), x[file_idx].astype(int)])
    #ax.imshow(tonemap(data), origin='lower')
    #ax.plot(x_array, y_array, marker='o', linestyle='-', label='Thalia')

    ax.plot(x, y, '-', linewidth=5, label=name, solid_capstyle='round')

    vx = np.polyfit(t[-10:], x[-10:], 1)[0]
    vy = np.polyfit(t[-10:], y[-10:], 1)[0]
    angle = np.arctan2(vy, vx)

    txt = ax.text(x[-1], y[-1], name, color=f'C{idx}', fontsize=8, rotation=np.degrees(angle), rotation_mode='anchor', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7, edgecolor='none'), transform=ax.transData)
    txt.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(0, 8, matplotlib.transforms.IdentityTransform()))

    ax.legend()
    plt.show()

# Lets render a video of Thalia moving about!
# 13.8582677776226 471.8114400533754 xlim
# 444.08058930317003 722.9332635106829 ylim
# Lets take each data and crop it to this region.

# What was Thalia's magnitude value?
mag = thalia_better[space_rock_columns.index('VMag (mag)'), :]
# Lets convert this to a flux.
Lum = 10**(-0.4*mag)

def ilerp(a,b,x):
    return (x-a)/(b-a)

# Now we can set the vmax of the slog to be proportional to this luminosity, so that Thalia is always visible but not too blown out.
img_seq = []
vmax = None
for file_idx in range(len(times)):
    data, header = fits.getdata(files[file_idx], header=True)
    data = data[444:723, 14:472]

    name = "Thalia"
    dset = thalia_better
    t = dset[0, file_idx]
    x = dset[1, file_idx] - 14
    y = dset[2, file_idx] - 444

    if vmax is None:
        vmax = (data)[y.astype(int), x.astype(int)]

    Lum_Ratio = Lum[file_idx] / Lum[0] # Relative luminosity compared to the first frame
    
    #img = slog(data) # Adjust the scale factor as needed to get good visibility of Thalia.
    #img = (img / img0) * Lum_Ratio # Scale the image by the luminosity ratio to keep Thalia visible.
    img = ot.sRGB_from_RGB((data/vmax)) # Use OpenTon's color conversion to get a more visually appealing result.
    img_seq.append(cmap(img))




if True:
    # Lets load and plot the first pic.
    file_idx = 0
    file_idx = len(times)-1
    data, header = fits.getdata(files[file_idx], header=True)

    data = data[444:723, 14:472]

    name = "Thalia"
    dset = thalia_better
    #t = dset[0, :]
    t = times
    x = dset[1, :] - 14
    y = dset[2, :] - 444

    fig, ax = plt.subplots()
    img = slog(data) # Lets set vmax to be the value of where thalia should be at this pixel
    ax.imshow(img, origin='lower', vmin=np.nanmin(img), vmax=img[y[file_idx].astype(int), x[file_idx].astype(int)])
    #ax.imshow(tonemap(data), origin='lower')
    #ax.plot(x_array, y_array, marker='o', linestyle='-', label='Thalia')

    ax.plot(x, y, '-', linewidth=5, label=name, solid_capstyle='round')

    vx = np.polyfit(t[-10:], x[-10:], 1)[0]
    vy = np.polyfit(t[-10:], y[-10:], 1)[0]
    angle = np.arctan2(vy, vx)

    txt = ax.text(x[-1], y[-1], name, color=f'C{idx}', fontsize=8, rotation=np.degrees(angle), rotation_mode='anchor', ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.7, edgecolor='none'), transform=ax.transData)
    txt.set_transform(ax.transData + matplotlib.transforms.ScaledTranslation(0, 8, matplotlib.transforms.IdentityTransform()))

    ax.legend()
    plt.show()



if True:
    vmax = None
    img_seq_grey = []
    for file_idx in range(len(times)):
        data, header = fits.getdata(files[file_idx], header=True)
        data = data[444:723, 14:472]

        name = "Thalia"
        dset = thalia_better
        t = dset[0, file_idx]
        x = dset[1, file_idx] - 14
        y = dset[2, file_idx] - 444

        if vmax is None:
            vmax = (data)[y.astype(int), x.astype(int)]

        #Lum_Ratio = Lum[file_idx] / Lum[0] # Relative luminosity compared to the first frame

        #img = slog(data) # Adjust the scale factor as needed to get good visibility of Thalia.
        #img = (img / img0) * Lum_Ratio # Scale the image by the luminosity ratio to keep Thalia visible.
        #img = ot.sRGB_from_RGB((data/vmax)) # Use OpenTon's color conversion to get a more visually appealing result.
        img = (data/vmax) * Lum_Ratio[file_idx]
        img_seq_grey.append(img)
    img_seq_grey = np.array(img_seq_grey)

    median = np.median(img_seq_grey, axis=0)
    img_seq_diff = img_seq_grey - median[None,:,:]
    for n in range(len(img_seq_diff)):
        img_seq_diff[n] -= np.median(img_seq_diff[n])
        img_seq_diff[n] -= np.median(img_seq_diff[n], axis=0)[None,:]
        img_seq_diff[n] -= np.median(img_seq_diff[n], axis=1)[:,None]
    animate_cv2(tonemap(img_seq_diff))



def kill_ram_processing_good_result():
    vmax = None
    img_seq_grey = []
    for file_idx in range(len(times)):
        data, header = fits.getdata(files[file_idx], header=True)
        data = data[444:723, 14:472]

        name = "Thalia"
        dset = thalia_better
        t = dset[0, file_idx]
        x = dset[1, file_idx] - 14
        y = dset[2, file_idx] - 444

        if vmax is None:
            vmax = (data)[y.astype(int), x.astype(int)]

        #Lum_Ratio = Lum[file_idx] / Lum[0] # Relative luminosity compared to the first frame

        #img = slog(data) # Adjust the scale factor as needed to get good visibility of Thalia.
        #img = (img / img0) * Lum_Ratio # Scale the image by the luminosity ratio to keep Thalia visible.
        #img = ot.sRGB_from_RGB((data/vmax)) # Use OpenTon's color conversion to get a more visually appealing result.
        img = (data/vmax) * Lum_Ratio[file_idx]
        img_seq_grey.append(img)
    img_seq_grey = np.array(img_seq_grey)

    img_seq_diff = img_seq_grey.copy()# - median[None,:,:]
    #for n in range(len(img_seq_diff)):
        #img_seq_diff[n] -= np.median(img_seq_diff[n])
    #    img_seq_diff[n] -= np.median(img_seq_diff[n], axis=0)[None,:]
    #    img_seq_diff[n] -= np.median(img_seq_diff[n], axis=1)[:,None]
    #my = np.median(img_seq_diff, axis=1)#[:,None,:]
    #mx = np.median(img_seq_diff, axis=2)#[:,:,None]
    #mt = np.median(img_seq_diff, axis=0)#[None,:,:]
    img_seq_diff -= np.median(img_seq_diff, axis=(0,1))[None,None,:]
    img_seq_diff -= np.median(img_seq_diff, axis=(1,2))[:,None,None]
    img_seq_diff -= np.median(img_seq_diff, axis=(0,2))[None,:,None]
    #med = np.median([my[:,None,:], mx[:,:,None], mt[None,:,:]], axis=0)
    #med = np.empty((3, *img_seq_diff.shape), dtype=img_seq_diff.dtype)
    #med[0] = my[:,None,:]
    #med[1] = mx[:,:,None]
    #med[2] = mt[None,:,:]
    #med = np.median(med, axis=0)
    animate_cv2(tonemap(img_seq_diff))# + median[None,:,:]))

# Due to significant RAM limitations, lets perform these median shifts above as in-place operations for each item in the independent axis.
def offset_2d_medians_inplace(sector_data):
    #img_seq_diff -= np.median(img_seq_diff, axis=(0,1))[None,None,:]
    #img_seq_diff -= np.median(img_seq_diff, axis=(1,2))[:,None,None]
    #img_seq_diff -= np.median(img_seq_diff, axis=(0,2))[None,:,None]

    # x axis offsets
    t0 = time_ns()
    for xx in range(sector_data.shape[2]):
        sector_data[:,:,xx] -= np.median(sector_data[:,:,xx])[None,None]
        t1 = time_ns()
         # I should have the screen real estate for the full temporal slices too!
        cv2_imshow(sector_data[:,:,xx], title=f"x={xx}/{sector_data.shape[2]}")
        if (t1-t0) > 60e9:
            print(f"Processed x={xx} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("x medians done")

    # I believe the order mattered here? But I'm not sure really. I suspect it shouldn't..? 
    # t axis offsets
    t0 = time_ns()
    for tt in range(sector_data.shape[0]):
        sector_data[tt,:,:] -= np.median(sector_data[tt,:,:])[None,None]
        t1 = time_ns()
        cv2_imshow(sector_data[tt,:,:], title=f"t={tt}/{sector_data.shape[0]}")
        if (t1-t0) > 60e9:
            print(f"Processed t={tt} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("t medians done")

    # y axis offsets
    t0 = time_ns()
    for yy in range(sector_data.shape[1]):
        sector_data[:,yy,:] -= np.median(sector_data[:,yy,:])[None,None]
        t1 = time_ns()
        cv2_imshow(sector_data[:,yy,:], title=f"y={yy}/{sector_data.shape[1]}")
        if (t1-t0) > 60e9:
            print(f"Processed y={yy} in {(t1-t0)*1e-9:.2f} seconds")
            t0 = t1
    print("y medians done")
    




def compress_ffmpeg(name, name2, audiosrc=None):
    command = 'ffmpeg -i "%s" -c:v libx264 -b:v 5M -preset veryslow -crf 22 -y "%s"'%(name, name2)
    if type(audiosrc) != type(None):
        command = 'ffmpeg -i "%s" -i "%s" -c:v libx264 -b:v 5M -preset veryslow -crf 22 -c:a aac -strict experimental -y "%s"'%(name, audiosrc, name2)
    print(">", command)
    #command = command.split(' ')
    #import subprocess as sp
    #sp.call(command)
    Call(command)

# Lets write a custom ffmpeg command to compress this *image sequence,* not from python.
# The images are stored in C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\tmp as Thalia_0228.png, Thalia_0229.png, ...
# We want to create a video called Thalia.mp4.
# ffmpeg -i "C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\tmp\Thalia_%04d.png" -c:v libx264 -b:v 5M -preset veryslow -crf 22 -y "C:\Users\Joe\Desktop\Projects\2026_Spring\DIA\tmp\Thalia.mp4"



data, header = fits.getdata(files[0], header=True)
data = data.astype(np.float64)
ra_deg = 80.485795
dec_deg = 43.989433
x_pix, y_pix = wcs.all_world2pix(ra_deg, dec_deg, 0)
x_pix, y_pix
(array(1426.36096803), array(1616.38496615))
#plt.imshow(data, origin='lower')

# Lets plot a 64x64 cutout around this point.
px = int(x_pix)
py = int(y_pix)
cutout = data[py-32:py+32, px-32:px+32]
plt.imshow(cutout, origin='lower')
px = int(x_pix);py = int(y_pix);cutout = data[py-32:py+32, px-32:px+32];fig,ax=plt.subplots();ax.imshow(tonemap(cutout), origin='lower', interpolation='lanczos');ax.set_title("64x64 Cutout surrounding TIC 369048064");plt.show()
# yippee!! that's the guy

