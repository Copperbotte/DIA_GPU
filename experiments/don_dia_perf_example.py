
# Joseph Kessler
# 2026 February 8
# don_dia_perf_example.py
################################################################################

# Likely global imports
import numpy as np
from time import time_ns
import humanize

#NUM_ITERATIONS = 2
NUM_ITERATIONS = 10
Statistics_Results = dict()
#TimerStats_Args = dict(runs=NUM_ITERATIONS, suppress=False)
#TimerStats_Args = dict(run_duration=NUM_ITERATIONS*1.0, suppress=False)
TimerStats_Args = dict(run_duration=30.0, suppress=True)

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

################################################################################
# This chunk was taken from D:\My Drive\Python Scripts\2025_Fall\gaia\gaia_cuda\timer_helper.py.

# Timer function usable with a with directive.
# with Timer():
#     ...
class Timer:
    def fmttime(t0_ns, t1_ns):
        return humanize.metric((t1_ns-t0_ns)*1e-9, 's')
    def __init__(self, prefix="", stats=None, suppress=False, declare=False):
        self.prefix = prefix
        self.t0, self.t1 = None, None
        self.stats = stats
        self.suppress = suppress
        self.declare = declare
    def __str__(self):
        if self.t0 is None or self.t1 is None:
            return "Nothing to show!"
        s = Timer.fmttime(self.t0, self.t1)
        if self.prefix != "":
            s = ' '.join([self.prefix, "in", s])
        return s
    def __enter__(self):
        if (not self.suppress) and self.declare:
            print("Running %s..."%self.prefix)
        self.t0 = time_ns()
        return self
    def __exit__(self, *args):
        self.t1 = time_ns()
        if self.stats is not None:
            self.stats.append((self.t1 - self.t0)*1e-9)
        if not self.suppress:
            print(str(self))

# Timer statistics function usable with a for directive.
# for timer in TimerStats():
#     with timer:
#         ...
class TimerStats:
    def __init__(self, prefix="", runs=10, suppress=False, multiplier=1, run_duration=None):
        """Iterator that yields `Timer` objects.

        Args:
            prefix: text prefix for printed messages
            runs: maximum number of timing runs (useful as an upper bound)
            suppress: mute per-run printing
            multiplier: scale stored times by this factor
            run_duration: optional float (seconds). If provided, the
                iterator will keep yielding runs until this many seconds
                have elapsed (or until `runs` is reached, if that is a
                positive integer).
        """
        self.prefix = prefix
        self.runs = runs
        self.suppress = suppress
        self.multiplier = multiplier
        self.run_duration = None if run_duration is None else float(run_duration)
        self.mode = "duration" if self.run_duration is not None else "runs"
        self.stats = []
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __iter__(self):
        print("collecting %d statistics for %s:"%(self.runs, self.prefix))
        self.t0 = time_ns()
        return self
    def __next__(self):
        spawn_new_timer = False
        if self.mode == "runs":
            spawn_new_timer = len(self.stats) < self.runs
        elif self.mode == "duration":
            elapsed = (time_ns() - self.t0)*1e-9
            spawn_new_timer = elapsed < self.run_duration

        if spawn_new_timer:
            if self.suppress:
                prefix = ""
            else:
                prefix = " ".join([self.prefix, "run", str(len(self.stats)+1), "done"])
            return Timer(prefix=prefix, stats=self.stats, suppress=self.suppress)
        else:
            with Timer("%s statistics collected in"%self.prefix) as timer:
                timer.t0 = self.t0
            self.stats = stats = np.array(self.stats) * self.multiplier
            #print("results: %.4e ± %.4e"%(np.mean(stats, axis=0), np.std(stats, axis=0)))
            mean_, std_ = np.mean(stats, axis=0), np.std(stats, axis=0)
            str_ = ""
            if 60 < mean_:
                over, mean_ = divmod(mean_, 60)
                str_ += humanize.precisedelta(60*over + 1, minimum_unit='seconds')
                str_ = str_[:str_.rfind(" and ")+5] # Slice at and
            str_ += humanize.metric(mean_, 's', 4) + " ± " + humanize.metric(std_, 's', 4)
            print(" "*4 + "results: %s"%(str_))
            if self.multiplier != 1:
                print(" "*4 + "Extrapolated with multiplier", self.multiplier)
            raise StopIteration

################################################################################
# Generate a random 2048x2048 array of floats
w0, h0 = 2048, 2048
array0 = np.random.random((w0, h0)).astype(np.float32)

# Generate some arbitrary kernel data for a 32x32 kernel. We will use a Gaussian-like kernel for this example.
kernel = np.random.random((32, 32)).astype(np.float32)
kx = np.linspace(-1, 1, kernel.shape[1])
ky = np.linspace(-1, 1, kernel.shape[0])
kr2 = kx[:, None]**2 + ky[None, :]**2
kernel *= np.exp(-kr2) 

array_dst = array0[:-kernel.shape[0], :-kernel.shape[1]]
################################################################################
# Test 1: Pure python

def test1_pure_python():
    # Use a smaller domain and extend with a multiplier
    w00, h00 = 128, 128
    array00 = np.random.random((w00, h00)).astype(np.float32)
    mult00 = (w0*h0) / (w00*h00)

    array1_adj = array00.copy()[:-kernel.shape[0], :-kernel.shape[1]]
    with TimerStats(prefix="Pure python convolution", multiplier=mult00, **TimerStats_Args) as timerStats:
        for timer in timerStats:
            with timer:
                h, w = array1_adj.shape
                kh, kw = kernel.shape
                for u in range(h-kh):
                    for v in range(w-kw):
                        accum = np.float32(0.0)
                        for i in range(kh):
                            for j in range(kw):
                                accum += array0[u+i, v+j] * kernel[i, j]
                        array1_adj[u, v] = accum
        Statistics_Results[1] = timerStats.stats
    print()
    globals().update(locals())
################################################################################
# Test 2: Numba

import numba

@numba.njit() # Duplicate to keep the warmup behavior in the loop
def convolve_numba_control(array, kernel, array0):
    h, w = array.shape
    kh, kw = kernel.shape
    for u in range(h-kh):
        for v in range(w-kw):
            accum = np.float32(0.0)
            for i in range(kh):
                for j in range(kw):
                    accum += array0[u+i, v+j] * kernel[i, j]
            array[u, v] = accum

array1 = array_dst.copy() # Copy for error checking. We'll assume numba is fine haha
with Timer("Numba convolution control done"):
    convolve_numba_control(array1, kernel, array0)


#@numba.njit(parallel=True)
@numba.njit()
def convolve_numba(array, kernel, array0):
    h, w = array.shape
    kh, kw = kernel.shape
    for u in range(h-kh):
        for v in range(w-kw):
            accum = np.float32(0.0)
            for i in range(kh):
                for j in range(kw):
                    accum += array0[u+i, v+j] * kernel[i, j]
            array[u, v] = accum

def test2_numba():
    array2 = array_dst.copy()
    with TimerStats(prefix="Numba convolution", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("Numba convolution warmup"):
            convolve_numba(array2, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_numba(array2, kernel, array0)
                err = np.max(np.abs(array2 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[2] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 3: Numba Parallel

@numba.njit(parallel=True, fastmath=False)
def convolve_numba_parallel(array, kernel, array0):
    h, w = array.shape
    kh, kw = kernel.shape
    u_max = h - kh
    v_max = w - kw

    for u in numba.prange(u_max):
        for v in range(v_max):
            accum = np.float32(0.0)
            for i in range(kh):
                for j in range(kw):
                    accum += array0[u+i, v+j] * kernel[i, j]
            array[u, v] = accum

def test3_numba_parallel():
    array3 = array_dst.copy()
    with TimerStats(prefix="Numba convolution parallel", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("Numba convolution parallel warmup"):
            convolve_numba_parallel(array3, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_numba_parallel(array3, kernel, array0)
                err = np.max(np.abs(array3 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[3] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 4: Cppyy directly
# This chunk was taken from projects\winter_2023\pi-gui-backup\tests\FlexCAN3Controller_Tests.py.
import os
import warnings
#warnings.simplefilter("ignore")
os.environ['EXTRA_CLING_ARGS'] = "-w -W -Wc++11-narrowing -O3 -march=native"
#os.environ['CPPYY_CRASH_QUIET'] = "1"

import cppyy
from array import array
import ctypes

cpp = cppyy.gbl

convolve_src = """
void convolve_cpp(float* array, float* kernel, float* array0, int h, int w, int kh, int kw, int h0, int w0)
{
    for (int u = 0; u < h - kh; u++) {
        for (int v = 0; v < w - kw; v++) {
            float accum = 0.0f;
            for (int i = 0; i < kh; i++) {
                for (int j = 0; j < kw; j++) {
                    accum += array0[(u + i) * w0 + (v + j)] * kernel[i * kw + j];
                }
            }
            array[u * w + v] = accum;
        }
    }
}
"""
cppyy.cppdef(convolve_src)

def convolve_cppyy(array, kernel, array0, cpp_mod=cpp):
    """Call the C++ `convolve_cpp` with proper contiguous float32 buffers.

    Parameters
    - array: destination numpy array (modified in-place)
    - kernel: 2D kernel numpy array
    - array0: source numpy array
    - cpp_mod: cppyy module (defaults to `cpp`)

    Returns the contiguous buffer used for the destination (for convenience).
    """
    arr_c = np.ascontiguousarray(array, dtype=np.float32)
    kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)
    cpp_mod.convolve_cpp(
        arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(arr_c.shape[0]), int(arr_c.shape[1]), int(kernel.shape[0]), int(kernel.shape[1]), int(array0.shape[0]), int(array0.shape[1])
    )
    if arr_c is not array:
        array[:] = arr_c
    return arr_c

def test4_cppyy():
    array4 = array_dst.copy()
    with TimerStats(prefix="Cppyy convolution", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("Cppyy convolution warmup"):
            convolve_cppyy(array4, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_cppyy(array4, kernel, array0)
                err = np.max(np.abs(array4 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[4] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 5: Cppyy with Multithreading
# Copilot suggested "Cppyy with OpenMP parallelization." I'll have to look into this later!

convolve_parallel_src = """

#include <thread>
#include <vector>
#include <algorithm>

void convolve_cpp_row(float* array, const float* kernel, const float* array0,
                      int h, int w, int kh, int kw, int h0, int w0, int row)
{
    const int u_max = h - kh;
    const int v_max = w - kw;
    if (row < 0 || row >= u_max || v_max <= 0) return;

    for (int v = 0; v < v_max; v++) {
        float accum = 0.0f;
        for (int i = 0; i < kh; i++) {
            const int row0 = (row + i) * w0;
            const int rowk = i * kw;
            for (int j = 0; j < kw; j++) {
                accum += array0[row0 + (v + j)] * kernel[rowk + j];
            }
        }
        array[row * w + v] = accum;
    }
}

static void convolve_cpp_rows(float* array, const float* kernel, const float* array0,
                              int h, int w, int kh, int kw, int h0, int w0,
                              int row0, int row1)
{
    const int u_max = h - kh;
    if (row0 < 0) row0 = 0;
    if (row1 > u_max) row1 = u_max;
    for (int row = row0; row < row1; ++row) {
        convolve_cpp_row(array, kernel, array0, h, w, kh, kw, h0, w0, row);
    }
}

void convolve_cpp_parallel(float* array, float* kernel, float* array0,
                           int h, int w, int kh, int kw, int h0, int w0,
                           int num_threads)
{
    const int u_max = h - kh;
    if (u_max <= 0) return;

    if (num_threads <= 0) num_threads = 1;
    num_threads = std::min(num_threads, u_max);

    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);

    int r0 = 0;
    for (int t = 0; t < num_threads; ++t) {
        int r1 = (int)((long long)(t + 1) * u_max / num_threads);
        threads.emplace_back(convolve_cpp_rows,
                             array, kernel, array0,
                             h, w, kh, kw, h0, w0,
                             r0, r1);
        r0 = r1;
    }

    for (auto& th : threads) th.join();
}
"""
cppyy.cppdef(convolve_parallel_src)

def convolve_cppyy_parallel(array, kernel, array0, num_threads=4, cpp_mod=cpp):
    """Call the C++ `convolve_cpp_parallel` with proper contiguous float32 buffers.

    Parameters
    - array: destination numpy array (modified in-place)
    - kernel: 2D kernel numpy array
    - array0: source numpy array
    - num_threads: number of threads to use in the C++ function
    - cpp_mod: cppyy module (defaults to `cpp`)

    Returns the contiguous buffer used for the destination (for convenience).
    """
    arr_c = np.ascontiguousarray(array, dtype=np.float32)
    kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)
    cpp_mod.convolve_cpp_parallel(
        arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(arr_c.shape[0]), int(arr_c.shape[1]), int(kernel.shape[0]), int(kernel.shape[1]), int(array0.shape[0]), int(array0.shape[1]),
        int(num_threads)
    )
    if arr_c is not array:
        array[:] = arr_c
    return arr_c

def test5_cppyy_parallel():
    array5 = array_dst.copy()
    with TimerStats(prefix="Cppyy convolution parallel", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("Cppyy convolution parallel warmup"):
            convolve_cppyy_parallel(array5, kernel, array0, num_threads=4)
        for timer in timerStats:
            with timer:
                convolve_cppyy_parallel(array5, kernel, array0, num_threads=4)
                err = np.max(np.abs(array5 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[5] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 6 & 7: External GCC Compilation using Call & cppyy

# Assume we prebuilt an object for cppyy to include
# Lets call it convolve_cpp_ext.o and it has the same functions as the cppyy-defined ones. We can compile it with:
# g++ -O3 -march=native -fopt-info-vec-optimized -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -pedantic -Wc++11-narrowing -c convolve_cpp_ext.cpp -o convolve_cpp_ext.o
# Then we can load it with convolve_cpp_ext.h

convolve_cpp_ext_header = r"""
#pragma once
#ifdef __cplusplus
extern "C" {
#endif

__declspec(dllexport)
void convolve_cpp(float* array, float* kernel, float* array0,
                  int h, int w, int kh, int kw, int h0, int w0);

__declspec(dllexport)
void convolve_cpp_parallel(float* array, float* kernel, float* array0,
                           int h, int w, int kh, int kw, int h0, int w0,
                           int num_threads);

#ifdef __cplusplus
}
#endif
"""
with open("convolve_cpp_ext.h", "w", encoding="utf-8") as f:
    f.write(convolve_cpp_ext_header)

convolve_cpp_ext_src = r"""
#include <thread>
#include <vector>
#include <algorithm>

extern "C" 
void convolve_cpp(float* array, float* kernel, float* array0, int h, int w, int kh, int kw, int h0, int w0)
{
    for (int u = 0; u < h - kh; u++) {
        for (int v = 0; v < w - kw; v++) {
            float accum = 0.0f;
            for (int i = 0; i < kh; i++) {
                for (int j = 0; j < kw; j++) {
                    accum += array0[(u + i) * w0 + (v + j)] * kernel[i * kw + j];
                }
            }
            array[u * w + v] = accum;
        }
    }
}

void convolve_cpp_row(float* array, const float* kernel, const float* array0,
                      int h, int w, int kh, int kw, int h0, int w0, int row)
{
    const int u_max = h - kh;
    const int v_max = w - kw;
    if (row < 0 || row >= u_max || v_max <= 0) return;

    for (int v = 0; v < v_max; v++) {
        float accum = 0.0f;
        for (int i = 0; i < kh; i++) {
            const int row0 = (row + i) * w0;
            const int rowk = i * kw;
            for (int j = 0; j < kw; j++) {
                accum += array0[row0 + (v + j)] * kernel[rowk + j];
            }
        }
        array[row * w + v] = accum;
    }
}

static void convolve_cpp_rows(float* array, const float* kernel, const float* array0,
                              int h, int w, int kh, int kw, int h0, int w0,
                              int row0, int row1)
{
    const int u_max = h - kh;
    if (row0 < 0) row0 = 0;
    if (row1 > u_max) row1 = u_max;
    for (int row = row0; row < row1; ++row) {
        convolve_cpp_row(array, kernel, array0, h, w, kh, kw, h0, w0, row);
    }
}

extern "C"
void convolve_cpp_parallel(float* array, float* kernel, float* array0,
                           int h, int w, int kh, int kw, int h0, int w0,
                           int num_threads)
{
    const int u_max = h - kh;
    if (u_max <= 0) return;

    if (num_threads <= 0) num_threads = 1;
    num_threads = std::min(num_threads, u_max);

    std::vector<std::thread> threads;
    threads.reserve((size_t)num_threads);

    int r0 = 0;
    for (int t = 0; t < num_threads; ++t) {
        int r1 = (int)((long long)(t + 1) * u_max / num_threads);
        threads.emplace_back(convolve_cpp_rows,
                             array, kernel, array0,
                             h, w, kh, kw, h0, w0,
                             r0, r1);
        r0 = r1;
    }

    for (auto& th : threads) th.join();
}
"""

with open("convolve_cpp_ext.cpp", "w", encoding="utf-8") as f:
    f.write(convolve_cpp_ext_src)

# Compile!
cmd = (
    "g++ -shared -O3 -march=native "
    "-fopt-info-vec-optimized "
    "-Wall -Wextra -Wno-unused-variable -Wno-unused-parameter -pedantic "
    "-o convolve_cpp_ext.dll convolve_cpp_ext.cpp"
)
Call(cmd)

# Bind to cppyy
#cppyy.include("convolve_cpp_ext.h")
#cppyy.load_library("convolve_cpp_ext.dll")
#dll = ctypes.CDLL(os.path.abspath("convolve_cpp_ext.dll"))

os.add_dll_directory(r"C:\tools\msys64\mingw64\bin")
dll = ctypes.CDLL(os.path.abspath("convolve_cpp_ext.dll"))

dll.convolve_cpp.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # array
    ctypes.POINTER(ctypes.c_float),  # kernel
    ctypes.POINTER(ctypes.c_float),  # array0
    ctypes.c_int, ctypes.c_int,      # h, w
    ctypes.c_int, ctypes.c_int,      # kh, kw
    ctypes.c_int, ctypes.c_int       # h0, w0
]
dll.convolve_cpp.restype = None

def convolve_gcc(array, kernel, array0):
    arr_c  = np.ascontiguousarray(array,  dtype=np.float32)
    kr_c   = np.ascontiguousarray(kernel,   dtype=np.float32).ravel()
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

    dll.convolve_cpp(
        arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr_c.shape[0], arr_c.shape[1],
        kernel.shape[0], kernel.shape[1],
        arr0_c.shape[0], arr0_c.shape[1]
    )
    if arr_c is not array:
        array[:] = arr_c
    return arr_c

def test6_gcc():
    array6 = array_dst.copy()
    with TimerStats(prefix="g++ convolution", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("g++ convolution warmup"):
            convolve_gcc(array6, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_gcc(array6, kernel, array0)
                err = np.max(np.abs(array6 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[6] = timerStats.stats
    print()
    globals().update(locals())


dll.convolve_cpp_parallel.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # array
    ctypes.POINTER(ctypes.c_float),  # kernel
    ctypes.POINTER(ctypes.c_float),  # array0
    ctypes.c_int, ctypes.c_int,      # h, w
    ctypes.c_int, ctypes.c_int,      # kh, kw
    ctypes.c_int, ctypes.c_int,      # h0, w0
    ctypes.c_int                    # num_threads
]
dll.convolve_cpp_parallel.restype = None

def convolve_gcc_parallel(array, kernel, array0, num_threads=8):
    arr_c  = np.ascontiguousarray(array,  dtype=np.float32)
    kr_c   = np.ascontiguousarray(kernel,   dtype=np.float32).ravel()
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

    dll.convolve_cpp_parallel(
        arr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kr_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr0_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        arr_c.shape[0], arr_c.shape[1],
        kernel.shape[0], kernel.shape[1],
        arr0_c.shape[0], arr0_c.shape[1],
        num_threads
    )
    if arr_c is not array:
        array[:] = arr_c
    return arr_c

def test7_gcc_parallel():
    array7 = array_dst.copy()
    with TimerStats(prefix="g++ convolution parallel", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("g++ convolution parallel warmup"):
            convolve_gcc_parallel(array7, kernel, array0, num_threads=8)
        for timer in timerStats:
            with timer:
                convolve_gcc_parallel(array7, kernel, array0, num_threads=8)
                err = np.max(np.abs(array7 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[7] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 8 & 9: Cuda with pycuda, without and with a constant memory kernel.

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

#import pycuda.gpuarray as gpuarray

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
extern "C" {
const unsigned int MAX_KH = %d;
const unsigned int MAX_KW = %d;
"""%(kernel.shape[0], kernel.shape[1])

cu_code += r"""
__constant__ float cmem_kernel[MAX_KH*MAX_KW];

__global__
void convolve_cuda(float* array, float* kernel, float* array0,
                  int h, int w, int kh, int kw, int h0, int w0)
{
    DEF_IDX(w, h) // int idx is defined here! // DEF_IDX(width, height)
    int u = idx / w; // *idx = y * width + x;
    int v = idx % w;

    if (u >= h - kh || v >= w - kw) return;

    float accum = 0.0f;
    for (int i = 0; i < kh; i++) {
        for (int j = 0; j < kw; j++) {
            accum += array0[(u + i) * w0 + (v + j)] * kernel[i * kw + j];
        }
    }
    array[u * w + v] = accum;
}

__global__
void convolve_cuda_cmem(float* array, float* array0,
                    int h, int w, int kh, int kw, int h0, int w0)
    {
    DEF_IDX(w, h) // int idx is defined here! // DEF_IDX(width, height)
    int u = idx / w; // *idx = y * width + x;
    int v = idx % w;

    if (u >= h - kh || v >= w - kw) return;

    float accum = 0.0f;
    for (int i = 0; i < kh; i++) {
        for (int j = 0; j < kw; j++) {
            accum += array0[(u + i) * w0 + (v + j)] * cmem_kernel[i * kw + j];
        }
    }
    array[u * w + v] = accum;
}

} // extern "C"
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

def convolve_cuda(array, kernel, array0):
    arr_c = np.ascontiguousarray(array, dtype=np.float32)
    kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

    c_array = arr_c.ravel().astype(np.float32)
    g_array = drv.mem_alloc(c_array.nbytes)
    drv.memcpy_htod(g_array, c_array)

    c_array0 = arr0_c.ravel().astype(np.float32)
    g_array0 = drv.mem_alloc(c_array0.nbytes)
    drv.memcpy_htod(g_array0, c_array0)

    c_kernel = kr_c.ravel().astype(np.float32)
    g_kernel = drv.mem_alloc(c_kernel.nbytes)
    drv.memcpy_htod(g_kernel, c_kernel)

    block_size = (16, 16, 1)
    grid_size = (
        (array.shape[1] + block_size[0] - 1) // block_size[0],
        (array.shape[0] + block_size[1] - 1) // block_size[1],
        1,
    )

    cu_convolve_cuda = krnl.get_function("convolve_cuda")
    cu_convolve_cuda(g_array, g_kernel, g_array0,
        np.int32(array.shape[0]), np.int32(array.shape[1]),
        np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
        np.int32(array0.shape[0]), np.int32(array0.shape[1]),
        block=block_size,
        grid=grid_size
    )

    drv.memcpy_dtoh(c_array, g_array)
    res = c_array.reshape(array.shape)

    #if arr_c is not array:
    array[:] = res
    return res

def convolve_cuda_cmem(array, kernel, array0):
    arr_c = np.ascontiguousarray(array, dtype=np.float32)
    kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
    arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

    c_array = arr_c.ravel().astype(np.float32)
    g_array = drv.mem_alloc(c_array.nbytes)
    drv.memcpy_htod(g_array, c_array)

    c_array0 = arr0_c.ravel().astype(np.float32)
    g_array0 = drv.mem_alloc(c_array0.nbytes)
    drv.memcpy_htod(g_array0, c_array0)

    c_kernel = kr_c.ravel().astype(np.float32)
    g_cmem_kernel, _ = krnl.get_global("cmem_kernel")
    drv.memcpy_htod(g_cmem_kernel, c_kernel)

    block_size = (16, 16, 1)
    grid_size = (
        (array.shape[1] + block_size[0] - 1) // block_size[0],
        (array.shape[0] + block_size[1] - 1) // block_size[1],
        1,
    )

    cu_convolve_cuda_cmem = krnl.get_function("convolve_cuda_cmem")
    cu_convolve_cuda_cmem(g_array, g_array0,
        np.int32(array.shape[0]), np.int32(array.shape[1]),
        np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
        np.int32(array0.shape[0]), np.int32(array0.shape[1]),
        block=block_size,
        grid=grid_size
    )

    drv.memcpy_dtoh(c_array, g_array)
    res = c_array.reshape(array.shape)

    #if arr_c is not array:
    array[:] = res
    return res

def test8_cuda():
    array8 = array_dst.copy()
    with TimerStats(prefix="cuda convolution", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution warmup"):
            convolve_cuda(array8, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_cuda(array8, kernel, array0)
                err = np.max(np.abs(array8 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[8] = timerStats.stats
    print()
    globals().update(locals())

def test9_cuda_cmem():
    array9 = array_dst.copy()
    with TimerStats(prefix="cuda convolution with constant memory kernel", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution with constant memory kernel warmup"):
            convolve_cuda_cmem(array9, kernel, array0)
        for timer in timerStats:
            with timer:
                convolve_cuda_cmem(array9, kernel, array0)
                err = np.max(np.abs(array9 - array1))
                if not timerStats.suppress:
                    print(" "*4 + "max error:", err)
    Statistics_Results[9] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 10, 11, 12, & 13: Cuda with and without constant memory using transfer-free calls & queues

def test10_cuda_notransfer():
    array10 = array_dst.copy()
    with TimerStats(prefix="cuda convolution no transfer", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution no transfer warmup"):
            array = array10
            arr_c = np.ascontiguousarray(array, dtype=np.float32)
            kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
            arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

            c_array = arr_c.ravel().astype(np.float32)
            g_array = drv.mem_alloc(c_array.nbytes)
            drv.memcpy_htod(g_array, c_array)

            c_array0 = arr0_c.ravel().astype(np.float32)
            g_array0 = drv.mem_alloc(c_array0.nbytes)
            drv.memcpy_htod(g_array0, c_array0)

            c_kernel = kr_c.ravel().astype(np.float32)
            g_kernel = drv.mem_alloc(c_kernel.nbytes)
            drv.memcpy_htod(g_kernel, c_kernel)

            block_size = (16, 16, 1)
            grid_size = (
                (array.shape[1] + block_size[0] - 1) // block_size[0],
                (array.shape[0] + block_size[1] - 1) // block_size[1],
                1,
            )

            cu_convolve_cuda = krnl.get_function("convolve_cuda")
            cu_convolve_cuda(g_array, g_kernel, g_array0,
                np.int32(array.shape[0]), np.int32(array.shape[1]),
                np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                block=block_size,
                grid=grid_size
            )

        for timer in timerStats:
            with timer:
                cu_convolve_cuda(g_array, g_kernel, g_array0,
                    np.int32(array.shape[0]), np.int32(array.shape[1]),
                    np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                    np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                    block=block_size,
                    grid=grid_size
                )
    Statistics_Results[10] = timerStats.stats
    print()
    globals().update(locals())

def test11_cuda_cmem_notransfer():
    array11 = array_dst.copy()
    with TimerStats(prefix="cuda convolution with const memory and no transfer", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution with const memory and no transfer warmup"):
            array = array11
            arr_c = np.ascontiguousarray(array, dtype=np.float32)
            kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
            arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

            c_array = arr_c.ravel().astype(np.float32)
            g_array = drv.mem_alloc(c_array.nbytes)
            drv.memcpy_htod(g_array, c_array)

            c_array0 = arr0_c.ravel().astype(np.float32)
            g_array0 = drv.mem_alloc(c_array0.nbytes)
            drv.memcpy_htod(g_array0, c_array0)

            c_kernel = kr_c.ravel().astype(np.float32)
            g_cmem_kernel, _ = krnl.get_global("cmem_kernel")
            drv.memcpy_htod(g_cmem_kernel, c_kernel)

            block_size = (16, 16, 1)
            grid_size = (
                (array.shape[1] + block_size[0] - 1) // block_size[0],
                (array.shape[0] + block_size[1] - 1) // block_size[1],
                1,
            )

            cu_convolve_cuda_cmem = krnl.get_function("convolve_cuda_cmem")
            cu_convolve_cuda_cmem(g_array, g_array0,
                np.int32(array.shape[0]), np.int32(array.shape[1]),
                np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                block=block_size,
                grid=grid_size
            )

        for timer in timerStats:
            with timer:
                cu_convolve_cuda_cmem(g_array, g_array0,
                    np.int32(array.shape[0]), np.int32(array.shape[1]),
                    np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                    np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                    block=block_size,
                    grid=grid_size
                )
    Statistics_Results[11] = timerStats.stats
    print()
    globals().update(locals())

def test12_cuda_streams():
    array12 = array_dst.copy()
    with TimerStats(prefix="cuda convolution streams", multiplier=1/1000, **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution streams warmup"):
            array = array12
            arr_c = np.ascontiguousarray(array, dtype=np.float32)
            kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
            arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

            c_array = arr_c.ravel().astype(np.float32)
            g_array = drv.mem_alloc(c_array.nbytes)
            drv.memcpy_htod(g_array, c_array)

            c_array0 = arr0_c.ravel().astype(np.float32)
            g_array0 = drv.mem_alloc(c_array0.nbytes)
            drv.memcpy_htod(g_array0, c_array0)

            c_kernel = kr_c.ravel().astype(np.float32)
            g_kernel = drv.mem_alloc(c_kernel.nbytes)
            drv.memcpy_htod(g_kernel, c_kernel)

            block_size = (16, 16, 1)
            grid_size = (
                (array.shape[1] + block_size[0] - 1) // block_size[0],
                (array.shape[0] + block_size[1] - 1) // block_size[1],
                1,
            )

            cu_convolve_cuda = krnl.get_function("convolve_cuda")
            cu_convolve_cuda(g_array, g_kernel, g_array0,
                np.int32(array.shape[0]), np.int32(array.shape[1]),
                np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                block=block_size,
                grid=grid_size
            )

        for timer in timerStats:
            with timer:
                stream = drv.Stream()
                for launch in range(1000):
                    cu_convolve_cuda(g_array, g_kernel, g_array0,
                        np.int32(array.shape[0]), np.int32(array.shape[1]),
                        np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                        np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                        block=block_size,
                        grid=grid_size,
                        stream=stream
                    )
                stream.synchronize()
    Statistics_Results[12] = timerStats.stats
    print()
    globals().update(locals())

def test13_cuda_cmem_streams():
    array13 = array_dst.copy()
    with TimerStats(prefix="cuda convolution with const memory and streams", multiplier=1/1000, **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda convolution with const memory and streams warmup"):
            array = array13
            arr_c = np.ascontiguousarray(array, dtype=np.float32)
            kr_c = np.ascontiguousarray(kernel.ravel(), dtype=np.float32)
            arr0_c = np.ascontiguousarray(array0, dtype=np.float32)

            c_array = arr_c.ravel().astype(np.float32)
            g_array = drv.mem_alloc(c_array.nbytes)
            drv.memcpy_htod(g_array, c_array)

            c_array0 = arr0_c.ravel().astype(np.float32)
            g_array0 = drv.mem_alloc(c_array0.nbytes)
            drv.memcpy_htod(g_array0, c_array0)

            c_kernel = kr_c.ravel().astype(np.float32)
            g_cmem_kernel, _ = krnl.get_global("cmem_kernel")
            drv.memcpy_htod(g_cmem_kernel, c_kernel)

            block_size = (16, 16, 1)
            grid_size = (
                (array.shape[1] + block_size[0] - 1) // block_size[0],
                (array.shape[0] + block_size[1] - 1) // block_size[1],
                1,
            )

            cu_convolve_cuda_cmem = krnl.get_function("convolve_cuda_cmem")
            cu_convolve_cuda_cmem(g_array, g_array0,
                np.int32(array.shape[0]), np.int32(array.shape[1]),
                np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                block=block_size,
                grid=grid_size
            )

        for timer in timerStats:
            with timer:
                stream = drv.Stream()
                for launch in range(1000):
                    cu_convolve_cuda_cmem(g_array, g_array0,
                        np.int32(array.shape[0]), np.int32(array.shape[1]),
                        np.int32(kernel.shape[0]), np.int32(kernel.shape[1]),
                        np.int32(array0.shape[0]), np.int32(array0.shape[1]),
                        block=block_size,
                        grid=grid_size,
                        stream=stream
                    )
                stream.synchronize()
    Statistics_Results[13] = timerStats.stats
    print()
    globals().update(locals())

################################################################################
# Test 14: Memory transfers only (Not included in original test)

def test14_cuda_cmem_streams():
    array14 = array_dst.copy()
    from astropy.io import fits
    import pathlib

    ROOT = pathlib.Path(".")
    DATA_DIR = ROOT / "DIA" / "TESS_sector_4"
    #OUT_CSV = ROOT / "sector4_index.csv"

    fits_files = sorted(DATA_DIR.glob("*.fits"), key=lambda file: file.stat().st_mtime)
    fits_files = fits_files[:-1] # Cheap way of excluding the incomplete download

    with TimerStats(prefix="cuda transfers test", **TimerStats_Args) as timerStats:
        # Do one warmup run to compile the function
        with Timer("cuda transfers test warmup"):
            with fits.open(fits_files[0], memmap=True) as hdul:
                src = hdul[1].data

            arr0_c = np.ascontiguousarray(src, dtype=np.float32)

            c_array0 = arr0_c.ravel().astype(np.float32)
            g_array0 = drv.mem_alloc(c_array0.nbytes)
            drv.memcpy_htod(g_array0, c_array0)

        for timer in timerStats:
            with timer:
                path = np.random.choice(fits_files)
                with fits.open(path, memmap=True) as hdul:
                    src = hdul[1].data
                    c_array0[:] = np.ascontiguousarray(src, dtype=np.float32).ravel()
                    drv.memcpy_htod(g_array0, c_array0)
    #Statistics_Results[14] = timerStats.stats
    stats = timerStats.stats
    print()
    globals().update(locals())

################################################################################

import json
"""
test1_pure_python()
test2_numba()
test3_numba_parallel()
test4_cppyy()
test5_cppyy_parallel()
test6_gcc()
test7_gcc_parallel()
test8_cuda()
test9_cuda_cmem()
test10_cuda_notransfer()
test11_cuda_cmem_notransfer()
test12_cuda_streams()
test13_cuda_cmem_streams()

# Write statistics to a json
for key in Statistics_Results.keys():
    Statistics_Results[key] = Statistics_Results[key].tolist()
with open("don_dia_perf_results3.json", 'w') as o:
    json.dump(Statistics_Results, o)
raise Exception
#"""

# Load from the json
"""
with open("don_dia_perf_results3.json", 'r') as o:
    Statistics_Results = json.load(o)
for key in Statistics_Results.keys():
    Statistics_Results[key] = np.array(Statistics_Results[key], dtype=np.float32)
"""
Statistics_Multipliers = {
    '1': 256.0,
    '12': 1/1000,
    '13': 1/1000,
}

import matplotlib.pyplot as plt

def plots():
    # Assume all statistics are gamma distributed
    fig, ax = plt.subplots(figsize=(12, 6))

    names = {
        "1": "Test 1: pure python",
        "2": "Test 2: numba",
        "3": "Test 3: numba parallel",
        "4": "Test 4: cppyy",
        "5": "Test 5: cppyy parallel",
        "6": "Test 6: gcc",
        "7": "Test 7: gcc parallel",
        "8": "Test 8: cuda",
        "9": "Test 9: cuda constant memory kernel",
        "10": "Test 10: cuda no transfer",
        "11": "Test 11: cuda constant memory kernel no transfer",
        "12": "Test 12: cuda streaming",
        "13": "Test 13: cuda constant memory kernel streaming",
    }

    for idx, (key, stats) in enumerate(Statistics_Results.items()):
        if len(stats) == 0:
            continue
        # Fit a gamma distribution to the data
        from scipy.stats import gamma
        from scipy.special import gammaln

        def gamma_logpdf(x, k, theta):
            return (k-1)*np.log(x) - x/theta - k*np.log(theta) - gammaln(k)

        def stable_pdf_from_logpdf(logpdf):
            m = np.max(logpdf)
            return np.exp(logpdf - m) * np.exp(m)
        
        import math
        try:
            shape, loc, scale = gamma.fit(stats, floc=0)
        except Exception:
            # Build shape and scale from the mean and std instead
            m = np.mean(stats)
            v = np.var(stats)
            loc = 0
            scale = v/m
            shape = (m*m) / v

        # from scipy.stats import loggamma
        # shape, loc, scale = loggamma.fit(np.log(stats), floc=0)
        val_min = np.min(stats)
        val_max = np.max(stats)

        #val_min = np.min([np.min(np.log(stat)) for stat in Statistics_Results.values()])
        #val_max = np.max([np.max(np.log(stat)) for stat in Statistics_Results.values()])

        val_mid = (val_min + val_max) / 2
        val_diff = (val_max - val_min) / 2

        # Set val_diff to be 5x the standard deviation
        ab2 = shape*scale**2
        std = ab2**.5
        val_diff = std*5.0

        # Set val_mid to be the maximum likelihood
        tipx = scale*(shape-1)
        if tipx < 0.0:
            tipx = np.mean(stats)
        val_mid = tipx

        val_min = val_mid - val_diff * 1.1  # Adjust val_min to be centered around val_mid
        val_max = val_mid + val_diff * 1.1  # Adjust val_max to be centered around val_mid
        val_min = max(val_min, 0)  # Ensure val_min is not negative

        x = np.linspace(val_min, val_max, 1000)
        #x = np.exp(np.linspace(val_min, val_max, 1000))
        #pdf = gamma.pdf(x, shape, loc=loc, scale=scale)
        #pdf = np.exp(shape*np.log(x) - x/scale) / (scale**shape * math.gamma(shape))
        lpdf = gamma_logpdf(x, shape, scale)
        pdf = stable_pdf_from_logpdf(lpdf)
        mpdf = np.exp(gamma_logpdf(tipx, shape, scale))
        # Approximate the normalization constant as a gaussian haha
        filtered_log_std = np.std(np.log(stats[np.where(stats > 0)]))
        #filtered_log_std = max(1/150, filtered_log_std)
        mpdf *= filtered_log_std

        pdf = pdf / mpdf
        #pdf = loggamma.pdf(np.log(x), shape, loc=loc, scale=scale)
        ax.plot(x, pdf, label=f"{names[key]}", color=f"C{idx}")

        #pdf_pts = gamma.pdf(stats, shape, loc=loc, scale=scale)
        #pdf_pts = loggamma.pdf(np.log(stats), shape, loc=loc, scale=scale)
        lpdf_pts = gamma_logpdf(stats, shape, scale)
        pdf_pts = stable_pdf_from_logpdf(lpdf_pts)
        pdf_pts /= mpdf
        pdf_pts *= np.random.random(len(pdf_pts))
        ax.scatter(stats, pdf_pts, color=f"C{idx}", s=1, alpha=0.5)
        
        #tipy = gamma.pdf(tipx, shape, loc=loc, scale=scale)
        #tipy = np.exp(shape*np.log(tipx) - tipx/scale) / (scale**shape * math.gamma(shape))
        #lpdf = gamma_logpdf(x, shape, scale)
        #pdf = stable_pdf_from_logpdf(lpdf)
        #ax.scatter(tipx, tipy, color=f"C{idx}", alpha=1, marker='v')

        print("Max likelihood of", names[key], "=", humanize.metric(tipx,'s'), '±', humanize.metric(std,'s'))

    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability Density")
    ax.set_title("Timing Distributions for Convolution Implementations")
    ax.legend()
    plt.show()


#plots()
