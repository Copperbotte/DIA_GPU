
# Joseph Kessler
# 2026 March 16
# hcongrid_testing.py
########################################################################################################################
# This is a testing file containing transformations for hcongrid, cleaned from tan_sip_to_icrs.py.

# Overview below written by Claude Opus 4.6 on 2026 March 14.
"""
tan_sip_to_icrs.py — Trig-free TAN-SIP pixel → unit ICRS vec3

Maps every pixel in a TESS FFI to a unit direction vector on the ICRS
celestial sphere, incorporating the full SIP distortion polynomial.

    ┌─────────────────────────────────────────────────────────┐
    │  Per-pixel cost:  ~55 FLOPs + 1 rsqrt                   │
    │  Transcendentals: ZERO                                  │
    │  Branches:        ZERO                                  │
    │                                                         │
    │  WCSLIB equivalent: ~30 FLOPs + 8 trig + 3-4 branches   │
    └─────────────────────────────────────────────────────────┘

The trick: instead of deprojecting to spherical (φ, θ) and then
rotating with trig, we go directly to Cartesian and rotate with a
matrix.  The TAN (gnomonic) projection inverse is just building a
3D direction vector from the 2D tangent-plane point:

    native_vec3 = normalize(-y, x, 180/π)

No trig needed.  The rotation from the native tangent-plane frame to
ICRS is a precomputed 3×3 matrix multiply (9 FMA).

This is exactly what a GPU vertex shader does:
    model → world → clip space = matrix multiplies + perspective divide.
In our case:
    pixel → SIP → CD matrix → tangent plane → ICRS = polynomial + mat2×vec2 + normalize + mat3×vec3.

Pipeline
========

    pixel (col, row)                    ← 0-based numpy array coords
        │
        ├── subtract CRPIX              ← 2 sub
        │   u = col - (CRPIX1 - 1)
        │   v = row - (CRPIX2 - 1)
        │
        ├── SIP distortion polynomial   ← ~25 FLOPs (12 terms per axis)
        │   u' = u + Σ A_ij · u^i · v^j
        │   v' = v + Σ B_ij · u^i · v^j
        │
        ├── CD matrix (2×2)             ← 6 FLOPs
        │   x = CD1_1·u' + CD1_2·v'
        │   y = CD2_1·u' + CD2_2·v'
        │
        ├── TAN deprojection            ← 0 trig!  just build the vector
        │   native = (-y, x, 180/π)       unnormalized direction on unit sphere
        │
        ├── Rotation matrix (3×3)       ← 9 FMA
        │   w = R · native               R precomputed from CRVAL1, CRVAL2
        │
        └── Normalize                   ← 1 rsqrt + 3 mul
            v_icrs = w / |w|

Usage
=====

    from tan_sip_to_icrs import build_wcs_transform, pixel_to_unit_icrs
    from astropy.io import fits

    header = fits.getheader('tess_ffi.fits', ext=1)
    T = build_wcs_transform(header)

    # Single point
    vx, vy, vz = pixel_to_unit_icrs(1000, 1000, T)

    # Full image map (2048 × 2048 × 3)
    icrs_map = make_icrs_map(T)

    # Check corners
    print_corners(T)

    # Validate against astropy
    validate(header)
"""

import numpy as np
import matplotlib.pyplot as plt
import time

# IO
from os.path import isfile, join
from astropy.io import fits
from astropy.wcs import WCS

# Numba Accelerator
from collections import namedtuple
import numba
#njargs = dict(parallel=True, fastmath=True, nogil=True)
njargs = dict(parallel=True, nogil=True) # Fastmath seems to be slower?
#njargs = dict(parallel=True) # nogil is as fast as numpy native. Drat! Unless the bulk of the code is the distort.

# Cuda Accelerator
# Cuda will be initialized in the cuda block.

# Profile timing
from time import time_ns
import humanize
# Usage: Create a TimerGroup with a set of keys.
# Then for each key, we call 'from timer_group with key'. Is that valid syntax?
# Can I instead do with timer_group['key'] and have it persist state?
class ProfileTimer_Inner:
    def __init__(self, parent, key):
        self.parent = parent
        self.key = key
        self.enable = True

    def __enter__(self):
        self.parent.start_times[self.key] = time_ns()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self.parent.start_times[self.key] is not None and self.enable:
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
        return {key: [np.mean(elapsed[1:]), np.std(elapsed[1:]), len(elapsed)-1] for key, elapsed in secs.items()}  # Return average time for each key
    
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
        return '\n'.join(f'{param[0]}: {param[1]} ± {param[2]} seconds ({param[5]}) for n={param[3]} in {param[4]} seconds' for param in params)
        #return '\n'.join(f'{key}: {elapsed[0]:.6f} ± {elapsed[1]:.6f} seconds for n={elapsed[2]} in {elapsed[0]*elapsed[2]:.6f} seconds' for key, elapsed in elapsed_times.items())

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  File IO
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

def get_file_list(camera=None, ccd=None):
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

    files = [f for f in rawdir.glob("*.fits") if isfile(join(rawdir, f))] #gets the relevant files with the proper extension

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

def make_file_generator(files):
    for f in files:
        yield fits.getdata(f, header=True)

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Setup — runs ONCE per WCS header (precompute all constants)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Evaluate SIP Distortion — runs ONCE per pixel (the bottleneck, so optimized with numba)
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

@numba.njit(**njargs)
def distort_sip(distort, vb, AB):
    for n in numba.prange(vb.shape[0]):
        u, v = vb[n]
        # Manually construct the power sets
        up = np.empty((5,), dtype=np.float64)
        vp = np.empty((5,), dtype=np.float64)
        up[0] = 1.0
        vp[0] = 1.0
        for i in range(1, 5):
            up[i] = u*up[i-1]
            vp[i] = v*vp[i-1]
        
        # Einsum
        distort[n,0] = 0.0
        distort[n,1] = 0.0
        for k in range(2, 5): # 2, 3, 4
            for i in range(k + 1): # 0..k
                j = k - i # because i+j=k. # Suppose k=2. Then, [i,j] = [0,2], [1,1], [2,0]. Perfect! This way we avoid the if-branch and just compute the powers as we go.
                distort[n,0] += AB[0,i,j] * up[i] * vp[j]
                distort[n,1] += AB[1,i,j] * up[i] * vp[j]
    #vb[:] += distort # Apply the SIP distortion to the pixel coordinates
    #return distort

@numba.njit(**njargs)
def distort_sip_inv_newton(result, out_ni, AB, iAB):
    """Invert the forward SIP distortion out = vb + SIP_fwd(vb) via Newton-Raphson.
    
    Given 'out' (distorted coords), find 'vb' (undistorted coords) such that
        vb + SIP_fwd(vb) = out
    
    Uses the AP/BP inverse polynomial as a warm start, then refines with
    Gauss-Newton using the analytical 2×2 Jacobian of the forward SIP.
    
    Parameters
    ----------
    result   : (N, 2) output — the recovered undistorted pixel coords
    out_ni   : (N, 2) input  — the distorted (post-SIP-forward) coords
    AB       : (2, 5, 5) forward SIP coefficients
    iAB      : (2, 5, 5) inverse SIP coefficients (AP/BP, for warm start)
    n_iter   : number of Newton refinement iterations (2-3 is typical) # Removed because numba may whine
    """
    n_iter=3 # Empirically, this has a max error of 3e-12. Machine precision! I'll take it.

    for n in numba.prange(out_ni.shape[0]):
        out_u = out_ni[n, 0]
        out_v = out_ni[n, 1]

        # ── Warm start: AP/BP inverse polynomial ──
        up = np.empty((5,), dtype=np.float64)
        vp = np.empty((5,), dtype=np.float64)
        up[0] = 1.0
        vp[0] = 1.0
        for p in range(1, 5):
            up[p] = out_u * up[p - 1]
            vp[p] = out_v * vp[p - 1]
        
        du0 = 0.0
        dv0 = 0.0
        for j in range(5):
            for i in range(5 - j):
                term_u = up[i] * vp[j]
                du0 += iAB[0, i, j] * term_u
                dv0 += iAB[1, i, j] * term_u

        # vb estimate = out + inverse_SIP(out)
        gu = out_u + du0
        gv = out_v + dv0

        # ── Newton iterations: solve f(vb) = vb + SIP_fwd(vb) - out = 0 ──
        for _ in range(n_iter):
            # Build power arrays at current guess
            up[0] = 1.0
            vp[0] = 1.0
            for p in range(1, 5):
                up[p] = gu * up[p - 1]
                vp[p] = gv * vp[p - 1]

            # Evaluate SIP_fwd and its 2×2 Jacobian simultaneously
            fu = 0.0   # SIP_fwd_u
            fv = 0.0   # SIP_fwd_v
            # J = I + dSIP/dvb, so we accumulate only the dSIP part
            j00 = 0.0  # ∂SIP_u/∂u
            j01 = 0.0  # ∂SIP_u/∂v
            j10 = 0.0  # ∂SIP_v/∂u
            j11 = 0.0  # ∂SIP_v/∂v

            #for j in range(5):
            #    for i in range(5 - j):
            for k in range(2, 5): # 2, 3, 4
                for i in range(k + 1): # 0..k
                    j = k - i # because i+j=k. # Suppose k=2. Then, [i,j] = [0,2], [1,1], [2,0]. Perfect! This way we avoid the if-branch and just compute the powers as we go.
                    
                    term = up[i] * vp[j]
                    a = AB[0, i, j]
                    b = AB[1, i, j]
                    fu += a * term
                    fv += b * term
                    if i > 0:
                        dterm_du = i * up[i - 1] * vp[j]
                        j00 += a * dterm_du
                        j10 += b * dterm_du
                    if j > 0:
                        dterm_dv = j * up[i] * vp[j - 1]
                        j01 += a * dterm_dv
                        j11 += b * dterm_dv

            # Residual: f(vb) = vb + SIP_fwd(vb) - out
            ru = gu + fu - out_u
            rv = gv + fv - out_v

            # Full Jacobian: J = I + dSIP/dvb
            j00 += 1.0
            j11 += 1.0

            # Invert 2×2 Jacobian analytically: J⁻¹ = adj(J) / det(J)
            det = j00 * j11 - j01 * j10
            inv_det = 1.0 / det
            # Update: vb <- vb - J⁻¹ @ residual
            du = inv_det * (j11 * ru - j01 * rv)
            dv = inv_det * (j00 * rv - j10 * ru)
            #if n == 0:
            #    print(f"Iter {_}: ru={ru:.3e}, rv={rv:.3e}, du={du:.3e}, dv={dv:.3e}, det={det:.3e}")
            gu -= du
            gv -= dv
            #gu -= inv_det * (j11 * ru - j01 * rv)
            #gv -= inv_det * (j00 * rv - j10 * ru)

        result[n, 0] = gu
        result[n, 1] = gv

def wcs_map_ICRS_from_TESS_SLOW(src, src_pixel_ni):
    """
    Convert a vertex buffer to dst from src's coordinate systems, with end-to-end distortion corrections. This is NOT high performance compute.
    """

    R0 = 180.0 / np.pi

    uv_ni = src_pixel_ni - (src.ref_px_coord_i - 1)# / 2 # Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    distort_ni = np.empty_like(uv_ni)
    distort_sip(distort_ni, uv_ni, src.fwd_distortion_coeffs_kij)
    uv_ni += distort_ni
    xy_ni = (src.cd_matrix_ij @ uv_ni.T).T
    xyz_ni = np.pad(xy_ni, ((0,0),(0,1)))
    xyz_ni[:,2] = R0

    # Rotate to ICRS. This is where we halt!
    icrs_xyz_ni = (src.R_ij @ xyz_ni.T).T
    return icrs_xyz_ni


def wcs_map_TESS_from_ICRS_SLOW(dst, icrs_xyz_ni):
    """
    Convert a vertex buffer to dst from src's coordinate systems, with end-to-end distortion corrections. This is NOT high performance compute.
    """

    R0 = 180.0 / np.pi

    xyz_ni = (dst.R_ij.T @ icrs_xyz_ni.T).T
    xy_ni = xyz_ni[:,:2] * R0 / xyz_ni[:,2:]
    uv_ni = (dst.cd_matrix_inv_ij @ xy_ni.T).T
    distort_ni = np.empty_like(uv_ni)
    #distort_sip(distort_ni, uv_ni, dst.inv_distortion_coeffs_kij)
    #distort_ni += uv_ni #uv_ni += distort_ni # Plus, or minus here?
    distort_sip_inv_newton(distort_ni, uv_ni, dst.fwd_distortion_coeffs_kij, dst.inv_distortion_coeffs_kij)
    # distort_ni is now uv_ni for this algorithm!
    dst_pixel_ni = distort_ni + (dst.ref_px_coord_i - 1)
    return dst_pixel_ni


# Variant of above with constant memory access
@numba.njit(**njargs)
def _wcs_map_ICRS_from_TESS(icrs_xyz_ni, distort_ni, 
    src__naxis_i, src__ref_px_coord_i, src__fwd_distortion_coeffs_kij, src__cd_matrix_ij, src__R_ij, # src, 
    src_pixel_ni):

    R0 = 180.0 / np.pi

    #src_pixel_ni[:] -= (src__ref_px_coord_i - 1).astype(src_pixel_ni.dtype) # / 2 # Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    for n in numba.prange(src_pixel_ni.shape[0]):
        for i in numba.prange(src_pixel_ni.shape[1]):
            src_pixel_ni[n,i] -= src__ref_px_coord_i[i] - 1.0
            #src_pixel_ni[n,i] = src_pixel_ni[n,i]*src__naxis_i[i] - (src__ref_px_coord_i[i] - 1.0)

    distort_sip(distort_ni, src_pixel_ni, src__fwd_distortion_coeffs_kij)

    for n in numba.prange(src_pixel_ni.shape[0]):

        for i in range(src_pixel_ni.shape[1]):
            src_pixel_ni[n,i] += distort_ni[n,i]
        
        #icrs_xyz_ni[n,:2] = src__cd_matrix_ij @ src_pixel_ni[n]
        for i in range(src_pixel_ni.shape[1]):
            icrs_xyz_ni[n,i] = 0.0
            for j in range(src_pixel_ni.shape[1]):
                icrs_xyz_ni[n,i] += src__cd_matrix_ij[i,j] * src_pixel_ni[n,j]
        
        icrs_xyz_ni[n,2] = R0

        # Rotate to ICRS. This is where we halt!
        #icrs_xyz_ni[n] = src__R_ij @ icrs_xyz_ni[n]
        for i in range(icrs_xyz_ni.shape[1]):
            #icrs_xyz_ni[n,i] = 0.0
            tmp = np.zeros_like(icrs_xyz_ni[n])
            for j in range(icrs_xyz_ni.shape[1]):
                #icrs_xyz_ni[n,i] += src__R_ij[i,j] * icrs_xyz_ni[n,j]
                tmp[i] += src__R_ij[i,j] * icrs_xyz_ni[n,j]
            icrs_xyz_ni[n,i] = tmp[i]

def wcs_map_ICRS_from_TESS(icrs_xyz_ni, distort_ni, src, src_pixel_ni):
    # Unpack src as numba is unhappy
    _wcs_map_ICRS_from_TESS(
        icrs_xyz_ni,
        distort_ni,
        src.naxis_i,
        src.ref_px_coord_i,
        src.fwd_distortion_coeffs_kij,
        src.cd_matrix_ij,
        src.R_ij,
        src_pixel_ni
    )

@numba.njit(**njargs)
def _wcs_map_TESS_from_ICRS(dst_pixel_ni, distort_ni, 
    dst__naxis_i, dst__ref_px_coord_i, dst__fwd_distortion_coeffs_kij,dst__inv_distortion_coeffs_kij, dst__cd_matrix_inv_ij, dst__R_ij, # dst, 
    icrs_xyz_ni):
    R0 = 180.0 / np.pi

    #icrs_xyz_ni[:] = (dst__R_ij.T @ icrs_xyz_ni.T).T
    for n in numba.prange(icrs_xyz_ni.shape[0]):
        accum = np.zeros((icrs_xyz_ni.shape[1],), dtype=icrs_xyz_ni.dtype)
        for i in range(icrs_xyz_ni.shape[1]):
            for j in range(icrs_xyz_ni.shape[1]):
                accum[i] += dst__R_ij[j,i] * icrs_xyz_ni[n,j] # Transposed!
        #for i in range(icrs_xyz_ni.shape[1]):
        #    icrs_xyz_ni[n,i] = accum[i]
        
        # dst_pixel_ni[:] = icrs_xyz_ni[:,:2] * R0 / icrs_xyz_ni[:,2:]
        for i in range(dst_pixel_ni.shape[1]):
            #dst_pixel_ni[n,i] = icrs_xyz_ni[n,i] * R0 / icrs_xyz_ni[n,2]
            dst_pixel_ni[n,i] = accum[i] * R0 / accum[2]
            accum[i] = 0.0

        # dst_pixel_ni[:] = (dst__cd_matrix_inv_ij @ dst_pixel_ni.T).T
        for i in range(dst_pixel_ni.shape[1]):
            for j in range(dst_pixel_ni.shape[1]):
                accum[i] += dst__cd_matrix_inv_ij[i,j] * dst_pixel_ni[n,j]
            #dst_pixel_ni[n,i] = accum[i]
        for i in range(dst_pixel_ni.shape[1]):
            dst_pixel_ni[n,i] = accum[i]
    
    #distort_sip(distort_ni, dst_pixel_ni, dst__inv_distortion_coeffs_kij)
    distort_sip_inv_newton(distort_ni, dst_pixel_ni, dst__fwd_distortion_coeffs_kij, dst__inv_distortion_coeffs_kij)
    # distort_ni is now dst_pixel_ni for this algorithm!

    for n in numba.prange(dst_pixel_ni.shape[0]):
        # dst_pixel_ni += distort_ni # Plus, or minus here?
        # dst_pixel_ni[:] += (dst__ref_px_coord_i - 1).astype(dst_pixel_ni.dtype)
        for i in range(dst_pixel_ni.shape[1]):
            #dst_pixel_ni[n,i] += distort_ni[n,i]
            #dst_pixel_ni[n,i] += dst__ref_px_coord_i[i] - 1.0
            # Un-center and un-normalize the pixel coordinates. This is in [0, N-1] coordinates.
            #dst_pixel_ni[n,i] = (distort_ni[n,i] + dst_pixel_ni[n,i] + (dst__ref_px_coord_i[i] - 1.0))# / dst__naxis_i[i] 
            dst_pixel_ni[n,i] = (distort_ni[n,i] + (dst__ref_px_coord_i[i] - 1.0))# / dst__naxis_i[i] 

def wcs_map_TESS_from_ICRS(dst_pixel_ni, distort_ni, dst, icrs_xyz_ni):
    _wcs_map_TESS_from_ICRS(
        dst_pixel_ni,
        distort_ni,
        dst.naxis_i,
        dst.ref_px_coord_i,
        dst.fwd_distortion_coeffs_kij,
        dst.inv_distortion_coeffs_kij,
        dst.cd_matrix_inv_ij,
        dst.R_ij,
        icrs_xyz_ni)

# # Example use of this chain!
# keys = wcs_variables_validate(header)
# TESS_Mapping_Variables = wcs_variables_load_nonan(header, keys)
# TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)
# wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, np.array([[1000.0, 1000.0], [1500.0, 1500.0]]))


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  CUDA Initialization
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  CUDA Kernels
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

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

// Sketch plan of different required kernels
//def wcs_map_ICRS_from_TESS(icrs_xyz_ni, distort_ni, src, src_pixel_ni)
//def wcs_map_TESS_from_ICRS(dst_pixel_ni, distort_ni, dst, icrs_xyz_ni)
//def distort_sip(distort, vb, AB):
//def distort_sip_inv_newton(result, out_ni, AB, iAB):
// And one more! Cubic Sample.
// Samples:
// __global__
// void rbf_interpolate_cuda_f64(double* dst, double* src_xy, double* src_c) {
//     int col = threadIdx.x + blockDim.x * blockIdx.x;
//     int row = threadIdx.y + blockDim.y * blockIdx.y;
//     if (col >= (int)c_width_height[1] || row >= (int)c_width_height[0]) return;
//     int idx = row * (int)c_width_height[1] + col;
// 
//     double u = (double)col;  // XI[row, col] = col
//     double v = (double)row;  // YI[row, col] = row
// 
// __global__
// void convolve_cuda_cmem(float* array, float* array0,
//                     int h, int w, int kh, int kw, int h0, int w0)
//     {
//     DEF_IDX(w, h) // int idx is defined here! // DEF_IDX(width, height)
//     int u = idx / w; // *idx = y * width + x;
//     int v = idx % w;
// 
//     if (u >= h - kh || v >= w - kw) return;
// 
//     float accum = 0.0f;
//     for (int i = 0; i < kh; i++) {
//         for (int j = 0; j < kw; j++) {
//             accum += array0[(u + i) * w0 + (v + j)] * cmem_kernel[i * kw + j];
//         }
//     }
//     array[u * w + v] = accum;
// }

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

__constant__ double ref_px_coord_i[2];
__constant__ double cd_ij[4];
__constant__ double cd_inv_ij[4];
__constant__ double Rotation_ij[9]; // This is the R_ij matrix in the code above. Its a View matrix!
__constant__ double fwd_AB[2 * 25]; // 2 channels, 25 coefficients each for the 5th order polynomial (i+j<=4)
__constant__ double inv_AB[2 * 25]; // 2 channels, 25 coefficients each for the 5th order polynomial (i+j<=4)
__constant__ int N_pts;

__constant__ int img_shape[2]; // height, width (same as numpy)

__global__
void distort_sip_cuda_f64(double* distort, double* vb) {
    //int idx = threadIdx.x + blockDim.x * blockIdx.x;
    //if (idx >= N) return;
    DEF_IDX(N_pts, 1) // int idx is defined here! // DEF_IDX(width, height)

    vec2_distort_sip_cuda_f64(&distort[idx*2], &vb[idx*2], fwd_AB); 
}

__global__
void distort_sip_inv_newton_cuda_f64(
    double* __restrict__ result,
    double* __restrict__ out_ni
) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N_pts) return;
    
    vec2_distort_sip_inv_newton_cuda_f64(&result[idx*2], &out_ni[idx*2], fwd_AB, inv_AB);
}

__device__
void ICRS_from_TESS_cuda_f64(
    double* __restrict__ icrs_xyz, // double[3]
    double* __restrict__ src_pixel // double[2]
){
    // Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    double uv[2];
    for(int i=0; i<2; i++)
        uv[i] = src_pixel[i] - (ref_px_coord_i[i] - 1);
    
    double distort[2];
    vec2_distort_sip_cuda_f64(distort, uv, fwd_AB);

    double xyz[3];
    for(int i=0; i<2; i++)
        xyz[i] = 0.0;

    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            xyz[i] += cd_ij[i*2 + j] * (uv[j] + distort[j]);

    const double PI = 3.1415926535897932384626433832795;
    xyz[2] = 180.0 / PI;

    double tmp[3];
    for(int i=0; i<3; i++)
        tmp[i] = 0.0;

    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            tmp[i] += Rotation_ij[i*3 + j] * xyz[j];

    for(int i=0; i<3; i++)
        icrs_xyz[i] = tmp[i];
}

__global__
void wcs_map_ICRS_from_TESS_cuda_f64(
    double* __restrict__ icrs_xyz_ni,
    double* __restrict__ src_pixel_ni
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N_pts) return;

    ICRS_from_TESS_cuda_f64(&icrs_xyz_ni[idx*3], &src_pixel_ni[idx*2]);
}

__device__
void TESS_from_ICRS_cuda_f64(
    double* __restrict__ dst_pixel, // double[2]
    double* __restrict__ icrs_xyz // double[3]
){
    const double PI = 3.1415926535897932384626433832795;

    double xyz[3];
    for(int i=0; i<3; i++)
        xyz[i] = 0.0;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            xyz[i] += Rotation_ij[j*3 + i] * icrs_xyz[j]; // Transposed!

    // Projection!
    for(int i=0; i<2; i++)
        xyz[i] *= (180.0 / PI) / xyz[2];
    
    double uv[2];
    for(int i=0; i<2; i++)
        uv[i] = 0.0;
    for(int i=0; i<2; i++)
        for(int j=0; j<2; j++)
            uv[i] += cd_inv_ij[i*2 + j] * xyz[j];

    double distort[2];
    vec2_distort_sip_inv_newton_cuda_f64(distort, uv, fwd_AB, inv_AB);

    for(int i=0; i<2; i++)
        dst_pixel[i] = distort[i] + (ref_px_coord_i[i] - 1);
}

__global__
void wcs_map_TESS_from_ICRS_cuda_f64(
    double* __restrict__ dst_pixel_ni,
    double* __restrict__ icrs_xyz_ni
){
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N_pts) return;

    TESS_from_ICRS_cuda_f64(&dst_pixel_ni[idx*2], &icrs_xyz_ni[idx*3]);
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

__device__ __forceinline__
double textel_bilinear(
    const double* __restrict__ img,
    int ix, int iy,
    double px, double py, // in [0,1)
    int w, int h
) {
    double a00 = textel(img, ix,   iy,   w, h);
    double a10 = textel(img, ix+1, iy,   w, h);
    double a01 = textel(img, ix,   iy+1, w, h);
    double a11 = textel(img, ix+1, iy+1, w, h);

    double b0 = lerp(a00, a10, px);
    double b1 = lerp(a01, a11, px);

    return lerp(b0, b1, py);
}

__device__
double texture_bilinear(
    const double* __restrict__ img,
    double x, double y,
    int w, int h
) {
    double px = x - floor(x);
    double py = y - floor(y);
    
    int ix = (int)x;
    int iy = (int)y;

    return textel_bilinear(img, ix, iy, px, py, w, h);
}

__device__ __forceinline__
double textel_bicubic(
    const double* __restrict__ img,
    int ix, int iy,
    double px, double py, // in [0,1)
    int w, int h
) {
    double a00 = textel_bilinear(img, ix,   iy,   px, py, w, h);
    double a10 = textel_bilinear(img, ix+1, iy,   px, py, w, h);
    double a01 = textel_bilinear(img, ix,   iy+1, px, py, w, h);
    double a11 = textel_bilinear(img, ix+1, iy+1, px, py, w, h);

    double b0 = lerp(a00, a10, px);
    double b1 = lerp(a01, a11, px);

    return lerp(b0, b1, py);
}


__device__
double texture_bicubic(
    const double* __restrict__ img,
    double x, double y,
    int w, int h
) {
    double px = x - floor(x);
    double py = y - floor(y);
    
    int ix = (int)x;
    int iy = (int)y;

    return textel_bicubic(img, ix, iy, px, py, w, h);
}

// __device__
// double texture_catmull_rom_nested(
//     const double* __restrict__ img,
//     double x, double y,
//     int w, int h)
// {
//     int ix = (int)floor(x);
//     int iy = (int)floor(y);
//     double tx = x - (double)ix;
//     double ty = y - (double)iy;
// 
//     // Catmull-Rom weights
//     double wx[4], wy[4];
//     wx[0] = tx * (-0.5 + tx * ( 1.0 - 0.5 * tx));
//     wx[1] = 1.0 + tx * tx * (-2.5 + 1.5 * tx);
//     wx[2] = tx * ( 0.5 + tx * ( 2.0 - 1.5 * tx));
//     wx[3] = tx * tx * (-0.5 + 0.5 * tx);
// 
//     wy[0] = ty * (-0.5 + ty * ( 1.0 - 0.5 * ty));
//     wy[1] = 1.0 + ty * ty * (-2.5 + 1.5 * ty);
//     wy[2] = ty * ( 0.5 + ty * ( 2.0 - 1.5 * ty));
//     wy[3] = ty * ty * (-0.5 + 0.5 * ty);
// 
//     // Group pairs
//     double gx0 = wx[0] + wx[1];    double gy0 = wy[0] + wy[1];
//     double gx1 = wx[2] + wx[3];    double gy1 = wy[2] + wy[3];
//     double ax  = wx[1] / gx0;      double ay  = wy[1] / gy0;
//     double bx  = wx[3] / gx1;      double by  = wy[3] / gy1;
// 
//     // 4 bilinear fetches at SHIFTED coordinates
//     double B00 = textel_bilinear(img, ix-1, iy-1, ax, ay, w, h);
//     double B10 = textel_bilinear(img, ix+1, iy-1, bx, ay, w, h);
//     double B01 = textel_bilinear(img, ix-1, iy+1, ax, by, w, h);
//     double B11 = textel_bilinear(img, ix+1, iy+1, bx, by, w, h);
// 
//     // 3 weighted lerps (NOT with tx/ty — with g0/g1!)
//     double c0 = gx0 * B00 + gx1 * B10;
//     double c1 = gx0 * B01 + gx1 * B11;
// 
//     return gy0 * c0 + gy1 * c1;
// }

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

__global__ // I dont think I'm going to test this :)
void texture_resample_f64(
    double* __restrict__ dst_img,
    double* __restrict__ src_img,
    double* __restrict__ src_coords, // [N_pts, 2]
    int dst_h, 
    int dst_w
){
    int idx;
    if(mapidx(&idx, dst_w, dst_h)) return; // Map 2D thread/block indices to a 1D index, return if out of bounds
    // idx is now bounded and has the right index!

    //int x = idx % dst_w;
    //int y = idx / dst_w;

    // src_coords is in [0, N-1] coordinates (if valid), so we can directly use it for texturing.
    // the sampler automatically handles out-of-bounds by returning 0, so we don't need an explicit check here.

    dst_img[idx] = texture_catmull_rom(src_img, src_coords[idx*2], src_coords[idx*2 + 1], img_shape[1], img_shape[0]);
}

// Context for you mr ai :)
// struct __align__(8) TESS_Mapping_Struct {
//     int    img_shape[2];       // 8 bytes
//     double ref_px_coord[2];    // 16 bytes
//     double cd[4];              // 32 bytes
//     double cd_inv[4];          // 32 bytes
//     double R[9];               // 72 bytes
//     double fwd_AB[50];         // 400 bytes
//     double inv_AB[50];         // 400 bytes
// };

// The big one >:)
// __global__ // Remaps pixels from src onto dst. do NOT have dst's pixels loaded into memory, that's where this writes.
// void cuda_hcongrid_f64(
//     double* __restrict__ dst_img,
//     double* __restrict__ src_img,
//     TESS_Mapping_Struct* dst,
//     TESS_Mapping_Struct* src,
// ){
//     int idx;
//     if(mapidx(&idx, dst->img_shape[1], dst->img_shape[0])) return; // Map 2D thread/block indices to a 1D index, return if out of bounds
//     // idx is now bounded and has the right index!

//     // It looks like I dont need to allocate anything more than what's here. Yippee!

//     // Let's sketch the entire shader.
//     // 1. Generate a pixel coordinate for this thread. (dst's coordinates)
//     // 2. Map dst -> ICRS
//     // 3. Map ICRS -> src
//     // 4. Sample src at the mapped coordinate.
//     // 5. Write to dst.'
//     // 6. Yippee!

//     // 1. Generate pixel coordinate [0, N)
//     int x = idx % dst->img_shape[1]; // These pictures are always 2048x2048. I wonder if we should hardcode this. eh
//     int y = idx / dst->img_shape[1];

//     // 2. Map dst -> ICRS
//     double xy[2] = {(double)x, (double)y};
//     double icrs[3];
//     ICRS_from_TESS_cuda_f64(icrs, xy, dst);
//     
//     // 3. Map ICRS -> src
//     TESS_from_ICRS_cuda_f64(xy, icrs, src);

//     // 4. Sample src at the mapped coordinate.
//     double lum = texture_catmull_rom(src_img, xy[0], xy[1], src->img_shape[1], src->img_shape[0]);

//     // 5. Write to dst.
//     dst_img[idx] = lum;

//     // 6. Yippee!
// }

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

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Cuda!
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# Context for you mr ai :)
# struct __align__(8) TESS_Mapping_Struct {
#     int    img_shape[2];       // 8 bytes
#     double ref_px_coord[2];    // 16 bytes
#     double cd[4];              // 32 bytes
#     double cd_inv[4];          // 32 bytes
#     double Rotation[9];        // 72 bytes
#     double fwd_AB[50];         // 400 bytes
#     double inv_AB[50];         // 400 bytes
# };

# TESS_Mapping_Struct_dtype = np.dtype([
#     ('img_shape',      np.int32, (2,)),  #  8 bytes
#     ('ref_px_coord', np.float64, (2,)),  #  16 bytes
#     ('cd',           np.float64, (4,)),  #  32 bytes
#     ('cd_inv',       np.float64, (4,)),  #  32 bytes
#     ('Rotation',     np.float64, (9,)),  #  72 bytes
#     ('fwd_AB',       np.float64, (50,)), # 400 bytes
#     ('inv_AB',       np.float64, (50,)), # 400 bytes
# ], align=True)
# 
# def pack_TMS(TESS_Mapping_Struct, data):
#     """Pack a WCSMappingStruct into a single numpy buffer matching the CUDA struct."""
#     TMS = np.zeros(1, dtype=TESS_Mapping_Struct_dtype)
#     TMS['img_shape']    = np.array(TESS_Mapping_Struct.img_shape_i, dtype=np.int32)
#     TMS['ref_px_coord'] = TESS_Mapping_Struct.ref_px_coord_i.ravel().astype(np.float64)
#     TMS['cd']           = TESS_Mapping_Struct.cd_matrix_ij.ravel().astype(np.float64)
#     TMS['cd_inv']       = TESS_Mapping_Struct.cd_matrix_inv_ij.ravel().astype(np.float64)
#     TMS['Rotation']     = TESS_Mapping_Struct.R_ij.ravel().astype(np.float64)
#     TMS['fwd_AB']       = TESS_Mapping_Struct.fwd_distortion_coeffs_kij.ravel().astype(np.float64)
#     TMS['inv_AB']       = TESS_Mapping_Struct.inv_distortion_coeffs_kij.ravel().astype(np.float64)
#     return TMS

class CUDAContainer:
    def __init__(self):
        self.initialized = False
        self.ftype = np.float64
        #self.ftype = np.float32

    # Runs on the first call to initialize kernels as a "JIT" approach.
    def initialize(self, TESS_Mapping_Struct, vertex_buffer_ni):
        if self.initialized:
            return
        
        self.cu_distort_sip = krnl.get_function("distort_sip_cuda_f64")
        #self.cu_distort_sip = krnl.get_function("distort_sip_cuda_f32")
        self.cu_wcs_map_ICRS_from_TESS = krnl.get_function("wcs_map_ICRS_from_TESS_cuda_f64")
        self.cu_wcs_map_TESS_from_ICRS = krnl.get_function("wcs_map_TESS_from_ICRS_cuda_f64")

        # Initialize constant memory parameters
        c_ref_px_coord = TESS_Mapping_Struct.ref_px_coord_i.astype(self.ftype).ravel()
        g_ref_px_coord, g_ref_px_coord_size = krnl.get_global("ref_px_coord_i")

        c_cd = TESS_Mapping_Struct.cd_matrix_ij.astype(self.ftype).ravel()
        g_cd, g_cd_size = krnl.get_global("cd_ij")

        c_cd_inv = TESS_Mapping_Struct.cd_matrix_inv_ij.astype(self.ftype).ravel()
        g_cd_inv, g_cd_inv_size = krnl.get_global("cd_inv_ij")

        c_Rotation = TESS_Mapping_Struct.R_ij.astype(self.ftype).ravel()
        g_Rotation, g_Rotation_size = krnl.get_global("Rotation_ij") 

        c_fwd_AB = TESS_Mapping_Struct.fwd_distortion_coeffs_kij.astype(self.ftype).ravel()
        g_fwd_AB, g_fwd_AB_size = krnl.get_global("fwd_AB")

        c_inv_AB = TESS_Mapping_Struct.inv_distortion_coeffs_kij.astype(self.ftype).ravel()
        g_inv_AB, g_inv_AB_size = krnl.get_global("inv_AB")

        c_N_pts = np.array([vertex_buffer_ni.shape[0]], np.int32)
        g_N_pts, g_N_pts_size = krnl.get_global("N_pts")

        # Initialize variable memory
        c_distort = np.empty_like(vertex_buffer_ni.ravel(), dtype=self.ftype)
        g_distort = drv.mem_alloc(c_distort.nbytes)

        c_vb = vertex_buffer_ni.astype(self.ftype).ravel()
        g_vb = drv.mem_alloc(c_vb.nbytes)

        c_icrs = np.empty((vertex_buffer_ni.shape[0], 3), dtype=self.ftype).ravel()
        g_icrs = drv.mem_alloc(c_icrs.nbytes)

        self.c_ref_px_coord, self.g_ref_px_coord = c_ref_px_coord, g_ref_px_coord
        self.c_cd, self.g_cd = c_cd, g_cd
        self.c_cd_inv, self.g_cd_inv = c_cd_inv, g_cd_inv
        self.c_Rotation, self.g_Rotation = c_Rotation, g_Rotation
        self.c_fwd_AB, self.g_fwd_AB = c_fwd_AB, g_fwd_AB
        self.c_inv_AB, self.g_inv_AB = c_inv_AB, g_inv_AB
        self.c_N_pts, self.g_N_pts = c_N_pts, g_N_pts

        self.c_distort, self.g_distort = c_distort, g_distort
        self.c_vb, self.g_vb = c_vb, g_vb
        self.c_icrs, self.g_icrs = c_icrs, g_icrs

        self.initialized = True

    def upload_constants(self, TESS_Mapping_Struct, n_pts):
        # This should be a constant buffer upload, but I'm a little uneasy of packing this in numpy.
        # I'll do it manually for now, but if this works we should wrap it in a nice API.

        self.c_ref_px_coord[:] = TESS_Mapping_Struct.ref_px_coord_i.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_ref_px_coord, self.c_ref_px_coord)

        self.c_cd[:] = TESS_Mapping_Struct.cd_matrix_ij.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_cd, self.c_cd)

        self.c_cd_inv[:] = TESS_Mapping_Struct.cd_matrix_inv_ij.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_cd_inv, self.c_cd_inv)

        self.c_Rotation[:] = TESS_Mapping_Struct.R_ij.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_Rotation, self.c_Rotation)

        self.c_fwd_AB[:] = TESS_Mapping_Struct.fwd_distortion_coeffs_kij.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_fwd_AB, self.c_fwd_AB)

        self.c_inv_AB[:] = TESS_Mapping_Struct.inv_distortion_coeffs_kij.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_inv_AB, self.c_inv_AB)

        self.c_N_pts[0] = n_pts
        drv.memcpy_htod(self.g_N_pts, self.c_N_pts)

    def distort_sip(self, TESS_Mapping_Struct, distort, vb):
        self.initialize(TESS_Mapping_Struct, vb)
        self.upload_constants(TESS_Mapping_Struct, vb.shape[0])

        self.c_vb[:] = vb.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_vb, self.c_vb)

        self.cu_distort_sip(
            self.g_distort,
            self.g_vb,
            block=(256,1,1), grid=((vb.shape[0] + 255) // 256, 1)
        )

        drv.memcpy_dtoh(self.c_distort, self.g_distort)
        return self.c_distort.reshape(vb.shape)

    def wcs_map_ICRS_from_TESS_SLOW(self, src, src_pixel_ni):
        R0 = 180.0 / np.pi

        uv_ni = src_pixel_ni - (src.ref_px_coord_i - 1)# / 2 # Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
        distort_ni = np.empty_like(uv_ni)
        self.distort_sip(src, distort_ni, uv_ni)
        uv_ni += distort_ni
        xy_ni = (src.cd_matrix_ij @ uv_ni.T).T
        xyz_ni = np.pad(xy_ni, ((0,0),(0,1)))
        xyz_ni[:,2] = R0

        # Rotate to ICRS. This is where we halt!
        icrs_xyz_ni = (src.R_ij @ xyz_ni.T).T
        return icrs_xyz_ni
    
    def wcs_map_ICRS_from_TESS(self, src, src_pixel_ni):
        self.initialize(src, src_pixel_ni)
        self.upload_constants(src, src_pixel_ni.shape[0])

        self.c_vb[:] = src_pixel_ni.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_vb, self.c_vb)

        self.cu_wcs_map_ICRS_from_TESS(
            self.g_icrs,
            self.g_vb,
            block=(256,1,1), grid=((src_pixel_ni.shape[0] + 255) // 256, 1)
        )

        drv.memcpy_dtoh(self.c_icrs, self.g_icrs)
        return self.c_icrs.reshape((src_pixel_ni.shape[0], 3))
    
    def wcs_map_TESS_from_ICRS(self, dst, icrs_xyz_ni):
        self.initialize(dst, icrs_xyz_ni)
        self.upload_constants(dst, icrs_xyz_ni.shape[0])

        self.c_icrs[:] = icrs_xyz_ni.ravel().astype(self.ftype)
        drv.memcpy_htod(self.g_icrs, self.c_icrs)

        self.cu_wcs_map_TESS_from_ICRS(
            self.g_vb,
            self.g_icrs,
            block=(256,1,1), grid=((icrs_xyz_ni.shape[0] + 255) // 256, 1)
        )

        drv.memcpy_dtoh(self.c_vb, self.g_vb)
        return self.c_vb.reshape((icrs_xyz_ni.shape[0], 2))

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Coordinate Validation
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# Old validation sampler built by Claude Opus 4.6 on 2026 March 14. We'll need to rebuild it for the new API.
def validate_coordinate_transforms(n_samples=4194304): # 2048*2048 #(n_samples=10000):
    """
    Compare our transform against astropy's all_pix2world for random
    pixels.  Reports angular separation statistics in arcseconds.
    """
    rng = np.random.default_rng(42)
    timers = ProfileTimer()

    files = get_file_list()
    data, header = fits.getdata(files[0], header=True)
    
    cuda = CUDAContainer()

    scope = locals()

    # Iterate a few times to saturate the profiler
    maxiters = 11
    for n in range(maxiters):
        with timers["wcs_variables_validate"]:
            keys = wcs_variables_validate(header)
        with timers["wcs_variables_load_nonan"]:
            TESS_Mapping_Variables = wcs_variables_load_nonan(header, keys)
        with timers["wcs_assemble_struct"]:
            TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)

        with timers["sample generation"]:
            ξ_ni = np.random.random((n_samples, 2)).astype(np.float64)
            uv_ni = ξ_ni * (TESS_Mapping_Struct.naxis_i[None, :] - 1.0)

        # Astropy Implementation=
        with timers["astropy pre"]:
            wcs = WCS(header)
            px, py = uv_ni[:,0], uv_ni[:,1]
        with timers["astropy all_pix2world"]:
            ra_ap, dec_ap = wcs.all_pix2world(px, py, 0)
        #with timers["astropy post"]:
        if True:
            # Convert to unit sphere
            c_ra_ap, s_ra_ap = np.cos(np.radians(ra_ap)), np.sin(np.radians(ra_ap))
            c_dec_ap, s_dec_ap = np.cos(np.radians(dec_ap)), np.sin(np.radians(dec_ap))

            x_ap = c_dec_ap * c_ra_ap
            y_ap = c_dec_ap * s_ra_ap
            z_ap = s_dec_ap

            xyz_ap = np.stack([x_ap, y_ap, z_ap], axis=-1)

        # Custom implementation
        with timers["custom ICRS from TESS"]:
            xyz_custom = wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, uv_ni)
        #with timers["custom post"]:
        if True:
            xyz_custom /= np.linalg.norm(xyz_custom, axis=-1, keepdims=True)

        # Custom CUDA implementation
        with timers["custom CUDA ICRS from TESS (Slow)"]:
            xyz_cuda = cuda.wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, uv_ni)

        #with timers["custom CUDA post"]:
        if True:
            xyz_cuda /= np.linalg.norm(xyz_cuda, axis=-1, keepdims=True)
            print(f"Custom CUDA implementation max radius from unit sphere: {np.max(np.abs(xyz_ap - xyz_custom), axis=0)[:5]},")

        with timers["custom CUDA ICRS from TESS"]:
            xyz_cuda = cuda.wcs_map_ICRS_from_TESS(TESS_Mapping_Struct, uv_ni)
        if True:
            xyz_cuda /= np.linalg.norm(xyz_cuda, axis=-1, keepdims=True)
            print(f"Custom CUDA implementation max radius from unit sphere: {np.max(np.abs(xyz_ap - xyz_custom), axis=0)[:5]},")


        with timers["custom CUDA TESS from ICRS"]:
            uv_ni_reversal_cuda = cuda.wcs_map_TESS_from_ICRS(TESS_Mapping_Struct, xyz_cuda)
        if True:
            print(f"Custom CUDA reversal max error: {np.max(np.abs(uv_ni_reversal_cuda - uv_ni), axis=0)[:5]},")
        with timers["custom TESS from ICRS"]:
            uv_ni_reversal = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, xyz_custom)
        with timers["custom TESS from ICRS"]:
            uv_ni_reversal_2 = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, xyz_ap)
        with timers["astropy world2pix"]:
            uv_ni_reversal_api = np.array(wcs.all_world2pix(ra_ap, dec_ap, 0)).T
        scope.update(locals()) # For interactive inspection
        print(f"Iteration {n+1}/{maxiters} complete.")
        if n != 0:
            print(timers)

    if False:
        # Cast TESS_Mapping_Struct to float32 for testing.
        TESS_Mapping_Struct_float = {**TESS_Mapping_Struct._asdict()} # Dict copy
        for key, value in TESS_Mapping_Struct._asdict().items():
            if isinstance(value, np.ndarray) and value.dtype == np.float32:
                TESS_Mapping_Struct_float[key] = value.astype(np.float64)
            elif isinstance(value, float):
                TESS_Mapping_Struct_float[key] = float(value)
            else:
                TESS_Mapping_Struct_float[key] = value
        TESS_Mapping_Struct_float = WCSMappingStruct(**TESS_Mapping_Struct_float) # Convert back to struct
            
        xyz_custom = wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct_float, uv_ni.astype(np.float32))
        xyz_custom /= np.linalg.norm(xyz_custom, axis=-1, keepdims=True)
        xyz_custom = xyz_custom.astype(np.float64) # Cast back to float64 for comparison. This is a bit sad but it allows us to test the float32 implementation against astropy's float64.

    # Angular seperation
    θ_n = np.arccos(np.clip(np.sum(xyz_ap * xyz_custom, axis=-1), -1.0, 1.0))
    sep_arcsec = np.degrees(θ_n) * 3600.0

    result = {
        'n_samples': n_samples,
        'max_error_arcsec': float(np.max(sep_arcsec)),
        'mean_error_arcsec': float(np.mean(sep_arcsec)),
        'median_error_arcsec': float(np.median(sep_arcsec)),
        'max_error_delta': str(np.max(np.abs(xyz_ap - xyz_custom), axis=0)),
        'mean_error_delta': str(np.mean(np.abs(xyz_ap - xyz_custom), axis=0)),
        'median_error_delta': str(np.median(np.abs(xyz_ap - xyz_custom), axis=0)),
    }

    print(f"\nValidation against astropy WCS ({n_samples} random pixels):")
    print(f"  Max error:    {result['max_error_arcsec']:.6e} arcsec")
    print(f"  Mean error:   {result['mean_error_arcsec']:.6e} arcsec")
    print(f"  Median error: {result['median_error_arcsec']:.6e} arcsec")
    print(f"  Max error (delta):    {result['max_error_delta']}")
    print(f"  Mean error (delta):   {result['mean_error_delta']}")
    print(f"  Median error (delta): {result['median_error_delta']}")

    tess_pixel_arcsec = 21.0
    print(f"  (TESS pixel scale: {tess_pixel_arcsec} arcsec/pixel)")
    print(f"  Max error = {result['max_error_arcsec']/tess_pixel_arcsec:.2e} pixels")

    globals().update(locals()) # For interactive inspection

    # with timers["custom TESS from ICRS"]:
    #     uv_ni_reversal = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, xyz_custom)
    # with timers["custom TESS from ICRS"]:
    #     uv_ni_reversal_2 = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, xyz_ap)
    # with timers["astropy world2pix"]:
    #     uv_ni_reversal_api = np.array(wcs.all_world2pix(ra_ap, dec_ap, 0)).T

    uv_ni_reversal_error = uv_ni_reversal - uv_ni
    uv_ni_reversal_2_error = uv_ni_reversal_2 - uv_ni
    uv_ni_reversal_api_error = uv_ni_reversal_api - uv_ni

    print(f"Reversal error ( custom → custom ): max={np.max(np.abs(uv_ni_reversal_error)):.3e} pixels, mean={np.mean(np.abs(uv_ni_reversal_error)):.3e} pixels")
    print(f"Reversal error (astropy → custom ): max={np.max(np.abs(uv_ni_reversal_2_error)):.3e} pixels, mean={np.mean(np.abs(uv_ni_reversal_2_error)):.3e} pixels")
    print(f"Reversal error (astropy → astropy): max={np.max(np.abs(uv_ni_reversal_api_error)):.3e} pixels, mean={np.mean(np.abs(uv_ni_reversal_api_error)):.3e} pixels")

    globals().update(locals())

    print(timers)
    globals().update(locals()) # For interactive inspection

    #return result


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Plot ICRS Reference Frame
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

# Plots ICRS Reference frame and the normal/tangent/binormal vectors
def plot_reference_frame():
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    long = np.linspace(0, 1, 1000)
    short = np.linspace(0, 1, 17)
    for s in short:
        ra = s*np.pi*2
        dec = long*np.pi*2
        x = np.cos(dec)*np.cos(ra)
        y = np.cos(dec)*np.sin(ra)
        z = np.sin(dec)
        ax.plot(x, y, z, color='blue' if s in (0,1) else 'cyan', alpha=0.5)
    for s in short:
        dec = s*np.pi*2
        ra = long*np.pi*2
        x = np.cos(dec)*np.cos(ra)
        y = np.cos(dec)*np.sin(ra)
        z = np.sin(dec)
        ax.plot(x, y, z, color='red' if s in (0,1) else 'magenta', alpha=0.5)

    # Pick the normal direction:
    #nrml = np.array([1, 1, 1], dtype=np.float64)
    ra0, de0 = 0.0, np.pi/2 # 90 degrees
    nrml = np.array([np.cos(de0)*np.cos(ra0), np.cos(de0)*np.sin(ra0), np.sin(de0)], dtype=np.float64)
    nrml /= np.linalg.norm(nrml)
    ax.quiver(*nrml, *nrml, color='black', length=1.0, label='Normal (CRVAL)')

    # From this, pick the tangent direction:
    tan = np.array([-nrml[1], nrml[0], 0], dtype=np.float64)
    tan /= np.linalg.norm(tan)
    ax.quiver(*nrml, *tan, color='red', length=1.0, label='Tangential (RA)')

    Bin = np.cross(nrml, tan)
    Bin /= np.linalg.norm(Bin)
    ax.quiver(*nrml, *Bin, color='green', length=1.0, label='Binormal (Dec)')

    # Plot an RGB gizmo at the origin with lengths of 0.5
    ax.quiver(0, 0, 0, 0.5, 0, 0, color='red', label='X-axis')
    ax.quiver(0, 0, 0, 0, 0.5, 0, color='green', label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 0.5, color='blue', label='Z-axis')

    ax.set_aspect('equal') # Does equal aspect ratio work in 3d?
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Reference Frame Vectors in Cartesian Space')
    plt.show()

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Main Execution
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    #plot_reference_frame()
    print(validate_coordinate_transforms(n_samples=2048*2048))
    pass

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Auxiliary data visualization
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════

if False: # Test results from validate_coordinate_transforms.
    data = [
    [128**2, 0.000000, 0.000521, 0.000000, 0.000000, 0.014567, 0.004535, 0.001002, 0.025624, 0.000000, 0.033502, 0.012582, ],
    [256**2, 0.000000, 0.000532, 0.000000, 0.002508, 0.014021, 0.032077, 0.007516, 0.039700, 0.002993, 0.038033, 0.081709, ],
    [512**2, 0.000000, 0.000998, 0.000000, 0.009032, 0.013586, 0.101996, 0.035118, 0.065252, 0.009498, 0.051590, 0.387970, ],
    [1024**2, 0.000000, 0.000000, 0.000000, 0.040698, 0.017050, 0.517556, 0.118995, 0.149174, 0.039130, 0.208157, 1.391886, ],
    [2048**2, 0.000000, 0.000409, 0.000152, 0.207451, 0.022495, 1.847724, 0.463264, 0.294955, 0.210375, 0.428163, 5.987498, ]
    ]
    data = np.array(data)

    keys = {
        0: "                 samples",
        1: "  wcs_variables_validate",
        2: "wcs_variables_load_nonan",
        3: "     wcs_assemble_struct",
        4: "       sample generation",
        5: "             astropy pre",
        6: "   astropy all_pix2world",
        7: "            astropy post",
        8: "   custom ICRS from TESS",
        9: "             custom post",
    10: "   custom TESS from ICRS",
    11: "       astropy world2pix",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    #for i, key in enumerate(keys[1+5:]):
    for i in [6,11,8,10]:
        key = keys[i]
        ax.plot(data[:, 0], data[:, i], marker='o', label=key)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Timing of WCS Mapping Components')
    ax.legend()
    plt.grid(True, which="both", ls="--")
    plt.show()


    # Copy result to keyboard via command due to remote access issues:
    import keyboard

    # get fig canvas rgba data as bytes
    img = np.array(fig.canvas.buffer_rgba())
    # Convert to PIL Image and save to clipboard
    if True:
        import pyperclip
        from PIL import Image
        import io
        pil_img = Image.fromarray(img)
        output = io.BytesIO()
        pil_img.save(output, format='PNG')
        data = output.getvalue()
        pyperclip.copy(data)
