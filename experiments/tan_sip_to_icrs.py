"""
tan_sip_to_icrs.py — Trig-free TAN-SIP pixel → unit ICRS vec3

Maps every pixel in a TESS FFI to a unit direction vector on the ICRS
celestial sphere, incorporating the full SIP distortion polynomial.

    ┌─────────────────────────────────────────────────────────┐
    │  Per-pixel cost:  ~55 FLOPs + 1 rsqrt                  │
    │  Transcendentals: ZERO                                  │
    │  Branches:        ZERO                                  │
    │                                                         │
    │  WCSLIB equivalent: ~30 FLOPs + 8 trig + 3-4 branches  │
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
import time

from collections import namedtuple
import numba

# TAN projection constant: the radius of curvature of the unit sphere
# in degrees.  This is the "focal length" of the gnomonic projection.
R0 = 180.0 / np.pi   # 57.29577951...°


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════

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
            params.append([f'{key}', f'{val:.6f}', f'{std:.6f}', f'{count}', f'{val*count:.6f}', f'{humanize.metric(1/val, 'Hz')}'])

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


# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════


# scratch:
# c⁰¹²³⁴⁵₀₁₂₃₄₅
# r²(A₂₀c²s⁰ + A₁₁c¹s¹ + A₀₂c⁰s²) + 
# r³(A₃₀c³s⁰ + A₂₁c²s¹ + A₁₂c¹s² + A₀₃c⁰s³) + 
# r⁴(A₄₀c⁴s⁰ + A₃₁c³s¹ + A₂₂c²s² + A₁₃c¹s³ + A₀₄c⁰s⁴)

# ```
# vec2 fragCoord : input; (bad syntax but you get the idea)
# vec2 pix = fragCoord - (CRPIX - 1.0);
# 
# // distortion
# vec5 up = vec5(1.0f, pix.x, pix.x**2, pix.x**3, pix.x**4); // Also bad syntax but you get the idea. This would be a function anyway.
# vec5 vp = vec5(1.0f, pix.y, pix.y**2, pix.y**3, pix.y**4);
# pix += einsum('ijk,i,j->k', AB, up, vp);
# 
# // pixel -> uv coordinates
# vec2 uv = CD * pix;
# ```
# Lets visualize the lens distortions!
def plot_distortions():
    # Assuming T is in scope:
    # >>> T.keys()
    # dict_keys(['crpix1', 'crpix2', 'cd', 'a_order', 'b_order', 'a_coeffs', 'b_coeffs', 'R', 'naxis1', 'naxis2'])

    # Build prep objects
    naxis = np.array([T['naxis1'], T['naxis2']]) # (nx, ny) indexing
    crpix = np.array([T['crpix1'], T['crpix2']]) # (x, y) indexing

    # AB coeffs are irritating, as they're indexed as pairs like so:
    # >>> a_coeffs
    # {(0, 2): <f64>, (0, 3): <f64>, ...}
    # This also uses the inaccurate pixel coordinate powers from normalized coordinates, but this is just for visualization so it doesn't matter.
    # Copilot filled in: "...instead of the more stable Horner's method, ..."
    # I wonder how Horners method would apply here? its an interesting suggestion. A later task!
    # Lets build the full 5x5x2 matrix of coefficients, filling in zeros where needed.
    AB = np.zeros((2, 5, 5), dtype=np.float64) # [A/B, i, j] 
    for (i, j), c in T['a_coeffs'].items():
        AB[0, i, j] = c
    for (i, j), c in T['b_coeffs'].items():
        AB[1, i, j] = c

    # >>> T['cd']
    # array([[-0.00541008,  0.00171852],
    #        [-0.00185093, -0.00544815]])
    # It looks like cd was precomputed. But... what's the order?
    # cd = np.array([[header['CD1_1'], header['CD1_2']],
    #                [header['CD2_1'], header['CD2_2']]])
    # Claude wrote:
    # du = _eval_sip(u, v, T['a_coeffs'], T['a_order'])
    # dv = _eval_sip(u, v, T['b_coeffs'], T['b_order'])
    # u_sip = u + du
    # v_sip = v + dv
    # 
    # # ── Step 3: CD matrix → intermediate world coordinates (degrees) ─
    # x = T['cd'][0, 0] * u_sip + T['cd'][0, 1] * v_sip
    # y = T['cd'][1, 0] * u_sip + T['cd'][1, 1] * v_sip
    # So in matrix algebra, given sip = [u_sip, v_sip] and xy = [x,y], then:
    # xy = cd @ sip? Or, is it xy = ct.T @ sip?
    # copilot: "Given the code snippet, it appears that the CD matrix is being applied to the SIP-corrected pixel coordinates (u_sip, v_sip) to produce the intermediate world coordinates (x, y). The operation is a matrix multiplication where the CD matrix is multiplied by the SIP-corrected coordinates. In this case, the correct operation would be xy = cd @ sip, where cd is the 2x2 CD matrix and sip is the 2-element vector [u_sip, v_sip]."

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    # Lets plot 32 lines for each axis. We'll need "short" and "long" interpolators.
    # Unsure if long should use the +1 or offset endpoints. We'll see if it makes a difference.
    shrt = np.linspace(0, 1, 32+1, dtype=np.float64) # Short lines for the SIP polynomial
    long = np.linspace(0, 1, 2048+1, dtype=np.float64) # Long lines for the full transform
    l0 = np.zeros_like(long)

    for s in shrt:
        ij = np.array([l0+s, long]).T*naxis[None,:] # set one axis to s, and the other to the full range of pixels. (N, 2) indexing
        ij = ij - (crpix[None,:] - 1)
        # Compute ij powers for the SIP polynomial. This is the "up" and "vp" vectors in the shader code.
        up = np.stack([ij[:,0]**i for i in range(5)], axis=0) # (5, N) indexing
        vp = np.stack([ij[:,1]**i for i in range(5)], axis=0) # (5, N) indexing
        distort = np.einsum('kij,iN,jN->Nk', AB, up, vp) # (N, 2) indexing.
        # distort is causing problems: ValueError: operands could not be broadcast together with remapped shapes [original->remapped]: (2,5,5)->(2,5,5) (5,2049)->(2049,newaxis,5,newaxis) (4,2049)->(2049,newaxis,newaxis,4) 
        # 

        # Visualize!
        dstrt = np.linalg.norm(distort, axis=-1)
        ax.plot(ij[:,0], ij[:,1], dstrt, color='blue' if s in (0,1) else 'cyan', alpha=0.5)

    # Repeat along other axis
    for s in shrt:
        ij = np.array([long, l0+s]).T*naxis[None,:] # set one axis to s, and the other to the full range of pixels. (N, 2) indexing
        ij = ij - (crpix[None,:] - 1)
        # Compute ij powers for the SIP polynomial. This is the "up" and "vp" vectors in the shader code.
        up = np.stack([ij[:,0]**i for i in range(5)], axis=0) # (5, N) indexing
        vp = np.stack([ij[:,1]**i for i in range(5)], axis=0) # (5, N) indexing
        distort = np.einsum('kij,iN,jN->Nk', AB, up, vp) # (N, 2) indexing.

        # Visualize!
        dstrt = np.linalg.norm(distort, axis=-1)
        ax.plot(ij[:,0], ij[:,1], dstrt, color='red' if s in (0,1) else 'magenta', alpha=0.5)
    
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_zlabel('SIP Distortion Magnitude (pixels)')
    ax.set_title('SIP Distortion Magnitude Across the Image')
    plt.show()
    """
    
    # Do a 2d plot of the pure distortions!
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Lets plot 32 lines for each axis. We'll need "short" and "long" interpolators.
    # Unsure if long should use the +1 or offset endpoints. We'll see if it makes a difference.
    shrt = np.linspace(0, 1, 32+1, dtype=np.float64) # Short lines for the SIP polynomial
    long = np.linspace(0, 1, 2048+1, dtype=np.float64) # Long lines for the full transform
    l0 = np.zeros_like(long)

    for s in shrt:
        ij = np.array([l0+s, long]).T*naxis[None,:] # set one axis to s, and the other to the full range of pixels. (N, 2) indexing
        ij = ij - (crpix[None,:] - 1)
        # Compute ij powers for the SIP polynomial. This is the "up" and "vp" vectors in the shader code.
        up = np.stack([ij[:,0]**i for i in range(5)], axis=0) # (5, N) indexing
        vp = np.stack([ij[:,1]**i for i in range(5)], axis=0) # (5, N) indexing
        distort = np.einsum('kij,iN,jN->Nk', AB, up, vp) # (N, 2) indexing.
        ij = ij + distort # Apply the SIP distortion to the pixel coordinates

        # cd transform
        xy = (T['cd'] @ ij.T).T # (2, N) indexing

        # Visualize!
        #ax.plot(*ij.T, color='blue' if s in (0,1) else 'cyan', alpha=0.5)
        ax.plot(*xy.T, color='blue' if s in (0,1) else 'cyan', alpha=0.5)
    
    # Repeat along other axis
    for s in shrt:
        ij = np.array([long, l0+s]).T*naxis[None,:] # set one axis to s, and the other to the full range of pixels. (N, 2) indexing
        ij = ij - (crpix[None,:] - 1)
        # Compute ij powers for the SIP polynomial. This is the "up" and "vp" vectors in the shader code.
        up = np.stack([ij[:,0]**i for i in range(5)], axis=0) # (5, N) indexing
        vp = np.stack([ij[:,1]**i for i in range(5)], axis=0) # (5, N) indexing
        distort = np.einsum('kij,iN,jN->Nk', AB, up, vp) # (N, 2) indexing.
        ij = ij + distort # Apply the SIP distortion to the pixel coordinates

        # cd transform
        xy = (T['cd'] @ ij.T).T # (2, N) indexing

        # Visualize!
        #ax.plot(*ij.T, color='red' if s in (0,1) else 'magenta', alpha=0.5)
        ax.plot(*xy.T, color='red' if s in (0,1) else 'magenta', alpha=0.5)
    
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    ax.set_aspect(1)
    ax.set_title('SIP Distortion Magnitude Across the Image')
    plt.show()
    
    
# ═══════════════════════════════════════════════════════════════════════
#  Setup — runs ONCE per WCS header (precompute all constants)
# ═══════════════════════════════════════════════════════════════════════

def _parse_sip(header, prefix, order):
    """Extract SIP polynomial coefficients {(i,j): value} from header.
    SIP convention: terms start at order 2 (linear terms are in CD + CRPIX)."""
    coeffs = {}
    for i in range(order + 1):
        for j in range(order + 1 - i):
            if i + j < 2:
                continue
            key = f'{prefix}_{i}_{j}'
            if key in header:
                coeffs[(i, j)] = header[key]
    return coeffs


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


# ═══════════════════════════════════════════════════════════════════════

#njargs = dict(parallel=True, fastmath=True, nogil=True)
njargs = dict(parallel=True, nogil=True) # Fastmath seems to be slower?
#njargs = dict(parallel=True) # nogil is as fast as numpy native. Drat! Unless the bulk of the code is the distort.

#@numba.njit#(parallel=True) # Lets work on parallel later. This is obviously embarrassingly parallel.
#@numba.njit(parallel=True)
@numba.njit(**njargs)
def distort_sip(distort, vb, AB):
    # I hope numba supports stack!
    # up = np.stack([vb[:,0]**i for i in range(5)], axis=0) # (5, N) indexing
    # vp = np.stack([vb[:,1]**i for i in range(5)], axis=0) # (5, N) indexing
    #distort = np.einsum('kij,iN,jN->Nk', AB, up, vp) # (N, 2) indexing.
    # I KNOW numba doesn't support einsum. Let's write it out manually. Actually, lets write out the whole thing manually. I guess.

    #distort = np.zeros_like(vb)
    # up = np.empty((5,), dtype=np.float64)
    # vp = np.empty((5,), dtype=np.float64)
    # up[0] = 1.0
    # vp[0] = 1.0
    #for n in range(vb.shape[0]):
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
        # for j in range(5):
        #     for i in range(5-j):
        #         distort[n,0] += AB[0,i,j] * up[i] * vp[j]
        #         distort[n,1] += AB[1,i,j] * up[i] * vp[j]
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


# Step 4-6: TAN deproject → rotate → normalize
@numba.njit#(allow_reuse=True) # This is the bottleneck, so we want to reuse the compiled version for all pixels.
def tan_deproject_rotate_normalize(vbxyz, vb, R):
    R0 = 180.0 / np.pi
    nx = -vb[:, 1]   # -y
    ny =  vb[:, 0]   #  x
    wx = R[0,0]*nx + R[0,1]*ny + R[0,2]*R0
    wy = R[1,0]*nx + R[1,1]*ny + R[1,2]*R0
    wz = R[2,0]*nx + R[2,1]*ny + R[2,2]*R0
    inv_norm = 1.0 / np.sqrt(wx*wx + wy*wy + wz*wz)
    # vb[:, 0] = wx * inv_norm
    # vb[:, 1] = wy * inv_norm
    # vz = wz * inv_norm  (if you want 3D later)
    # res = np.empty_like(vb)
    # res[:, 0] = wx * inv_norm
    # res[:, 1] = wy * inv_norm
    # 
    # return res
    vbxyz[:, 0] = wx * inv_norm
    vbxyz[:, 1] = wy * inv_norm
    vbxyz[:, 2] = wz * inv_norm

@numba.njit
def make_boundary_vb(nx, ny, step=32):
    """Build a closed boundary loop in raw 0-based pixel coords."""
    l1 = np.arange(0, nx + step, step)
    l1[-1] = nx - 1  # clamp to last valid pixel
    l0 = np.zeros_like(l1)
    xhat = np.array([l1, l0])          # (2, N)
    yhat = np.array([l0, l1.copy()])   # need separate l1 for y
    # Recompute for y-axis using ny
    l1y = np.arange(0, ny + step, step)
    l1y[-1] = ny - 1
    l0y = np.zeros_like(l1y)
    yhat = np.array([l0y, l1y])        # (2, M)
    # Build 4 edges of the boundary
    line0 =  xhat                                                     # bottom: (0,0)→(nx-1,0)
    line1 =  yhat[:,1:] + np.array([[nx-1],[0]])                      # right:  (nx-1,step)→(nx-1,ny-1)
    line2 = -xhat[:,1:] + np.array([[nx-1],[ny-1]])                   # top:    (nx-1-step,ny-1)→(0,ny-1)
    line3 = -yhat[:,1:] + np.array([[0],[ny-1]])                      # left:   (0,ny-1-step)→(0,0)
    return np.hstack([line0, line1, line2, line3]).T.astype(np.float64)  # (N, 2) raw pixel coords

# ═══════════════════════════════════════════════════════════════════════

def wcs_map_ICRS_from_TESS_SLOW(src, src_pixel_ni):
    """
    Convert a vertex buffer to dst from src's coordinate systems, with end-to-end distortion corrections. This is NOT high performance compute.
    """

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

    xyz_ni = (dst.R_ij.T @ icrs_xyz_ni.T).T
    xy_ni = xyz_ni[:,:2] * R0 / xyz_ni[:,2:]
    uv_ni = (dst.cd_matrix_inv_ij @ xy_ni.T).T
    distort_ni = np.empty_like(uv_ni)
    #distort_sip(distort_ni, uv_ni, dst.inv_distortion_coeffs_kij)
    #uv_ni += distort_ni # Plus, or minus here?
    distort_sip_inv_newton(distort_ni, uv_ni, dst.fwd_distortion_coeffs_kij, dst.inv_distortion_coeffs_kij)
    # distort_ni is now uv_ni for this algorithm!
    dst_pixel_ni = distort_ni + (dst.ref_px_coord_i - 1)
    return dst_pixel_ni


# Variant of above with constant memory access
@numba.njit(**njargs)
def _wcs_map_ICRS_from_TESS(icrs_xyz_ni, distort_ni, 
    src__naxis_i, src__ref_px_coord_i, src__fwd_distortion_coeffs_kij, src__cd_matrix_ij, src__R_ij, # src, 
    src_pixel_ni):

    #src_pixel_ni[:] -= (src__ref_px_coord_i - 1).astype(src_pixel_ni.dtype) # / 2 # Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    for n in numba.prange(src_pixel_ni.shape[0]):
        for i in numba.prange(src_pixel_ni.shape[1]):
            #src_pixel_ni[n,i] -= src__ref_px_coord_i[i] - 1.0
            src_pixel_ni[n,i] = src_pixel_ni[n,i]*src__naxis_i[i] - (src__ref_px_coord_i[i] - 1.0)

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
    # src__naxis_i = src.naxis_i
    # src__ref_px_coord_i = src.ref_px_coord_i
    # src__fwd_distortion_coeffs_kij = src.fwd_distortion_coeffs_kij
    # src__cd_matrix_ij = src.cd_matrix_ij
    # src__R_ij = src.R_ij
    # _wcs_map_ICRS_from_TESS(icrs_xyz_ni, distort_ni,
    #     src__naxis_i, src__ref_px_coord_i, src__fwd_distortion_coeffs_kij, src__cd_matrix_ij, src__R_ij,
    #     src_pixel_ni)

    # Debug print the shapes and dtypes of each of these objects for allocating in another function
    # print(f"""
    #     {icrs_xyz_ni.shape = }, {icrs_xyz_ni.dtype = }
    #     {distort_ni.shape = }, {distort_ni.dtype = }
    #     {src.naxis_i.shape = }, {src.naxis_i.dtype = }
    #     {src.ref_px_coord_i.shape = }, {src.ref_px_coord_i.dtype = }
    #     {src.fwd_distortion_coeffs_kij.shape = }, {src.fwd_distortion_coeffs_kij.dtype = }
    #     {src.cd_matrix_ij.shape = }, {src.cd_matrix_ij.dtype = }
    #     {src.R_ij.shape = }, {src.R_ij.dtype = }
    #     {src_pixel_ni.shape = }, {src_pixel_ni.dtype = }
    #       """)
    # globals().update(locals())
    # raise Exception


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
    # dst__naxis_i = dst.naxis_i
    # dst__ref_px_coord_i = dst.ref_px_coord_i
    # dst__fwd_distortion_coeffs_kij = dst.fwd_distortion_coeffs_kij
    # dst__inv_distortion_coeffs_kij = dst.inv_distortion_coeffs_kij
    # dst__cd_matrix_inv_ij = dst.cd_matrix_inv_ij
    # dst__R_ij = dst.R_ij
    # _wcs_map_TESS_from_ICRS(dst_pixel_ni, distort_ni,
    #     dst__naxis_i, dst__ref_px_coord_i, dst__fwd_distortion_coeffs_kij, dst__inv_distortion_coeffs_kij, dst__cd_matrix_inv_ij, dst__R_ij,
    #     icrs_xyz_ni)
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
# wcs_map_ICRS_from_TESS(TESS_Mapping_Struct, np.array([[1000.0, 1000.0], [1500.0, 1500.0]]))

# ═══════════════════════════════════════════════════════════════════════
#  Cuda kernels for the above transformations. This is where the real performance wins will be.
# ═══════════════════════════════════════════════════════════════════════

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
void vec2_distort_sip_cuda_f64(double* distort, double* uv, double* AB) {
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

// __global__
// void distort_sip_cuda_f64(double* distort, double* vb, double* AB)
//     const int width = 2048;
//     const int height = 2048;
//     DEF_IDX(width, height) // int idx is defined here! // DEF_IDX(width, height)
//     // This index shouldn't matter if we treat this as a single vertex buffer, maybe.
//     int u = idx / w; // *idx = y * width + x;
//     int v = idx % w;
// 
//     if (u >= h - kh || v >= w - kw) return;
//     
//     vec2_distort_sip_cuda_f64(&distort[idx*2], &vb[idx*2], AB);
// }

__global__
void distort_sip_cuda_f64(double* distort, double* vb, double* AB, int N) {
    //int idx = threadIdx.x + blockDim.x * blockIdx.x;
    //if (idx >= N) return;
    DEF_IDX(N, 1) // int idx is defined here! // DEF_IDX(width, height)
    
    vec2_distort_sip_cuda_f64(&distort[idx*2], &vb[idx*2], AB); 
    //distort[idx*2 + 0] = 3.141592;
    //distort[idx*2 + 1] = 2.718281;
}

// This looks good! Thanks copilot! :)
// Now lets do the harder one: the inverse distortion. Lets fix the iterations at 3. 

__global__
void distort_sip_inv_newton_cuda_f64(double* result, double* out_ni, double* AB, double* iAB, int N) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= N) return;

    // # ── Warm start: AP/BP inverse polynomial ──    
    double out_i[2] = {out_ni[idx*2 + 0], out_ni[idx*2 + 1]};
    double distort[2];
    vec2_distort_sip_cuda_f64(distort, out_i, iAB);

    double vb[2] = {
        out_i[0] + distort[0],
        out_i[1] + distort[1]
    }; // This is the initial guess for vb, which is out + inverse_SIP(out). We compute inverse_SIP(out) using the iAB coefficients, which are the coefficients of the inverse polynomial fit.

    for(int iter=0; iter<3; iter++) // Empirically, this has a max error of 3e-12. Machine precision! I'll take it.
    {
        // Build power arrays at current guess
        double up[5+2]; // Pad by 2 to avoid out-of-bounds when computing derivatives. We'll just ignore the last two elements.
        double vp[5+2];
        up[0] = 0.0; up[5+1] = 0.0;  // pad: ensures 0*up[0]=0 even if it were NaN
        vp[0] = 0.0; vp[5+1] = 0.0;
        up[1+0] = 1.0;
        vp[1+0] = 1.0;
        for(int i=1; i<5; i++)
        {
            up[1+i] = vb[0] * up[1+i - 1]; // I'm using an explicit 1+ to keep the pad in mind. The 1+i-1 should constexpr away.
            vp[1+i] = vb[1] * vp[1+i - 1];
        }

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

    //result[n, 0] = gu
    //result[n, 1] = gv
    result[idx*2 + 0] = vb[0];
    result[idx*2 + 1] = vb[1];
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

class CudaDistortionMapper:
    def __init__(self):
        self.initialized = False # Initialize on the first call 

def wcs_map_ICRS_from_TESS_cuda(cudaContainer, src, src_pixel_ni):
    """
    Convert a vertex buffer to dst from src's coordinate systems, with end-to-end distortion corrections. This is NOT high performance compute.
    """

    uv_ni = src_pixel_ni - (src.ref_px_coord_i - 1)# / 2 # Center at the middle of the pixel grid. This is in [0, N-1] coordinates.
    distort_ni = np.empty_like(uv_ni)
    #distort_sip(distort_ni, uv_ni, src.fwd_distortion_coeffs_kij)

    cu_distort_sip_cuda_f64 = krnl.get_function("distort_sip_cuda_f64")
    # Mallocing here is bad form but eh. No. Lets do it right.

    if not cudaContainer.initialized:
        # Initialize distort_ni, uv_ni, and fwd_distortion_coeffs_kij

        cudaContainer.c_distort_ni = np.empty_like(distort_ni.ravel(), dtype=np.float64) # vbuffer
        cudaContainer.g_distort_ni = drv.mem_alloc(cudaContainer.c_distort_ni.nbytes)

        cudaContainer.c_uv_ni = np.empty_like(uv_ni.ravel(), dtype=np.float64) # vbuffer
        cudaContainer.g_uv_ni = drv.mem_alloc(cudaContainer.c_uv_ni.nbytes)

        cudaContainer.c_fwd_distortion_coeffs_kij = np.empty_like(src.fwd_distortion_coeffs_kij.ravel(), dtype=np.float64)
        cudaContainer.g_fwd_distortion_coeffs_kij = drv.mem_alloc(cudaContainer.c_fwd_distortion_coeffs_kij.nbytes)

        cudaContainer.initialized = True

    # Copy data to GPU
    cudaContainer.c_uv_ni[:] = uv_ni.ravel()
    cudaContainer.c_fwd_distortion_coeffs_kij[:] = src.fwd_distortion_coeffs_kij.ravel()
    drv.memcpy_htod(cudaContainer.g_uv_ni, cudaContainer.c_uv_ni)
    drv.memcpy_htod(cudaContainer.g_fwd_distortion_coeffs_kij, cudaContainer.c_fwd_distortion_coeffs_kij)

    # Call kernel
    N = len(uv_ni.ravel())//2
    cu_distort_sip_cuda_f64(
        cudaContainer.g_distort_ni, cudaContainer.g_uv_ni, cudaContainer.g_fwd_distortion_coeffs_kij, np.int32(N),
        block=(256,1,1),
        grid=((N+255)//256,1)
    )
    # Is this block and grid right? I want one thread every *two* datapoints.

    # Copy result back to CPU
    drv.memcpy_dtoh(cudaContainer.c_distort_ni, cudaContainer.g_distort_ni)
    distort_ni[:] = cudaContainer.c_distort_ni.reshape(distort_ni.shape)
    #print(f"{distort_ni[:5] = }")

    # Validate against known good path
    distort_ni_ref = np.empty_like(distort_ni)
    distort_sip(distort_ni_ref, uv_ni, src.fwd_distortion_coeffs_kij)
    #print(f"{distort_ni_ref[:5] = }")

    print(f"Max absolute error in distortion: {np.max(np.abs(distort_ni - distort_ni_ref))}")

    #globals().update(locals())
    #raise Exception("Halt after distortion for testing")

    uv_ni += distort_ni_ref#distort_ni
    xy_ni = (src.cd_matrix_ij @ uv_ni.T).T
    xyz_ni = np.pad(xy_ni, ((0,0),(0,1)))
    xyz_ni[:,2] = R0

    # Rotate to ICRS. This is where we halt!
    icrs_xyz_ni = (src.R_ij @ xyz_ni.T).T
    return icrs_xyz_ni

# ═══════════════════════════════════════════════════════════════════════
#  Boundary loop transform (for testing and visualization)
# ═══════════════════════════════════════════════════════════════════════

# Im going insane. I need a visual for this.
#c_ra, s_ra = np.cos(ref_radec_coord_i[0] / ref_R0), np.sin(ref_radec_coord_i[0] / ref_R0)
#c_de, s_de = np.cos(ref_radec_coord_i[1] / ref_R0), np.sin(ref_radec_coord_i[1] / ref_R0)
#ref_normal_i = np.array([
#    c_de * c_ra, # x
#    c_de * s_ra, # y
#    s_de         # z
#], dtype=FLOAT_T) 
# Lets write a function to visualize this reference frame. It should be a 3d plot of the normal, tan, and bin vectors in Cartesian space. This will help me understand the geometry of the problem better.
def plot_reference_frame():
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # 
    # # Extract the reference frame vectors from T
    # ref_tan = T['R'][:, 0]  # Tangential direction (RA)
    # ref_bin = T['R'][:, 1]  # Binormal direction (Dec)
    # ref_normal = T['R'][:, 2]  # Normal direction (pointing to CRVAL)
    # 
    # # Create a 3D plot
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # 
    # # Plot the reference frame vectors
    # origin = np.array([0, 0, 0])
    # ax.quiver(*origin, *ref_tan, color='r', length=1.0, label='Tangential (RA)')
    # ax.quiver(*origin, *ref_bin, color='g', length=1.0, label='Binormal (Dec)')
    # ax.quiver(*origin, *ref_normal, color='b', length=1.0, label='Normal (CRVAL)')
    # 
    # # Set labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Reference Frame Vectors in Cartesian Space')
    # ax.legend()
    # plt.show()
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


order_vals = [set(), set()]

def build_wcs_transform(header):
    """
    Precompute everything needed for the per-pixel transform.

    Returns a dict — the "uniform buffer" that gets uploaded once
    to the GPU and read by every thread.
    """
    crpix1 = header['CRPIX1']
    crpix2 = header['CRPIX2']
    cd = np.array([[header['CD1_1'], header['CD1_2']],
                   [header['CD2_1'], header['CD2_2']]])

    a_order = header.get('A_ORDER', 0)
    b_order = header.get('B_ORDER', 0)
    a_coeffs = _parse_sip(header, 'A', a_order)
    b_coeffs = _parse_sip(header, 'B', b_order)

    global order_vals
    order_vals[0].add(a_order)
    order_vals[1].add(b_order)

    # print(f"{a_coeffs = }") # DEBUG
    # print(f"{b_coeffs = }") # DEBUG
    # globals().update(locals()) # DEBUG

    # ── Rotation matrix: native tangent-plane frame → ICRS Cartesian ──
    #
    # For a standard TAN projection (θ₀ = 90°, φₚ = 180°), the native
    # north pole IS the reference point (CRVAL1, CRVAL2).
    #
    # The rotation R maps:
    #   native x-axis (φ=0°, θ=0°)  → "south" from CRVAL on the sky
    #   native y-axis (φ=90°, θ=0°) → "east" from CRVAL on the sky
    #   native z-axis (north pole)   → the CRVAL direction itself
    #
    # Derived by computing where each native basis vector maps in sphx2s,
    # then converting to ICRS Cartesian:
    #
    #   R = Rz(RA₀) · Ry(90° - Dec₀) · Rz(-180°)
    #
    # which multiplies out to:
    ra0  = np.radians(header['CRVAL1'])
    dec0 = np.radians(header['CRVAL2'])
    cr, sr   = np.cos(ra0),  np.sin(ra0)
    cd0, sd0 = np.cos(dec0), np.sin(dec0)

    R = np.array([
        [ sd0*cr,  -sr,  cd0*cr],
        [ sd0*sr,   cr,  cd0*sr],
        [-cd0,     0.0,  sd0   ],
    ])

    return {
        'crpix1': crpix1, 'crpix2': crpix2,
        'cd': cd,
        'a_order': a_order, 'b_order': b_order,
        'a_coeffs': a_coeffs, 'b_coeffs': b_coeffs,
        'R': R,
        'naxis1': header.get('NAXIS1', 2048),
        'naxis2': header.get('NAXIS2', 2048),
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-pixel transform (the "vertex shader")
# ═══════════════════════════════════════════════════════════════════════

def _eval_sip(u, v, coeffs, order):
    """Evaluate SIP polynomial: result = Σ C_{ij} · u^i · v^j
    Vectorized over the full pixel grid."""
    if not coeffs:
        return np.zeros_like(u)
    # Build power lookup tables: u^0, u^1, ..., u^order (same for v)
    upow = [np.ones_like(u)]
    vpow = [np.ones_like(v)]
    for k in range(1, order + 1):
        upow.append(upow[-1] * u)
        vpow.append(vpow[-1] * v)
    result = np.zeros_like(u)
    for (i, j), c in coeffs.items():
        result += c * upow[i] * vpow[j]
    return result


def pixel_to_unit_icrs(px, py, T):
    """
    Map 0-based pixel coordinates to unit ICRS direction vectors.

    Parameters
    ----------
    px, py : array_like
        0-based pixel coordinates (matching numpy indexing).
        px = column (FITS x-axis), py = row (FITS y-axis).
    T : dict
        Precomputed transform from build_wcs_transform().

    Returns
    -------
    vx, vy, vz : ndarray
        Unit vectors in ICRS Cartesian coordinates.
        vx = cos(Dec)·cos(RA), vy = cos(Dec)·sin(RA), vz = sin(Dec).
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)

    # ── Step 1: pixel offset from CRPIX ───────────────────────────
    # Convert 0-based numpy coords → FITS 1-based → subtract CRPIX
    u = px - (T['crpix1'] - 1.0)
    v = py - (T['crpix2'] - 1.0)

    # ── Step 2: SIP distortion polynomial ─────────────────────────
    # u' = u + A(u,v),  v' = v + B(u,v)
    # IMPORTANT: both polynomials are evaluated at the ORIGINAL (u, v),
    # not the corrected values.  (Same as WCSLIB's linp2x path C.)
    du = _eval_sip(u, v, T['a_coeffs'], T['a_order'])
    dv = _eval_sip(u, v, T['b_coeffs'], T['b_order'])
    u_sip = u + du
    v_sip = v + dv

    # ── Step 3: CD matrix → intermediate world coordinates (degrees) ─
    x = T['cd'][0, 0] * u_sip + T['cd'][0, 1] * v_sip
    y = T['cd'][1, 0] * u_sip + T['cd'][1, 1] * v_sip

    # ── Step 4: TAN deprojection → native Cartesian ──────────────
    #
    # THIS IS THE KEY INSIGHT.  No trig needed!
    #
    # The gnomonic (TAN) projection maps a 3D direction to a point
    # on the tangent plane by perspective division:
    #     x_proj = R0 · (native_y / native_z)     [note: native_y, not x!]
    #     y_proj = -R0 · (native_x / native_z)    [sign flip from FITS convention]
    #
    # Inverting: given (x, y) on the tangent plane, the native
    # direction vector (before normalization) is simply:
    #     native = (-y, x, R0)
    #
    # The normalization (dividing by |native|) happens in step 6.
    # We DON'T need atan2 to get (φ, θ) and then sin/cos to get back
    # to Cartesian — we skip the spherical round-trip entirely.
    #
    nx = -y
    ny =  x
    # nz = R0 for every pixel (constant)

    # ── Step 5: Rotation matrix × native direction ───────────────
    # R is precomputed from CRVAL in build_wcs_transform().
    # Since R is orthogonal, ||R·n|| = ||n||, so we can normalize after.
    R = T['R']
    wx = R[0, 0]*nx + R[0, 1]*ny + R[0, 2]*R0
    wy = R[1, 0]*nx + R[1, 1]*ny + R[1, 2]*R0
    wz = R[2, 0]*nx + R[2, 1]*ny + R[2, 2]*R0

    # ── Step 6: Normalize to unit sphere ─────────────────────────
    # inv_norm = rsqrt(wx² + wy² + wz²)
    # On GPU this is a single hardware instruction (__drsqrt_rn).
    inv_norm = 1.0 / np.sqrt(wx*wx + wy*wy + wz*wz)

    return wx * inv_norm, wy * inv_norm, wz * inv_norm


# ═══════════════════════════════════════════════════════════════════════
#  Convenience functions
# ═══════════════════════════════════════════════════════════════════════

def make_icrs_map(T):
    """Generate the full (NAXIS2, NAXIS1, 3) unit ICRS direction map."""
    ny, nx = T['naxis2'], T['naxis1']
    py, px = np.mgrid[0:ny, 0:nx].astype(np.float64)
    vx, vy, vz = pixel_to_unit_icrs(px, py, T)
    return np.stack([vx, vy, vz], axis=-1)


def unit_icrs_to_radec(vx, vy, vz):
    """Convert unit ICRS vec3 → (RA, Dec) in degrees.
    These are the only trig calls in the entire pipeline — and they're
    only needed if you want human-readable angles for verification.
    The vec3 IS the fully specified direction; RA/Dec is redundant."""
    ra  = np.degrees(np.arctan2(vy, vx)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(vz, -1.0, 1.0)))
    return ra, dec


def print_corners(T):
    """Print the four corners and center of the image in RA/Dec."""
    nx, ny = T['naxis1'], T['naxis2']
    labels  = ['Bottom-left', 'Bottom-right', 'Top-left', 'Top-right', 'Center']
    px_vals = [0,    nx-1,  0,    nx-1,  (nx-1)/2.0]
    py_vals = [0,    0,     ny-1, ny-1,  (ny-1)/2.0]

    print(f"\n{'Label':>15s}   {'px':>7s}  {'py':>7s}   {'RA (°)':>12s} {'Dec (°)':>12s}   {'vec3':>40s}")
    print("─" * 105)
    for label, px, py in zip(labels, px_vals, py_vals):
        vx, vy, vz = pixel_to_unit_icrs(px, py, T)
        ra, dec = unit_icrs_to_radec(vx, vy, vz)
        print(f"  {label:>13s}   {px:7.1f}  {py:7.1f}   {float(ra):12.6f} {float(dec):12.6f}"
              f"   ({float(vx):+.8f}, {float(vy):+.8f}, {float(vz):+.8f})")
    print()


# ═══════════════════════════════════════════════════════════════════════
#  Validation against astropy WCS
# ═══════════════════════════════════════════════════════════════════════

def validate(header, n_samples=10000):
    """
    Compare our transform against astropy's all_pix2world for random
    pixels.  Reports angular separation statistics in arcseconds.
    """
    from astropy.wcs import WCS

    T = build_wcs_transform(header)
    wcs = WCS(header)

    rng = np.random.default_rng(42)
    px = rng.uniform(0, T['naxis1'] - 1, n_samples)
    py = rng.uniform(0, T['naxis2'] - 1, n_samples)

    # Our trig-free transform
    vx, vy, vz = pixel_to_unit_icrs(px, py, T)
    ra_ours, dec_ours = unit_icrs_to_radec(vx, vy, vz)

    # Astropy (through WCSLIB)
    ra_ap, dec_ap = wcs.all_pix2world(px, py, 0)

    # Angular separation via Vincenty formula (arcseconds)
    r1, r2 = np.radians(ra_ours), np.radians(ra_ap)
    d1, d2 = np.radians(dec_ours), np.radians(dec_ap)
    dra = r1 - r2
    num = np.sqrt((np.cos(d2)*np.sin(dra))**2 +
                  (np.cos(d1)*np.sin(d2) - np.sin(d1)*np.cos(d2)*np.cos(dra))**2)
    den = np.sin(d1)*np.sin(d2) + np.cos(d1)*np.cos(d2)*np.cos(dra)
    sep_arcsec = np.degrees(np.arctan2(num, den)) * 3600.0

    result = {
        'n_samples': n_samples,
        'max_error_arcsec': float(np.max(sep_arcsec)),
        'mean_error_arcsec': float(np.mean(sep_arcsec)),
        'median_error_arcsec': float(np.median(sep_arcsec)),
    }

    print(f"\nValidation against astropy WCS ({n_samples} random pixels):")
    print(f"  Max error:    {result['max_error_arcsec']:.6e} arcsec")
    print(f"  Mean error:   {result['mean_error_arcsec']:.6e} arcsec")
    print(f"  Median error: {result['median_error_arcsec']:.6e} arcsec")

    tess_pixel_arcsec = 21.0
    print(f"  (TESS pixel scale: {tess_pixel_arcsec} arcsec/pixel)")
    print(f"  Max error = {result['max_error_arcsec']/tess_pixel_arcsec:.2e} pixels")

    return result


# ═══════════════════════════════════════════════════════════════════════
#  CUDA kernel (for when you're ready to port)
# ═══════════════════════════════════════════════════════════════════════

CUDA_KERNEL = r"""
// Upload once per WCS:
__constant__ double c_crpix[2];     // {CRPIX1-1, CRPIX2-1}  (0-based offset)
__constant__ double c_cd[4];        // {CD1_1, CD1_2, CD2_1, CD2_2}
__constant__ double c_R[9];         // rotation matrix (row-major)
__constant__ double c_sip_a[15];    // SIP A coefficients (packed: A_2_0..A_0_4)
__constant__ double c_sip_b[15];    // SIP B coefficients
__constant__ int    c_sip_order;    // SIP polynomial order

// Per-pixel: 0 trig, 0 branches, ~55 FLOPs + 1 rsqrt
__global__ void pixel_to_icrs(
    double* __restrict__ out_vx,
    double* __restrict__ out_vy,
    double* __restrict__ out_vz,
    int width, int height)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col >= width || row >= height) return;
    int idx = row * width + col;

    // Step 1: pixel offset
    double u = (double)col - c_crpix[0];
    double v = (double)row - c_crpix[1];

    // Step 2: SIP distortion (unrolled for order 4)
    double u2 = u*u, v2 = v*v, uv = u*v;
    double u3 = u2*u, v3 = v2*v;
    double u4 = u2*u2, v4 = v2*v2;
    double du = c_sip_a[0]*u2 + c_sip_a[1]*uv + c_sip_a[2]*v2
              + c_sip_a[3]*u3 + c_sip_a[4]*u2*v + c_sip_a[5]*uv*v + c_sip_a[6]*v3
              + c_sip_a[7]*u4 + c_sip_a[8]*u3*v + c_sip_a[9]*u2*v2 + c_sip_a[10]*uv*v2 + c_sip_a[11]*v4;
    double dv = c_sip_b[0]*u2 + c_sip_b[1]*uv + c_sip_b[2]*v2
              + c_sip_b[3]*u3 + c_sip_b[4]*u2*v + c_sip_b[5]*uv*v + c_sip_b[6]*v3
              + c_sip_b[7]*u4 + c_sip_b[8]*u3*v + c_sip_b[9]*u2*v2 + c_sip_b[10]*uv*v2 + c_sip_b[11]*v4;
    double us = u + du;
    double vs = v + dv;

    // Step 3: CD matrix
    double x = c_cd[0]*us + c_cd[1]*vs;
    double y = c_cd[2]*us + c_cd[3]*vs;

    // Step 4: TAN deprojection (trig-free!)
    double nx = -y;
    double ny =  x;
    // nz = R0 = 57.29577951... (compile-time constant)
    const double R0 = 180.0 / 3.14159265358979323846;

    // Step 5: Rotate native → ICRS
    double wx = c_R[0]*nx + c_R[1]*ny + c_R[2]*R0;
    double wy = c_R[3]*nx + c_R[4]*ny + c_R[5]*R0;
    double wz = c_R[6]*nx + c_R[7]*ny + c_R[8]*R0;

    // Step 6: Normalize
    double inv_norm = rsqrt(wx*wx + wy*wy + wz*wz);
    out_vx[idx] = wx * inv_norm;
    out_vy[idx] = wy * inv_norm;
    out_vz[idx] = wz * inv_norm;
}
"""


# ═══════════════════════════════════════════════════════════════════════
#  Self-contained test with the TESS FFI header (Sector 4, Camera 1 CCD 4)
# ═══════════════════════════════════════════════════════════════════════

def make_tess_test_header():
    """Build a minimal FITS header dict with the WCS keywords from
    the user's TESS FFI (Sector 4, Camera 1, CCD 4)."""
    return {
        'NAXIS1': 2048, 'NAXIS2': 2048,
        'CTYPE1': 'RA---TAN-SIP', 'CTYPE2': 'DEC--TAN-SIP',
        'CRVAL1':  33.9220620718762600,
        'CRVAL2':   1.3757010398066702,
        'CRPIX1': 1001.0, 'CRPIX2': 1001.0,
        'CD1_1': -0.005410076863953,
        'CD1_2':  0.00171852128744147,
        'CD2_1': -0.001850933895580,
        'CD2_2': -0.005448150209205,
        'A_ORDER': 4, 'B_ORDER': 4,
        'A_2_0': -1.912760306632E-05, 'A_1_1':  1.704286159867E-05,
        'A_0_2': -2.964525745984E-06, 'A_3_0': -2.774535353715E-09,
        'A_2_1':  1.588310303521E-10, 'A_1_2': -2.884679846948E-09,
        'A_0_3':  2.366943787189E-10, 'A_4_0': -2.362760769233E-13,
        'A_3_1':  3.526672024282E-13, 'A_2_2': -3.520226252598E-13,
        'A_1_3':  2.538477092313E-13, 'A_0_4': -9.662159796600E-14,
        'B_2_0':  3.105552648743E-06, 'B_1_1': -1.642745933175E-05,
        'B_0_2':  1.981608227294E-05, 'B_3_0':  2.152216479199E-10,
        'B_2_1': -2.903783853793E-09, 'B_1_2':  1.590454156397E-10,
        'B_0_3': -2.786344423182E-09, 'B_4_0':  7.522395766081E-14,
        'B_3_1': -2.832170944601E-13, 'B_2_2':  3.496996398765E-13,
        'B_1_3': -3.394815570443E-13, 'B_0_4':  2.622391178501E-13,
        'A_DMAX': 46.51183583994646, 'B_DMAX': 46.779221083899756,
        # Inverse SIP (for world → pixel, not needed here but included for completeness)
        'AP_ORDER': 4, 'BP_ORDER': 4,
        'AP_1_0': -8.115432076330E-05, 'AP_0_1': 5.281320103066225E-05,
        'AP_2_0': 1.912596257792368E-05, 'AP_1_1': -1.704167127557E-05,
        'AP_0_2': 2.959658833224E-06, 'AP_3_0': 3.729360594920E-09,
        'AP_2_1': -1.645751141087E-09, 'AP_1_2': 3.951926785720087E-09,
        'AP_0_3': -4.818424968658E-10, 'AP_4_0': 5.636313313086E-13,
        'AP_3_1': -7.513897199631E-13, 'AP_2_2': 8.120411537871E-13,
        'AP_1_3': -6.181288100400E-13, 'AP_0_4': 1.612051466228E-13,
        'BP_1_0': 5.254454525532626E-05, 'BP_0_1': -8.281219825626E-05,
        'BP_2_0': -3.102413796374E-06, 'BP_1_1': 1.642764250810E-05,
        'BP_0_2': -1.981375285211E-05, 'BP_3_0': -4.593640767052E-10,
        'BP_2_1': 3.943349064442731E-09, 'BP_1_2': -1.647322694151E-09,
        'BP_0_3': 3.798649219183717E-09, 'BP_4_0': -1.363657358488E-13,
        'BP_3_1': 6.347845462898E-13, 'BP_2_2': -8.199374978838E-13,
        'BP_1_3': 7.323325631036E-13, 'BP_0_4': -6.063611809855E-13,
    }

# ═══════════════════════════════════════════════════════════════════════


def plot_icrs_for_all_in_sector():
    import glob, os
    from os import listdir
    from os.path import isfile, join
    from astropy.io import fits

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
    #files = list(filter(filter_file, files))

    files.sort()
    nfiles = len(files)
    print(f"{nfiles = }, {files[0] = }")
    globals().update(locals())

    # prep vertex buffer
    ##long = np.linspace(0, 2048, 32)
    #l1 = np.arange(0, 2048, 32)
    #l0 = l1*0+1
    #e0 = np.array([l1, l0*0])
    #e1 = np.array([l0*0, l1])
    #line0 = e0
    #line1 = line0[-1] + e1
    #line2 = line1[-1] - e0
    #line3 = line2[-1] - e1
    # Add on one additional point for the last step
    # Build boundary vertex buffer from image dimensions.
    # Moved inside loop so it adapts to each file's NAXIS.

    #l1 = np.arange(0, 2048+32, 32)
    #l1[-1] = nx - 1  # clamp to last valid pixel
    samples = 64
    samples = 16
    l1 = np.linspace(0, 1, samples)
    l0 = np.zeros_like(l1)
    xhat = np.array([l1, l0]) # (2, N)
    yhat = np.array([l0, l1])
    # Build 4 edges of the boundary
    line0 =  xhat                # bottom: (0,0)→(nx-1,0)
    line1 =  yhat + line0[:,-1:] # right:  (nx-1,step)→(nx-1,ny-1)
    line2 = -xhat + line1[:,-1:] # top:    (nx-1-step,ny-1)→(0,ny-1)
    line3 = -yhat + line2[:,-1:] # left:   (0,ny-1-step)→(0,0)
    vb0 = np.hstack([line0, line1[:,1:], line2[:,1:], line3[:,1:]]).T.astype(np.float64)  # (N, 2) raw pixel coords
    vb = vb0.copy() # This will be modified in place by the distortion function, so we keep a copy of the original raw pixel coords.
    distort = np.zeros_like(vb)
    vbxyz = np.pad(vb, ((0,0),(0,1)), mode='constant', constant_values=0.0) # (N, 3) array to hold the final ICRS vec3 results. We can reuse this for the tan_deproject_rotate_normalize step to avoid an extra allocation.

    AB = np.zeros((2, 5, 5), dtype=np.float64) # [A/B, i, j]

    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(projection='3d', facecolor='black')) # Set figure background to black
    # Invert colors of background and all text
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Get colors from a colormap
    cmap = plt.get_cmap('viridis')
    colors = [cmap(p)[:3] for p in np.linspace(0, 1, 16)]

    import time
    t0 = time.time()
    t00 = t0
    nfiles_since_last_print = 0
    nfiles_read = 0
    nskipped = 0

    timers = ProfileTimer()

    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    # Plot in gonomic? or is this true 3d coords? 
    for nfile, file in enumerate(files):
        with timers['iter']:
            if nfile == 1:
                t00 = time.time() # reset timer after the first file, which is often an outlier due to disk spin-up or caching effects. This gives a more representative time estimate for the remaining files.

            data, header = fits.getdata(file, header=True)
            #header = fits.getheader(file) # This is much faster than getdata with memmap, which still reads the header of the image extension. We only need the header for the WCS info, so we can skip reading the image data entirely here.
            # Try the with variant with hdus:
            # with fits.open(file, memmap=True) as hdul:
            #     print(f"{type(hdul[0]) = }") # There's 3 of them!
            #     print(f"{hdul.info()}") # 0 is primary (empty), 1 is image extension, 2 is binary table with the same header as 1. So we can read the header from either 1 or 2, but we should read the data from 1.
            #     data = hdul[1].data

            # Debug code I'll paste somewhere else
            # s = set()
            # # Lets get a list of all the hduls in every file and put them into a set.
            # for file in files:
            #     with fits.open(file, memmap=True) as hdul:
            #         s.add(len(hdul))

            # Extract camera and ccd from filename
            cam, ccd = get_camera_and_ccd(file)
            color = colors[(int(cam)-1)*4 + (int(ccd)-1)]

            try:
                T = build_wcs_transform(header)
            except Exception as e:
                nskipped += 1
                continue
                globals().update(locals())
                raise e

            # Build prep objects
            naxis = np.array([T['naxis1'], T['naxis2']]).astype(np.float64) # (nx, ny) indexing
            crpix = np.array([T['crpix1'], T['crpix2']]).astype(np.float64) # (x, y) indexing
            AB.fill(0.0)
            for (i, j), c in T['a_coeffs'].items():
                AB[0, i, j] = c
            for (i, j), c in T['b_coeffs'].items():
                AB[1, i, j] = c
            #print(f"{naxis.dtype = }", f"{crpix.dtype = }", f"{AB.dtype = }")
            # Error if any of this is nan
            if np.isnan(naxis).any() or np.isnan(crpix).any() or np.isnan(AB).any() or np.isnan(T['cd']).any(): 
                # Find the data burglar
                e = ""
                e += "naxis is NaN. " if np.isnan(naxis).any() else ""
                e += "crpix is NaN. " if np.isnan(crpix).any() else ""
                e += "AB is NaN. " if np.isnan(AB).any() else ""
                e += "CD is NaN. " if np.isnan(T['cd']).any() else ""
                print(f"NaN  in file {nfile+1}/{nfiles}: {file.name}: {e}")
                globals().update(locals())
                raise e
                continue


            #vb = make_boundary_vb(int(naxis[0]), int(naxis[1])) # Raw pixel coords (N, 2)
            # Reset from [0,1] template, scale to pixel coords, subtract CRPIX in one fused op:
            #   vb = vb0 * (naxis-1)  →  raw pixel coords [0, naxis-1]
            #   vb -= (crpix - 1)     →  CRPIX-relative coords (Step 1)
            vb[:] = vb0 * (naxis - 1) - (crpix - 1)  # Step 1: [0,1] → pixel offset from CRPIX

            distort_sip(distort, vb, AB) # Step 2: SIP distortion polynomial. This modifies vb **in place** to apply the SIP distortion. After this step, vb contains the distorted pixel coordinates.
            vb[:] += distort # Apply the SIP distortion to the pixel coordinates
            vb[:] = vb @ T['cd'].T # Step 3: CD matrix. This applies the CD matrix to the distorted pixel coordinates, resulting in intermediate world coordinates (x, y) in degrees. Indexing (N, 2).

            #print(f"{vb.shape = }, {vb.dtype = }, {vb[:5] = }")
            #color = 'white'
            #color = colors[(cam, ccd)]
            #ax.plot(*vb.T, color=color, alpha=0.5) # Plot the intermediate world coordinates (x, y) after the CD matrix. This is still in the tangent plane, not yet deprojected to ICRS. We can see the effect of the SIP distortion here as deviations from a regular grid.

            # Step 4: TAN deprojection. This maps the (x, y) coordinates on the tangent plane to native Cartesian coordinates (nx, ny, nz). Since nz is constant (R0), we can just compute nx and ny and keep track of the fact that we're in a 3D space where z = R0. Indexing (N, 2) for (nx, ny).
            tan_deproject_rotate_normalize(vbxyz, vb, T['R'].astype(np.float64)) # This modifies vbxyz in place to contain the final unit ICRS direction vectors (vx, vy, vz).
            # This projection is called gnomonic, and it maps great circles to straight lines. So the grid lines should still look like straight lines, just distorted by the SIP polynomial and the CD matrix. If we plotted the intermediate (x, y) coordinates after the CD matrix but before the TAN deprojection, we would see the SIP distortion more clearly as deviations from a regular grid. After the TAN deprojection and rotation to ICRS, we are plotting points on the unit sphere in 3D space, so the grid will look more distorted due to the projection effects.


            # TEST: Lets validate this against my new method.
            # keys = wcs_variables_validate(header)
            # TESS_Mapping_Variables = wcs_variables_load_nonan(header, keys)
            # TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)
            # vb1 = vb0 * (TESS_Mapping_Struct.naxis_i - 1)
            # vbxyz1 = wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, vb1)
            # vbxyz1[:] /= np.linalg.norm(vbxyz1, axis=1, keepdims=True) # Normalize to unit vectors
            # print(f"{np.max(np.linalg.norm(vbxyz - vbxyz1, axis=1)) = }")
            # assert np.allclose(vbxyz, vbxyz1, atol=1e-6) # This is good agreement, within the expected numerical precision of the operations. The small differences are likely due to the order of operations and floating point rounding, but they are negligible for our purposes.

            #ax.plot(*vb.T, color=color, alpha=0.5)
            if hasattr(ax, 'get_zlim'):
                #ax.plot(*vbxyz.T, color=color, alpha=0.5)
                mins = np.minimum(mins, vbxyz.min(axis=0))
                maxs = np.maximum(maxs, vbxyz.max(axis=0))
            else: # 2d

                vb[:, 0] = vbxyz[:, 0] # Update vb with the ICRS vx
                vb[:, 0] = vbxyz[:, 1] # Update vb with the ICRS vy
                vb[:, 1] = vbxyz[:, 2] # Update vb with the ICRS vz (not used for plotting but included for completeness)

                ax.plot(*vb.T, color=color, alpha=0.5)

            #print(f"Processed file {nfile+1}/{nfiles}: {file.name}")
            # Print if enough time has elapsed, *or* one file has been processed if its slow.
            if nfile > 256:
                #breakpoint() # Too many files, let's debug this one before we plot them all.
                # huh, I didn't know breakpoint was a thing in Python. This is super nice for debugging! Lets use this to debug the first file, then we can remove it and run through all the files.
                #break
                pass

            t1 = time.time()
            nfiles_since_last_print += 1
            nfiles_read += 1
            if t1 - t0 > 5.0:# or nfiles_since_last_print >= 1:
                print(f"Processed file {nfile+1}/{nfiles}: {file.name}, rate: {(nfiles_read-1)/(t1-t00):.1f} files/s")
                t0 = t1
                nfiles_since_last_print = 0
    print(f"Done. Plotted {nfiles - nskipped}/{nfiles} files, skipped {nskipped} ({100*nskipped/nfiles:.1f}%) with missing WCS.")
    print(timers)
    
    if isinstance(ax, plt.Axes) and hasattr(ax, 'get_zlim'):
        #ax.set_box_aspect(maxs - mins)  # Set aspect ratio based on the range of the data in each dimension
        ax.set_aspect('equal')

        ax.set_xlabel('ICRS X', color='white')
        ax.set_ylabel('ICRS Y', color='white')
        ax.set_zlabel('ICRS Z', color='white')
        #ax.set_title(f'TESS Sector 4, Camera {camera}, CCD {ccd}\nSIP-distorted grid in ICRS', color='white')
        #ax.set_title(f'TESS Sector 4\nSIP-distorted grid in ICRS', color='white')
    else:
        ax.set_aspect(1)
        ax.set_xlabel('Tangent Plane X (degrees)', color='white')
        ax.set_ylabel('Tangent Plane Y (degrees)', color='white')
    #ax.set_title(f'TESS Sector 4, Camera {camera}, CCD {ccd}\nSIP-distorted grid in tangent plane', color='white')
    #ax.set_title(f'TESS Sector 4\nSIP-distorted grid in tangent plane', color='white')
    plt.show()

    globals().update(locals())

#plot_icrs_for_all_in_sector()

# From above:
# keys = wcs_variables_validate(header)
# TESS_Mapping_Variables = wcs_variables_load_nonan(header, keys)
# TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)
# wcs_map_ICRS_from_TESS(TESS_Mapping_Struct, np.array([[1000.0, 1000.0], [1500.0, 1500.0]]))

def plot_icrs_for_all_in_sector_custom():
    import glob, os
    from os import listdir
    from os.path import isfile, join
    from astropy.io import fits

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
    #files = list(filter(filter_file, files))

    files.sort()
    nfiles = len(files)
    print(f"{nfiles = }, {files[0] = }")
    globals().update(locals())

    #l1 = np.arange(0, 2048+32, 32)
    #l1[-1] = nx - 1  # clamp to last valid pixel
    #samples = 64
    #samples = 16
    samples = 256
    l1 = np.linspace(0, 1, samples)
    l0 = np.zeros_like(l1)
    xhat = np.array([l1, l0]) # (2, N)
    yhat = np.array([l0, l1])
    # Build 4 edges of the boundary
    line0 =  xhat                # bottom: (0,0)→(nx-1,0)
    line1 =  yhat + line0[:,-1:] # right:  (nx-1,step)→(nx-1,ny-1)
    line2 = -xhat + line1[:,-1:] # top:    (nx-1-step,ny-1)→(0,ny-1)
    line3 = -yhat + line2[:,-1:] # left:   (0,ny-1-step)→(0,0)
    vb0 = np.hstack([line0, line1[:,1:], line2[:,1:], line3[:,1:]]).T.astype(np.float64)  # (N, 2) raw pixel coords
    vb = vb0.copy() # This will be modified in place by the distortion function, so we keep a copy of the original raw pixel coords.
    distort = np.zeros_like(vb)
    vbxyz = np.pad(vb, ((0,0),(0,1)), mode='constant', constant_values=0.0) # (N, 3) array to hold the final ICRS vec3 results. We can reuse this for the tan_deproject_rotate_normalize step to avoid an extra allocation.

    AB = np.zeros((2, 5, 5), dtype=np.float64) # [A/B, i, j]

    import matplotlib.pyplot as plt
    #fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='black', subplot_kw=dict(projection='3d', facecolor='black')) # Set figure background to black
    # Invert colors of background and all text
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    # Get colors from a colormap
    cmap = plt.get_cmap('viridis')
    colors = [cmap(p)[:3] for p in np.linspace(0, 1, 16)]

    import time
    t0 = time.time()
    t00 = t0
    nfiles_since_last_print = 0
    nfiles_read = 0
    nskipped = 0

    timers = ProfileTimer()
    cudaContainer = CudaDistortionMapper() 

    mins = np.array([np.inf, np.inf, np.inf])
    maxs = np.array([-np.inf, -np.inf, -np.inf])
    # Plot in gonomic? or is this true 3d coords? 
    for nfile, file in enumerate(files):
        with timers['iter']:
            if nfile == 1:
                t00 = time.time() # reset timer after the first file, which is often an outlier due to disk spin-up or caching effects. This gives a more representative time estimate for the remaining files.

            with timers["fits getdata"]:
                data, header = fits.getdata(file, header=True)

            # Extract camera and ccd from filename
            with timers["get_camera_and_ccd"]:
                cam, ccd = get_camera_and_ccd(file)
                color = colors[(int(cam)-1)*4 + (int(ccd)-1)]

            try:
                with timers["wcs_variables_validate"]:
                    keys = wcs_variables_validate(header)
            except Exception as e:
                nskipped += 1
                continue
                globals().update(locals())
                raise e

            with timers["wcs_variables_load_nonan"]:
                TESS_Mapping_Variables = wcs_variables_load_nonan(header, keys)
            with timers["wcs_assemble_struct"]:
                TESS_Mapping_Struct = wcs_assemble_struct(TESS_Mapping_Variables)
            with timers["vb reset"]:
                vb[:] = vb0 * (TESS_Mapping_Struct.naxis_i - 1)
            with timers["wcs_map_ICRS_from_TESS_SLOW"]:
                vbxyz[:] = wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, vb)
            with timers["wcs_map_TESS_from_ICRS_SLOW"]:
                vb[:] = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, vbxyz)

            with timers["vb reset"]:
                vb[:] = vb0# * (TESS_Mapping_Struct.naxis_i - 1)
            with timers["wcs_map_ICRS_from_TESS"]:
                wcs_map_ICRS_from_TESS(vbxyz, distort, TESS_Mapping_Struct, vb) # This modifies vbxyz in place to contain the final ICRS vectors.
            with timers["wcs_map_TESS_from_ICRS"]:
                wcs_map_TESS_from_ICRS(vb, distort, TESS_Mapping_Struct, vbxyz)

            with timers["vb reset"]:
                vb[:] = vb0 * (TESS_Mapping_Struct.naxis_i - 1)

            with timers['wcs_map_ICRS_from_TESS_cuda']:
                vb3 = wcs_map_ICRS_from_TESS_cuda(cudaContainer, TESS_Mapping_Struct, vb)

            vbxyz[:] /= np.linalg.norm(vbxyz, axis=1, keepdims=True) # Normalize to unit vectors
            vb3[:] /= np.linalg.norm(vb3, axis=1, keepdims=True) # Normalize to unit vectors
            print(f"{np.max(np.linalg.norm(vbxyz - vb3, axis=1)) = }")
            

            
            #vb[:] = vb0# * (TESS_Mapping_Struct.naxis_i - 1)
            #vbxyz[:] = wcs_map_ICRS_from_TESS_SLOW(TESS_Mapping_Struct, vb)
            vbxyz[:] = vb3

            #

            vbxyz[:] /= np.linalg.norm(vbxyz, axis=1, keepdims=True) # Normalize to unit vectors
            #wcs_map_ICRS_from_TESS(vbxyz, distort, TESS_Mapping_Struct, vb) # This should give the same result as the full method above, but it's optimized to fuse all the steps together and avoid intermediate allocations. It modifies vbxyz in place to contain the final ICRS vectors.

            # Map back to validate
            #wcs_map_TESS_from_ICRS(vb, distort, TESS_Mapping_Struct, vbxyz)
            #vb2 = wcs_map_TESS_from_ICRS_SLOW(TESS_Mapping_Struct, vbxyz) # This should give us back the original pixel coordinates (after scaling and CRPIX offset), within the expected numerical precision of the operations. It modifies vb in place to contain the pixel coordinates.
            #print(f"{vb0[:5] = }") # These should be the same as the original vb0 pixel coordinates (after scaling and CRPIX offset), within the expected numerical precision of the operations. Small differences are likely due to floating point rounding, but they should be negligible for our purposes.
            #print(f"{vb[:5] = }")  # These should match vb0[:5] if the mapping is correct. This serves as a sanity check that the forward and inverse mappings are consistent with each other.
            #print(f"{np.max(np.linalg.norm(vb0 - vb, axis=1)) = }") # This should be 1.0 if the vectors are properly normalized to unit length on the sphere.
            #globals().update(locals())
            #return


            #ax.plot(*vb.T, color=color, alpha=0.5)
            if hasattr(ax, 'get_zlim'):
                ax.plot(*vbxyz.T, color=color, alpha=0.5)
                mins = np.minimum(mins, vbxyz.min(axis=0))
                maxs = np.maximum(maxs, vbxyz.max(axis=0))
            else: # 2d

                vb[:, 0] = vbxyz[:, 0] # Update vb with the ICRS vx
                vb[:, 0] = vbxyz[:, 1] # Update vb with the ICRS vy
                vb[:, 1] = vbxyz[:, 2] # Update vb with the ICRS vz (not used for plotting but included for completeness)

                ax.plot(*vb.T, color=color, alpha=0.5)

            #print(f"Processed file {nfile+1}/{nfiles}: {file.name}")
            # Print if enough time has elapsed, *or* one file has been processed if its slow.
            if nfile > 256:
                #breakpoint() # Too many files, let's debug this one before we plot them all.
                # huh, I didn't know breakpoint was a thing in Python. This is super nice for debugging! Lets use this to debug the first file, then we can remove it and run through all the files.
                break
                pass

            t1 = time.time()
            nfiles_since_last_print += 1
            nfiles_read += 1
            if t1 - t0 > 5.0:# or nfiles_since_last_print >= 1:
                print(f"Processed file {nfile+1}/{nfiles}: {file.name}, rate: {(nfiles_read-1)/(t1-t00):.1f} files/s")
                t0 = t1
                nfiles_since_last_print = 0
    print(f"Done. Plotted {nfiles_read - nskipped}/{nfiles_read} files of {nfiles_read}, skipped {nskipped} ({100*nskipped/nfiles_read:.1f}%) with missing WCS.")
    print(timers)
    
    if isinstance(ax, plt.Axes) and hasattr(ax, 'get_zlim'):
        ax.set_box_aspect(maxs - mins)  # Set aspect ratio based on the range of the data in each dimension
        #ax.set_aspect('equal')

        ax.set_xlabel('ICRS X', color='white')
        ax.set_ylabel('ICRS Y', color='white')
        ax.set_zlabel('ICRS Z', color='white')
        #ax.set_title(f'TESS Sector 4, Camera {camera}, CCD {ccd}\nSIP-distorted grid in ICRS', color='white')
        #ax.set_title(f'TESS Sector 4\nSIP-distorted grid in ICRS', color='white')
    else:
        ax.set_aspect(1)
        ax.set_xlabel('Tangent Plane X (degrees)', color='white')
        ax.set_ylabel('Tangent Plane Y (degrees)', color='white')
    #ax.set_title(f'TESS Sector 4, Camera {camera}, CCD {ccd}\nSIP-distorted grid in tangent plane', color='white')
    #ax.set_title(f'TESS Sector 4\nSIP-distorted grid in tangent plane', color='white')
    plt.show()

    globals().update(locals())

# ═══════════════════════════════════════════════════════════════════════

# if __name__ == '__main__':
#     header = make_tess_test_header()
#     T = build_wcs_transform(header)

#     # Show the four corners + center
#     print_corners(T)

#     # Time the full-image map
#     print("Generating full 2048×2048 ICRS direction map...")
#     t0 = time.perf_counter()
#     icrs_map = make_icrs_map(T)
#     t1 = time.perf_counter()
#     print(f"  {T['naxis1']}×{T['naxis2']} = {T['naxis1']*T['naxis2']:,} pixels in {t1-t0:.3f}s")
#     print(f"  Output shape: {icrs_map.shape}, dtype: {icrs_map.dtype}")

#     # Verify all vectors are unit length
#     norms = np.sqrt(np.sum(icrs_map**2, axis=-1))
#     print(f"  ‖v‖ range: [{norms.min():.15f}, {norms.max():.15f}]  (should be 1.0)")
#     print()

#     # Validate against astropy
#     try:
#         from astropy.io.fits import Header
#         # Build a proper FITS Header object for astropy WCS
#         fits_header = Header()
#         for k, v in header.items():
#             fits_header[k] = v
#         validate(fits_header)
#     except ImportError:
#         print("astropy not installed — skipping validation.")
#         print("Install with: pip install astropy")

if __name__ == '__main__':
    plot_icrs_for_all_in_sector_custom()
