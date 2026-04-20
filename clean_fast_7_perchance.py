

# This is not clean_fast_7. These are changes I've been applying to the shell after running clean_fast_6. This script will not run lol

def hybrid_forward_and_jac(params, img, med, ky0, kx0, hc, cx, cy, Y, X,
                           blur=None, sqrtW=None):
    A, B, dy, dx, a, b, c, d = params
    Λ = np.array([[a, b],
                  [c, d]], dtype=np.float64)
    Λi = np.linalg.inv(Λ)

    # 1) exact subpixel shift in Fourier
    ph = np.exp(-2j*np.pi*(ky0*dy + kx0*dx))
    fft = np.fft.fft2(img) * ph
    fft = fft if blur is None else fft * blur   # this changes the forward model

    s = np.fft.ifft2(fft).real

    # 2) source gradients of shifted image
    sy = np.fft.ifft2(fft * (2j*np.pi*ky0)).real
    sx = np.fft.ifft2(fft * (2j*np.pi*kx0)).real

    # 3) warp image and gradients once
    w  = hc.affine_resample(s,  Λi)
    wy = hc.affine_resample(sy, Λi)
    wx = hc.affine_resample(sx, Λi)

    # 4) destination-centered coordinates
    Uy = Λ[0,0]*(Y-cy) + Λ[0,1]*(X-cx)
    Ux = Λ[1,0]*(Y-cy) + Λ[1,1]*(X-cx)

    # g^T J e1 and g^T J e2
    gJe1 = wy*Λ[0,0] + wx*Λ[1,0]
    gJe2 = wy*Λ[0,1] + wx*Λ[1,1]

    dw_da = -Uy * gJe1
    dw_db = -Ux * gJe1
    dw_dc = -Uy * gJe2
    dw_dd = -Ux * gJe2

    pred = A*w + B
    r = pred - med

    # residual Jacobian columns
    Jcols = [
        w,
        np.ones_like(w),
        A * (-wy),
        A * (-wx),
        A * dw_da,
        A * dw_db,
        A * dw_dc,
        A * dw_dd,
    ]

    # Apply spectral metric after the residual/Jacobian are formed.
    # This preserves the least-squares / Gauss-Newton structure.
    if sqrtW is not None:
        r_w = np.fft.ifft2(np.fft.fft2(r) * sqrtW).real
        Jcols_w = [
            np.fft.ifft2(np.fft.fft2(c) * sqrtW).real
            for c in Jcols
        ]
    else:
        r_w = r
        Jcols_w = Jcols

    L = 0.5 * np.mean(r_w * r_w)
    J = np.array([np.mean(r_w * c) for c in Jcols_w], dtype=np.float64)
    H = np.array([[np.mean(ci * cj) for cj in Jcols_w] for ci in Jcols_w],
                 dtype=np.float64)

    return L, J, H

def big_iterator(epochs_per_frame=10, total_epochs=100, dset_limit=None, dsize=32, η=1e-3, λ=1e-1, mrate=.1):
    #seq = sector_data[slice(dset_limit), slice(sector_data.shape[1]//2-dsize//2,sector_data.shape[1]//2+dsize//2), slice(sector_data.shape[2]//2-dsize//2,sector_data.shape[2]//2+dsize//2)]
    seq = sd.copy()[slice(dset_limit), slice(sector_data.shape[1]//2-dsize//2,sector_data.shape[1]//2+dsize//2), slice(sector_data.shape[2]//2-dsize//2,sector_data.shape[2]//2+dsize//2)]
    #seq = seq - seq.min()# - seq.min(axis=(1,2))[:,None,None]
    #seq = slog(seq)
    hc = CUDAHcongridContainer()
    hc.initialize_affine(*seq[0].shape)

    global params_dset
    params_dset = np.zeros((len(seq), 8), dtype=np.float64)
    params_dset[:,0] = 1.0 # Initialize to A=1, and an identity affine
    params_dset[:,4] = 1.0
    params_dset[:,7] = 1.0

    #med = np.asarray(sector_median, dtype=np.float64)
    #med = np.copy(seq[0])
    #med = np.median(seq[:64], axis=0)
    #med = np.mean(seq[:64], axis=0)
    med = seq[0].copy()


    ky, kx = centered_k(seq[0].shape)
    ky0 = np.fft.ifftshift(ky)
    kx0 = np.fft.ifftshift(kx)
    #blursize = 1.0#16#2.0 #pixels
    blursize = 4.0
    blur = np.exp(-0.5*(kx0**2 + ky0**2)*(blursize*max(kx.max(), ky.max())/seq[0].shape[0])**2)  # Gaussian blur in frequency domain to improve stability
    #blur = np.power(blur, 10)
    #kr = (kx**2 + ky**2)**.5
    kr = (kx0**2 + ky0**2)**.5
    #blur = airy_disk(kr*np.pi*2.0)
    #b = np.fft.ifft2(blur).real.sum()
    #blur /= b
    #plt.imshow(blur)
    #plt.show()

    Y, X = np.indices(seq[0].shape, dtype=np.float64)

    vmin, vmax = None, None
    nmin, nmax = 1e30, -1e30

    sector_cache = np.empty_like(seq, dtype=np.float64)
    par_m = np.zeros_like(params_dset)
    par_v = np.zeros_like(params_dset)
    β1,β2 = 0.9,0.999

    shrp = np.array([η, λ], dtype=np.float64)
    shrp_m = np.zeros_like(shrp)
    shrp_v = np.zeros_like(shrp)

    mfft = np.fft.fft2(med)
    #mfft = med
    med_m = np.zeros_like(mfft)
    med_v = np.zeros_like(mfft)
    mβ1, mβ2 = 0.9,0.999

    q2 = (2*np.pi)**2 * (kx0*kx0 + ky0*ky0)
    alpha, beta = 1e-2, 1e-4
    W = (1.0 + alpha*q2) / (1.0 + beta*q2*q2)
    sqrtW = np.sqrt(W)

    t0 = time_ns()
    for epoch in range(total_epochs):
        rolling_mean = np.zeros_like(med)
        rolling_sum = 0.0
        #chunklim = (1+epoch)*1
        chunklim = len(seq)
        idx = np.arange(len(seq))
        #np.random.shuffle(idx)
        for framenum in range(chunklim):
            img = seq[idx[framenum]]
            params = params_dset[idx[framenum]]
            #trgt = med# if 0 < framenum else seq[0]
            trgt = seq[0]

            for minibatch in range(epochs_per_frame):
                #L,J,H = hybrid_forward_and_jac(params, img, med, ky0, kx0, hc, cx=img.shape[1]/2, cy=img.shape[0]/2, Y=Y, X=X, blur=blur)
                L,J,H = hybrid_forward_and_jac(params, img, trgt, ky0, kx0, hc, cx=img.shape[1]/2, cy=img.shape[0]/2, Y=Y, X=X, blur=None, sqrtW=None)
                #H = H + np.diag(np.diag(H))*1e-6  # Lavenberg-Marquardt style damping to improve stability
                #step = J
                try:
                    step = np.linalg.solve(H,J)
                except np.linalg.LinAlgError:
                    step = J / (np.dot(J,J) + 1e-8)  # Fallback to gradient step if Hessian is singular
                rate = np.array([1,1,.5,.5,.1,.1,.1,.1])# * np.exp(-epoch) + 1.0*(1.0-np.exp(-epoch))
                #if epoch >= 2:
                #    rate = np.array([1.0]*8)
                #params -= step*np.array([1,1,.5,.5,.1,.1,.1,.1])#*1e-3
                #params -= step*rate
                #step = J+0.0
                #par_m[framenum,:] = β1*par_m[framenum,:] + (1-β1)*step
                #par_v[framenum,:] = β2*par_v[framenum,:] + (1-β2)*step**2
                #mhat = par_m[framenum,:]/(1+β1**(1 + epoch*epochs_per_frame+minibatch))
                #vhat = par_v[framenum,:]/(1+β2**(1 + epoch*epochs_per_frame+minibatch))
                #step = mhat/(vhat**.5 + 1e-8)
                step = step*rate#*.1#*.01
                #step = J+0.0
                #params -= step*mrate#*1e-3
                #try:
                #    step = np.linalg.solve(H,step)
                #except np.linalg.LinAlgError:
                #    pass
                params -= step*mrate
                params[:2] = [1.0,0.0]

                """
                result = forward_model(params, img, kx0, ky0, hc)
                diff = result - med # np.arcsinh(result - med)
                #dfft = np.fft.fft2(diff)*sqrtW
                dfft = diff
                med_m[:] = med_m*mβ1 + (1-mβ1)*dfft
                med_v[:] = med_v*mβ2 + (1-mβ2)*np.abs(dfft)**2
                mh = med_m/(1+mβ1**(1+epoch*chunklim+framenum))
                vh = med_v/(1+mβ2**(1+epoch*chunklim+framenum))
                #med -= 0.1*mh/(vh**.5 + 1e-8)
                #mfft -= mrate*mh/(vh**.5 + 1e-8)
                med -= mrate*mh/(vh**.5 + 1e-8)
                #med = np.fft.ifft2(mfft).real
                #med = mfft
                #"""

            img = seq[framenum]
            params = params_dset[framenum]

            result = forward_model(params, img, kx0, ky0, hc)
            diff = result - med # np.arcsinh(result - med)
            #diff = np.fft.fftshift(np.log(np.abs(np.fft.fft2(diff))))
            #if epoch < 3:
            if True:
                ignore, smin, smax = sigclip_histogram(diff, 5.0, 5.0)
                nmin = min(nmin, smin)
                nmax = max(nmax, smax)

            t1 = time_ns()
            #if epoch > 3:
            if (t1-t0)*1e-9 > 1.0/60:
                #cv2_imshow(cv2.resize(diff, (2048, 2048), interpolation=cv2.INTER_LANCZOS4), title=f'[{framenum:04d}] @ epoch={epoch}, loss={L:.16e}', Min=vmin, Max=vmax)
                cv2_imshow(cv2.resize(diff, (1024,1024), interpolation=cv2.INTER_LANCZOS4), title=f'[{framenum:04d}] @ epoch={epoch}, loss={L:.16e}', Min=vmin, Max=vmax)
                #cv2_imshow(cv2.resize(diff, (1024, 1024), interpolation=cv2.INTER_LANCZOS4), title=f'[{framenum:04d}] @ epoch={epoch}, loss={L:.16e}')
                t0 = time_ns()
            #rolling_mean += result
            #rolling_sum += 1.0
            """
            dfft = np.fft.fft2(diff)
            med_m[:] = med_m*mβ1 + (1-mβ1)*dfft
            med_v[:] = med_v*mβ2 + (1-mβ2)*np.abs(dfft)**2
            mh = med_m/(1+mβ1**(1+epoch*chunklim+framenum))
            vh = med_v/(1+mβ2**(1+epoch*chunklim+framenum))
            #med -= 0.1*mh/(vh**.5 + 1e-8)
            mfft -= mrate*mh/(vh**.5 + 1e-8)
            med = np.fft.ifft2(mfft).real
            #"""

            sector_cache[framenum] = result
        #med = rolling_mean / rolling_sum
        med = np.median(sector_cache, axis=0)
        ###mfft = np.fft.fft2(med)
        ###rfft = np.fft.fft2(result)
        """
        for iterations in range(100):
            η,λ = shrp
            mfft2 = mfft * (1 + η*(2*np.pi)**2*kr**2)/(1 + λ*(2*np.pi)**4*kr**4)
            dL_dη = η*(rfft.conj()*mfft2*(2*np.pi)**2*kr**2/(1 + η*(2*np.pi)**2*kr**2)).mean().real
            dL_dλ = λ*(rfft.conj()*mfft2*(2*np.pi)**4*kr**4/(1 + λ*(2*np.pi)**4*kr**4)).mean().real
            #print(dL_dη, dL_dλ)
            shrp_g = np.array([dL_dη, dL_dλ], dtype=np.float64)
            shrp_m = β1*shrp_m + (1-β1)*shrp_g
            shrp_v = β2*shrp_v + (1-β2)*shrp_g**2
            mhat = shrp_m/(1+β1**(1 + epoch))
            vhat = shrp_v/(1+β2**(1 + epoch))
            dshrp = mhat/(vhat**.5 + 1e-8)
            #shrp -= dshrp
            shrp - np.exp(np.log(shrp) - 0.1*dshrp)
        """
        η,λ = shrp
        #mfft = mfft * (1 + η*(2*np.pi)**2*kr**2)/(1 + λ*(2*np.pi)**4*kr**4)
        #mfft = mfft * airy_disk(kr**2/np.pi**2)**2

        #mfft = mfft * np.exp(0.5*kr**2*(2*np.pi*1.0)**2)
        ###med = np.fft.ifft2(mfft).real
        #med = np.median(sector_cache, axis=0).astype(np.float64)
        #if epoch <= 15:
        #    blur = np.exp(-0.5*(kx**2 + ky**2)*((15-epoch)*max(kx.max(), ky.max())/seq[0].shape[0])**2)
        #if epoch == 2:
        if epoch == 1:
            pass
            #blur = None
            #blur = None
            #epochs_per_frame = 3
        #if epoch < 3:
        if True:
            vmin, vmax = nmin, nmax
            nmin, nmax = 1e30, -1e30
        print(epoch)
        globals().update(locals())

>>> big_iterator(epochs_per_frame=1, total_epochs=100, dset_limit=None, dsize=256, λ=1e-2, η=1e-5, mrate=1e+0)

# Those two are almost all you need.  The following is an earlier kernel fitting test. It works shockingly well! But, only very well if I crank the lanczos resampling to 2048 and kill the ram.

if True:
    cu_code3 = """// ── Affine resample (center-relative, B-spline) ──────────────────────
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
    krnl = cuda_compile(cu_code + cu_code3) # Note: cu_code is everything except that one. It should be present in its current form in clean_fast_6.py. I genuinely can't remember.
    hc = CUDAHcongridContainer()
    hc.initialize_affine(*seq[0].shape)
    res = hc.affine_resample(seq[0], np.eye(2, dtype=np.float64))
    fig, ax = plt.subplots(1,4)
    ax[0].imshow(res)
    θ = np.radians(45/2)
    c, s = np.cos(θ), np.sin(θ)
    M = np.array([[c, -s], [s, c]])
    res = hc.affine_resample(seq[0], M)
    ax[1].imshow(res)
    θ = np.radians(45*2/2)
    c, s = np.cos(θ), np.sin(θ)
    M = np.array([[c, -s], [s, c]])
    res = hc.affine_resample(seq[0], M)
    ax[2].imshow(res)
    θ = np.radians(45*3/2)
    c, s = np.cos(θ), np.sin(θ)
    M = np.array([[c, -s], [s, c]])
    res = hc.affine_resample(seq[0], M)
    ax[3].imshow(res)
    plt.show()

def kill_ram_chunk(crop_width=512):
    px0, py0 = 1426, 1616
    #crop_width = 48//2#512//2
    #crop_width = 512//2
    crop_width = crop_width//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    #idx_crop = (slice(None), slice(None))

    # Load one file
    #"""
    camera_ = int(camera)
    ccd_ = int(ccd)
    #files = get_file_list(camera=1, ccd=4)[:lim] # Filter for camera 1, CCD 4
    files = get_file_list(camera=camera_, ccd=ccd_) # Filter for specified camera and CCD
    data, header = fits.getdata(files[0], header=True)
    data = data.astype(np.float64)[idx_crop]
    sz = data.shape
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
    m = np.zeros(sz, dtype=np.float64) # TEMP CROP TO 64X64
    v = np.zeros(sz, dtype=np.float64)
    last_m_hat = np.zeros(sz, dtype=np.complex128)
    px, py = 1426, 1616

    for ii, (data, header) in enumerate(make_file_generator(files)):
        #crop = data[:sz[0],:sz[1]]
        crop = data[idx_crop]
        sector_data[ii] = crop# - np.median(crop, axis=0)[None,:]  # This median removes the column bias. lets ignore that for now.
        sector_headers.append(header)
        #offset_sector_data(sector_data, ii, data)
        if ii > 0:
            #cv2_imshow(tonemap(sector_data[ii]-sector_data[ii-1], bins=512), title=f'tess s4 c1 ccd4 f{ii}')
            curr = sector_data[ii] - sector_data[ii-1]
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

    hc = CUDAHcongridContainer()
    hc.initialize_affine(*sector_data[0].shape)
    hc.initialize_undistort(sector_headers[0], sector_data[0].shape)#, chunk_offset=(1024,1024))

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
    #cuda_hcongrid_container.initialize_undistort(sector_headers[0], sector_data[0].shape)#, chunk_offset=(1024,1024))

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
    

@numba.njit
def sample_lanczos_numba(img, y_f, x_f, a=3):
    """
    Numba-optimized Lanczos sampler.
    Returns: value, grad_y, grad_x, hess_yy, hess_xx, hess_yx
    """
    y0 = int(np.floor(y_f))
    x0 = int(np.floor(x_f))
    
    # Pre-calculate 1D kernels for the window
    # In Numba, we just use small fixed-size arrays
    Ly = np.zeros(2*a)
    dLy = np.zeros(2*a)
    ddLy = np.zeros(2*a)
    Lx = np.zeros(2*a)
    dLx = np.zeros(2*a)
    ddLx = np.zeros(2*a)
    
    # Fill kernels (reusing your lanczos_ops logic inside)
    for i in range(2*a):
        # Calculate distances dy and dx
        dist_y = y_f - (y0 - a + 1 + i)
        dist_x = x_f - (x0 - a + 1 + i)
        
        # Call a njit-version of your lanczos_ops or inline it:
        Ly[i], dLy[i], ddLy[i] = lanczos_kernel_calc(dist_y, a)
        Lx[i], dLx[i], ddLx[i] = lanczos_kernel_calc(dist_x, a)

    # Accumulators (the "shader" loop)
    val = 0.0
    gy, gx = 0.0, 0.0
    hyy, hxx, hyx = 0.0, 0.0, 0.0
    
    H, W = img.shape
    
    for i in range(2*a):
        py = y0 - a + 1 + i
        # Manual boundary clip
        if py < 0: py = 0
        if py >= H: py = H - 1
            
        for j in range(2*a):
            px = x0 - a + 1 + j
            if px < 0: px = 0
            if px >= W: px = W - 1
            
            pixel = img[py, px]
            
            # Weight components
            w = Ly[i] * Lx[j]
            val += pixel * w
            
            # Derivatives using product rule
            gy += pixel * (dLy[i] * Lx[j])
            gx += pixel * (Ly[i] * dLx[j])
            
            hyy += pixel * (ddLy[i] * Lx[j])
            hxx += pixel * (Ly[i] * ddLx[j])
            hyx += pixel * (dLy[i] * dLx[j])
            
    return val, gy, gx, hyy, hxx, hyx

@numba.njit(inline='always')
def lanczos_scalar(x, a=3):
    """
    Returns L(x), L'(x), L''(x) for scalar x.
    Assumes standard Lanczos support |x| < a.
    """
    ax = abs(x)

    if ax >= a:
        return 0.0, 0.0, 0.0

    if x == 0.0:
        # Center limits
        L = 1.0
        dL = 0.0
        ddL = -(np.pi**2 / 3.0) * (1.0 + 1.0 / (a*a))
        return L, dL, ddL

    px = np.pi * x
    pxa = px / a

    s_px  = np.sin(px)
    c_px  = np.cos(px)
    s_pxa = np.sin(pxa)
    c_pxa = np.cos(pxa)

    px2 = px * px

    # Kernel
    L = (a * s_px * s_pxa) / px2

    # First derivative
    dL = (a*np.pi*c_px*s_pxa + np.pi*s_px*c_pxa - 2.0*a*s_px*s_pxa/x) / px2

    # Your second-derivative formula
    term1 = -a*np.pi*np.pi * s_px * s_pxa
    term2 =  2.0*np.pi*np.pi * c_px * c_pxa
    term3 = -(np.pi*np.pi/a) * s_px * s_pxa
    ddL = (term1 + term2 + term3) / px2 + (2.0*a*s_px*s_pxa)/(np.pi*x*x*x)

    return L, dL, ddL

@numba.njit
def lanczos_ops_1d(v, a=3):
    """
    Vector version for a 1D array only.
    """
    n = v.shape[0]
    L = np.empty(n, dtype=np.float64)
    dL = np.empty(n, dtype=np.float64)
    ddL = np.empty(n, dtype=np.float64)

    for i in range(n):
        L[i], dL[i], ddL[i] = lanczos_scalar(v[i], a)

    return L, dL, ddL

@numba.njit
def sample_lanczos(img, pos, a=3):
    """
    Samples image at float pos=(y, x).
    Returns:
        value,
        gradient [dy, dx],
        Hessian [[dyy, dyx], [dxy, dxx]]
    """
    y_f = pos[0]
    x_f = pos[1]

    y0 = int(np.floor(y_f))
    x0 = int(np.floor(x_f))

    n = 2 * a

    ys = np.empty(n, dtype=np.int64)
    xs = np.empty(n, dtype=np.int64)
    dy = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)

    for i in range(n):
        yi = y0 - a + 1 + i
        xi = x0 - a + 1 + i

        ys[i] = yi
        xs[i] = xi
        dy[i] = y_f - yi
        dx[i] = x_f - xi

    Ly, dLy, ddLy = lanczos_ops_1d(dy, a)
    Lx, dLx, ddLx = lanczos_ops_1d(dx, a)

    # Accumulate directly instead of forming patch
    val = 0.0
    grad_y = 0.0
    grad_x = 0.0
    hess_yy = 0.0
    hess_xx = 0.0
    hess_yx = 0.0

    h = img.shape[0]
    w = img.shape[1]

    for iy in range(n):
        yy = ys[iy]
        if yy < 0:
            yy = 0
        elif yy >= h:
            yy = h - 1

        Ly_i = Ly[iy]
        dLy_i = dLy[iy]
        ddLy_i = ddLy[iy]

        for ix in range(n):
            xx = xs[ix]
            if xx < 0:
                xx = 0
            elif xx >= w:
                xx = w - 1

            p = img[yy, xx]

            Lx_i = Lx[ix]
            dLx_i = dLx[ix]
            ddLx_i = ddLx[ix]

            val     += p * Ly_i   * Lx_i
            grad_y  += p * dLy_i  * Lx_i
            grad_x  += p * Ly_i   * dLx_i
            hess_yy += p * ddLy_i * Lx_i
            hess_xx += p * Ly_i   * ddLx_i
            hess_yx += p * dLy_i  * dLx_i

    grad = np.empty(2, dtype=np.float64)
    grad[0] = grad_y
    grad[1] = grad_x

    hess = np.empty((2, 2), dtype=np.float64)
    hess[0, 0] = hess_yy
    hess[0, 1] = hess_yx
    hess[1, 0] = hess_yx
    hess[1, 1] = hess_xx

    return val, grad, hess

@numba.njit(inline='always')
def lanczos_scalar_gradonly(x, a=3):
    """
    Returns L(x), L'(x), L''(x) for scalar x.
    Assumes standard Lanczos support |x| < a.
    """
    ax = abs(x)

    if ax >= a:
        return 0.0, 0.0, 0.0

    if x == 0.0:
        # Center limits
        L = 1.0
        dL = 0.0
        return L, dL

    px = np.pi * x
    pxa = px / a

    s_px  = np.sin(px)
    c_px  = np.cos(px)
    s_pxa = np.sin(pxa)
    c_pxa = np.cos(pxa)

    px2 = px * px

    # Kernel
    L = (a * s_px * s_pxa) / px2

    # First derivative
    dL = (a*np.pi*c_px*s_pxa + np.pi*s_px*c_pxa - 2.0*a*s_px*s_pxa/x) / px2

    return L, dL

@numba.njit
def lanczos_ops_1d_gradonly(v, a=3):
    """
    Vector version for a 1D array only.
    """
    n = v.shape[0]
    L = np.empty(n, dtype=np.float64)
    dL = np.empty(n, dtype=np.float64)

    for i in range(n):
        L[i], dL[i] = lanczos_scalar_gradonly(v[i], a)

    return L, dL

@numba.njit
def sample_lanczos_gradonly(img, pos, a=3):
    """
    Samples image at float pos=(y, x).
    Returns:
        value,
        gradient [dy, dx],
        Hessian [[dyy, dyx], [dxy, dxx]]
    """
    y_f = pos[0]
    x_f = pos[1]

    y0 = int(np.floor(y_f))
    x0 = int(np.floor(x_f))

    n = 2 * a

    ys = np.empty(n, dtype=np.int64)
    xs = np.empty(n, dtype=np.int64)
    dy = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)

    for i in range(n):
        yi = y0 - a + 1 + i
        xi = x0 - a + 1 + i

        ys[i] = yi
        xs[i] = xi
        dy[i] = y_f - yi
        dx[i] = x_f - xi

    Ly, dLy = lanczos_ops_1d_gradonly(dy, a)
    Lx, dLx = lanczos_ops_1d_gradonly(dx, a)

    # Accumulate directly instead of forming patch
    val = 0.0
    grad_y = 0.0
    grad_x = 0.0
    hess_yy = 0.0
    hess_xx = 0.0
    hess_yx = 0.0

    h = img.shape[0]
    w = img.shape[1]

    for iy in range(n):
        yy = ys[iy]
        if yy < 0:
            yy = 0
        elif yy >= h:
            yy = h - 1

        Ly_i = Ly[iy]
        dLy_i = dLy[iy]
        # ddLy_i = ddLy[iy]

        for ix in range(n):
            xx = xs[ix]
            if xx < 0:
                xx = 0
            elif xx >= w:
                xx = w - 1

            p = img[yy, xx]

            Lx_i = Lx[ix]
            dLx_i = dLx[ix]
            # ddLx_i = ddLx[ix]

            val     += p * Ly_i   * Lx_i
            grad_y  += p * dLy_i  * Lx_i
            grad_x  += p * Ly_i   * dLx_i
            # hess_yy += p * ddLy_i * Lx_i
            # hess_xx += p * Ly_i   * ddLx_i
            # hess_yx += p * dLy_i  * dLx_i

    grad = np.empty(2, dtype=np.float64)
    grad[0] = grad_y
    grad[1] = grad_x

    # hess = np.empty((2, 2), dtype=np.float64)
    # hess[0, 0] = hess_yy
    # hess[0, 1] = hess_yx
    # hess[1, 0] = hess_yx
    # hess[1, 1] = hess_xx

    return val, grad#, hess


@numba.njit(parallel=True)
def aggregate_basins(img):
    Y,X = np.indices(img.shape)
    pts = np.zeros(Y.shape + (3,), dtype=np.float64)
    for y in numba.prange(img.shape[0]):
        for x in numba.prange(img.shape[1]):
            pos = np.array([y,x], dtype=np.float64)
            for n in range(100):
                #v,g,h = sample_lanczos(img, pos)
                v,g = sample_lanczos_gradonly(img, pos)
                #w = np.linalg.eigvalsh(h)
                #pos -= g*0.1#np.linalg.solve(h+np.eye(2)*1e-8,g)
                #pos += np.linalg.solve(h,g)
                pos += g*.2# - np.linalg.solve(h,g)
            pts[y,x] = pos[0], pos[1], v
    return pts

if True:
    pts = aggregate_basins(slog(sector_data[0]-sector_data[0].min()))
    plt.imshow(img, interpolation='lanczos')
    plt.scatter(*pts.reshape((-1,3)).T[:2][::-1], s=1)
    plt.show()


def cull_points_eps(pts, eps):
    """
    pts: (...,3) array of [y, x, value]
    keeps strongest point within each eps-neighborhood
    """
    arr = pts.reshape(-1, 3).copy()

    # strongest first
    order = np.argsort(arr[:, 2])[::-1]
    arr = arr[order]

    inv_cell = 1.0 / eps
    eps2 = eps * eps

    grid = {}   # (cy, cx) -> list of kept indices into 'kept'
    kept = []

    for p in arr:
        y, x, v = p
        cy = int(np.floor(y * inv_cell))
        cx = int(np.floor(x * inv_cell))

        reject = False

        # Check this cell and all 8 neighbors
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                key = (cy + dy, cx + dx)
                if key not in grid:
                    continue
                for j in grid[key]:
                    qy, qx, qv = kept[j]
                    ddy = y - qy
                    ddx = x - qx
                    if ddy*ddy + ddx*ddx < eps2:
                        reject = True
                        break
                if reject:
                    break
            if reject:
                break

        if not reject:
            idx = len(kept)
            kept.append(p)
            key = (cy, cx)
            if key in grid:
                grid[key].append(idx)
            else:
                grid[key] = [idx]

    return np.array(kept, dtype=np.float64)


@numba.njit
def build_phase_vector_fftfreq(n, pos):
    """
    Build exp(-2j*pi*fftfreq(n)*pos) without calling exp for every element.
    Only one sin/cos pair is used for the positive step and one for the negative step.
    """
    out = np.empty(n, dtype=np.complex128)

    out[0] = 1.0 + 0.0j

    if n == 1:
        return out

    # positive frequencies: j / n
    half = n // 2

    theta = -2.0 * np.pi * pos / n
    step_pos = np.cos(theta) + 1j * np.sin(theta)

    z = 1.0 + 0.0j
    for j in range(1, half + 1):
        z *= step_pos
        out[j] = z

    # negative frequencies
    # fftfreq layout:
    # even n: indices half+1..n-1 correspond to -(half-1), ..., -1
    # odd  n: indices half+1..n-1 correspond to -half, ..., -1
    theta_neg = +2.0 * np.pi * pos / n
    step_neg = np.cos(theta_neg) + 1j * np.sin(theta_neg)

    z = 1.0 + 0.0j
    start = half + 1
    for j in range(n - 1, start - 1, -1):
        z *= step_neg
        out[j] = z

    return out

@numba.njit
def precompute_exponentials_recurrence(culled, H, W):
    N = culled.shape[0]

    Ey = np.empty((N, H), dtype=np.complex128)
    Ex = np.empty((N, W), dtype=np.complex128)

    for n in range(N):
        y = culled[n, 0]
        x = culled[n, 1]

        Ey[n, :] = build_phase_vector_fftfreq(H, y)
        Ex[n, :] = build_phase_vector_fftfreq(W, x)

    return Ey, Ex

@numba.njit(parallel=True)
def build_delta_field_recurrence(culled, Ey, Ex):
    N = culled.shape[0]
    H = Ey.shape[1]
    W = Ex.shape[1]

    delta_field = np.empty((H, W), dtype=np.complex128)

    for i in numba.prange(H):
        for j in range(W):
            s = 0.0 + 0.0j
            for n in range(N):
                s += culled[n, 2] * Ey[n, i] * Ex[n, j]
            delta_field[i, j] = s

    return delta_field

@numba.njit
def precompute_exponentials(culled, ky, kx):
    N = culled.shape[0]
    H = ky.shape[0]
    W = kx.shape[0]

    Ey = np.empty((N, H), dtype=np.complex128)
    Ex = np.empty((N, W), dtype=np.complex128)

    for n in range(N):
        y = culled[n, 0]
        x = culled[n, 1]

        for i in range(H):
            Ey[n, i] = np.exp(-2j * np.pi * ky[i] * y)

        for j in range(W):
            Ex[n, j] = np.exp(-2j * np.pi * kx[j] * x)

    return Ey, Ex

@numba.njit(parallel=True)
def build_delta_field(culled, Ey, Ex):
    N = culled.shape[0]
    H = Ey.shape[1]
    W = Ex.shape[1]

    delta_field = np.empty((H, W), dtype=np.complex128)

    for i in numba.prange(H):
        for j in range(W):
            s = 0.0 + 0.0j
            for n in range(N):
                s += culled[n, 2] * Ey[n, i] * Ex[n, j]
            delta_field[i, j] = s

    return delta_field

def compute_krnl(img, shape=(128,128)):
    i0 = img.min()
    pts = aggregate_basins(slog(img-i0))
    culled = cull_points_eps(pts, eps=0.75)
    culled[..., -1] = np.exp(culled[..., -1]) + i0
    bigimg = cv2.resize(img, shape, interpolation=cv2.INTER_LANCZOS4)
    culled[..., 0] *= bigimg.shape[0]/img.shape[0]
    culled[..., 1] *= bigimg.shape[1]/img.shape[1]
    #ky = np.fft.fftfreq(bigimg.shape[0])
    #kx = np.fft.fftfreq(bigimg.shape[1])
    #delta_field = build_delta_field(img, culled, ky,kx)
    #krnl = np.fft.fft2(img)/delta_field
    #delta_field = build_delta_field(bigimg, culled, ky,kx)
    #Ey, Ex = precompute_exponentials(culled, ky, kx)
    #delta_field = build_delta_field(culled, Ey, Ex)
    #krnl = np.fft.fft2(bigimg)/delta_field
    #H, W = bigimg.shape
    #Ey, Ex = precompute_exponentials_recurrence(culled, H, W)
    #delta_field = build_delta_field_recurrence(culled, Ey, Ex)
    kernel = np.fft.fft2(bigimg)/delta_field
    return kernel

def compute_mean_krnl(sector):
    sd = sector.copy()
    img = sd[0]
    shape = img.shape
    kernel = np.zeros(shape, dtype=np.complex128)
    for frame in range(len(sd)):
        kernel += compute_krnl(sd[frame], shape)
        frame1 = frame+1
        if frame1%100 == 0:
            print(frame1)
        kernel_vis = np.fft.fftshift(np.fft.ifft2(kernel/frame1).real)
        cv2_imshow(cv2.resize(kernel_vis, (1024, 1024), interpolation=cv2.INTER_LANCZOS4), title=f'[{frame:04d}]')
        globals().update(locals())
    return kernel/len(sd)

def compute_mean_krnl_faster(sector):
    sd = sector.copy()
    img = sd[0]
    shape = img.shape
    kernel = np.zeros(shape, dtype=np.complex128)
    for frame in range(len(sd)):
        kernel += compute_krnl(sd[frame], shape)
        frame1 = frame+1
        if frame1%100 == 0:
            print(frame1)
        kernel_vis = np.fft.fftshift(np.fft.ifft2(kernel/frame1).real)
        cv2_imshow(cv2.resize(kernel_vis, (1024, 1024), interpolation=cv2.INTER_LANCZOS4), title=f'[{frame:04d}]')
        globals().update(locals())
    return kernel/len(sd)

if True:
    kill_ram_chunk(256)
    kernel = compute_mean_krnl(sector_data)
    
    krnl = cuda_compile(cu_code + cu_code3)
    
    sd = sector_data.copy()
    ky,kx = np.meshgrid(np.fft.fftfreq(len(kernel)), np.fft.fftfreq(len(kernel)), indexing='ij')
    kr = (ky**2+kx**2)**.5
    bsize = 2.0
    kernel_blur = np.exp(-2.0*np.pi**2*bsize**2 * kr**2)
    #sd = np.fft.ifft2(np.fft.fft2(sd)*kernel_blur/kernel).real
    sd = np.fft.ifft2(np.fft.fft2(sd)/kernel).real

# The following were testing I did before, and doesn't work right and I'm bitter about it.

if True:
    from scipy.ndimage import map_coordinates

    def raw_xy(shape):
        return np.indices(shape, dtype=np.float64)

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
        #Y, X = centered_xy(img.shape)
        Y, X = raw_xy(img.shape)

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

    def interp_conj_cpu_with_derivs(F0, qy, qx, order=1, eps_q=1e-6):

        # 5 tap samples to estimate derivatives as finite differences
        # L[y,x] from the loest value (-1,-1)
        Lyx = []
        for dy in [-eps_q, 0, eps_q]:
            Ly = []
            for dx in [-eps_q, 0, eps_q]:
                Ly.append(interp_conj_cpu(F0, qy + dy, qx + dx, order=order))
            Lyx.append(Ly)
        #Lyx = np.array(Lyx)  # shape (3,3)

        # L11 = interp_conj_cpu(F0, qy, qx, order=order)
        # L12 = interp_conj_cpu(F0, qy, qx + eps_q, order=order)
        # L10 = interp_conj_cpu(F0, qy, qx - eps_q, order=order)
        # L01 = interp_conj_cpu(F0, qy + eps_q, qx, order=order)
        # L21 = interp_conj_cpu(F0, qy - eps_q, qx, order=order)

        # Lc = L11

        # Lx = (L12 - L10) / (2 * eps_q)
        # Ly = (L01 - L21) / (2 * eps_q)

        # Do the same for Lxx, Lxy, Lyy, the 2nd derivatives
        # Lxx = (L12 - 2*L11 + L10) / (eps_q**2)

        Lc = Lyx[1][1]
        Lx = (Lyx[1][2] - Lyx[1][0]) / (2 * eps_q)  # Lx = (Lyx[1,2] - Lyx[1,0]) / (2 * eps_q)
        Ly = (Lyx[2][1] - Lyx[0][1]) / (2 * eps_q)  # Ly = (Lyx[2,1] - Lyx[0,1]) / (2 * eps_q)
        Lxx = (Lyx[1][2] - 2*Lyx[1][1] + Lyx[1][0]) / (eps_q**2)
        Lxy = (Lyx[2][2] - Lyx[2][0] - Lyx[0][2] + Lyx[0][0]) / (4 * eps_q**2)
        Lyy = (Lyx[2][1] - 2*Lyx[1][1] + Lyx[0][1]) / (eps_q**2)

        return Lc, Lx, Ly, Lxx, Lxy, Lyy

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

        #im = 2j*np.pi
        #return Lc, im*Lx, im*Ly, im**2*Lxx, im**2*Lxy, im**2*Lyy, kx, ky

        Lc, Lx, Ly, Lxx, Lxy, Lyy = interp_conj_cpu_with_derivs(F0, qy, qx, order=order)
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
        Λ = np.array([
            [a, b],
            [c, d],
        ])
        iΛ = np.array([
            [d, -b],
            [-c, a],
        ]) / det
        #iΛ = np.linalg.inv(Λ)
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

        # Base
        C = (Lc * Z).real.mean() / N

        # Translation first
        Ty = ((-im*ky) * Lc * Z).real.mean() / N
        Tx = ((-im*kx) * Lc * Z).real.mean() / N

        # Affine first
        Ayy = ((ky) * Ly * Z).real.mean() / N
        Ayx = ((ky) * Lx * Z).real.mean() / N
        Axy = ((kx) * Ly * Z).real.mean() / N
        Axx = ((kx) * Lx * Z).real.mean() / N

        # Translation-translation
        Tyy = ((-im*ky)*(-im*ky) * Lc * Z).real.mean() / N
        Tyx = ((-im*ky)*(-im*kx) * Lc * Z).real.mean() / N
        Txx = ((-im*kx)*(-im*kx) * Lc * Z).real.mean() / N

        # Translation-affine
        Ty_Ayy = ((-im*ky)*(ky) * Ly * Z).real.mean() / N
        Ty_Ayx = ((-im*ky)*(ky) * Lx * Z).real.mean() / N
        Ty_Axy = ((-im*ky)*(kx) * Ly * Z).real.mean() / N
        Ty_Axx = ((-im*ky)*(kx) * Lx * Z).real.mean() / N
        Tx_Ayy = ((-im*kx)*(ky) * Ly * Z).real.mean() / N
        Tx_Ayx = ((-im*kx)*(ky) * Lx * Z).real.mean() / N
        Tx_Axy = ((-im*kx)*(kx) * Ly * Z).real.mean() / N
        Tx_Axx = ((-im*kx)*(kx) * Lx * Z).real.mean() / N

        # Affine-affine
        Ayy_Ayy = ((ky)*(ky) * Lyy * Z).real.mean() / N
        Ayy_Ayx = ((ky)*(ky) * Lxy * Z).real.mean() / N
        Ayy_Axy = ((ky)*(kx) * Lyy * Z).real.mean() / N
        Ayy_Axx = ((ky)*(kx) * Lxy * Z).real.mean() / N
        Ayx_Ayx = ((ky)*(ky) * Lxx * Z).real.mean() / N
        Ayx_Axy = ((ky)*(kx) * Lxy * Z).real.mean() / N
        Ayx_Axx = ((ky)*(kx) * Lxx * Z).real.mean() / N
        Axy_Axy = ((kx)*(kx) * Lyy * Z).real.mean() / N
        Axy_Axx = ((kx)*(kx) * Lxy * Z).real.mean() / N
        Axx_Axx = ((kx)*(kx) * Lxx * Z).real.mean() / N

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
        dΔy = -A*det*Ty
        dΔx = -A*det*Tx
        dΛyy = iΛ.T[0,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Ayy
        dΛyx = iΛ.T[0,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Ayx
        dΛxy = iΛ.T[1,0]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Axy
        dΛxx = iΛ.T[1,1]*det*(0.5*A**2*LL + A*B*l0 - A*C) - A*det*Axx
        J = np.array([dA, dB, dΔy, dΔx, dΛyy, dΛyx, dΛxy, dΛxx], dtype=np.float64)

        # Compute the Hessian
        dAdA = det*LL
        dAdB = det*l0
        dAdΔy = -det*Ty
        dAdΔx = -det*Tx
        dAdΛyy = iΛ.T[0,0]*det*(A*LL + B*l0 - C) - det*Ayy
        dAdΛyx = iΛ.T[0,1]*det*(A*LL + B*l0 - C) - det*Ayx
        dAdΛxy = iΛ.T[1,0]*det*(A*LL + B*l0 - C) - det*Axy
        dAdΛxx = iΛ.T[1,1]*det*(A*LL + B*l0 - C) - det*Axx

        dBdB = 1.0
        dBdΔy = 0.0
        dBdΔx = 0.0
        dBdΛyy = iΛ.T[0,0]*det*l0
        dBdΛyx = iΛ.T[0,1]*det*l0
        dBdΛxy = iΛ.T[1,0]*det*l0
        dBdΛxx = iΛ.T[1,1]*det*l0

        dΔydΔy = -A*det*Tyy
        dΔydΔx = -A*det*Tyx
        dΔydΛyy = -A*iΛ.T[0,0]*det*Ty - A*det*Ty_Ayy
        dΔydΛyx = -A*iΛ.T[0,1]*det*Ty - A*det*Ty_Ayx
        dΔydΛxy = -A*iΛ.T[1,0]*det*Ty - A*det*Ty_Axy
        dΔydΛxx = -A*iΛ.T[1,1]*det*Ty - A*det*Ty_Axx

        dΔxdΔx = -A*det*Txx
        dΔxdΛyy = -A*iΛ.T[0,0]*det*Tx - A*det*Tx_Ayy
        dΔxdΛyx = -A*iΛ.T[0,1]*det*Tx - A*det*Tx_Ayx
        dΔxdΛxy = -A*iΛ.T[1,0]*det*Tx - A*det*Tx_Axy
        dΔxdΛxx = -A*iΛ.T[1,1]*det*Tx - A*det*Tx_Axx

        #dΛijdΛyz = det*((iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(A**2*LL/2 + A*B*l0
        # - A*det*(
        # (iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*C
        # + iΛ[j,y]*C_yz
        # + iΛ[z,y]*C_ij
        # + C_yijz
        # )

        Aij = np.array([
            [Ayy, Ayx],
            [Axy, Axx],
        ], dtype=np.float64)

        #         # Affine-affine
        # Ayy_Ayy = ((ky)*(ky) * Lyy * Z).real.mean() / N
        # Ayy_Ayx = ((ky)*(ky) * Lxy * Z).real.mean() / N
        # Ayy_Axy = ((ky)*(kx) * Lyy * Z).real.mean() / N
        # Ayy_Axx = ((ky)*(kx) * Lxy * Z).real.mean() / N
        # Ayx_Ayx = ((ky)*(ky) * Lxx * Z).real.mean() / N
        # Ayx_Axy = ((ky)*(kx) * Lxy * Z).real.mean() / N
        # Ayx_Axx = ((ky)*(kx) * Lxx * Z).real.mean() / N
        # Axy_Axy = ((kx)*(kx) * Lyy * Z).real.mean() / N
        # Axy_Axx = ((kx)*(kx) * Lxy * Z).real.mean() / N
        # Axx_Axx = ((kx)*(kx) * Lxx * Z).real.mean() / N

        # Aij_Ayz = np.array([
        #     Ayy_Ayy, Ayy_Ayx, Ayy_Axy, Ayy_Axx,
        #     Ayx_Ayy, Ayx_Ayx, Ayx_Axy, Ayx_Axx,
        #     Axy_Ayy, Axy_Ayx, Axy_Axy, Axy_Axx,
        #     Axx_Ayy, Axx_Ayx, Axx_Axy, Axx_Axx,
        # ], dtype=np.float64).reshape((2,2,2,2))
        Aij_Ayz = np.array([
            Ayy_Ayy, Ayy_Ayx, Ayy_Axy, Ayy_Axx,
            Ayy_Ayx, Ayx_Ayx, Ayx_Axy, Ayx_Axx,
            Ayy_Axy, Ayx_Axy, Axy_Axy, Axy_Axx,
            Ayy_Axx, Ayx_Axx, Axy_Axx, Axx_Axx,
        ], dtype=np.float64).reshape((2,2,2,2))

        dΛijdΛyz = np.zeros((2,2,2,2), dtype=np.float64)
        for i in range(2):
            for j in range(2):
                for y in range(2):
                    for z in range(2):
                        dΛijdΛyz[i,j,y,z] = det*(iΛ[j,i]*iΛ[z,y] - iΛ[j,y]*iΛ[z,i])*(0.5*A**2*LL + A*B*l0 - A*C)
                        dΛijdΛyz[i,j,y,z] -= A*det*(iΛ[z,y]*Aij[i,j] + iΛ[j,i]*Aij[y,z] + Aij_Ayz[y,i,j,z])

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
        rotated = hc.affine_resample(img, Λ)
        kx0 = np.fft.ifftshift(kx) # Uncenter K for the visualization below
        ky0 = np.fft.ifftshift(ky)
        phasor = np.exp(-2j * np.pi * (ky0 * dy + kx0 * dx))
        result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A + B
        #med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))
        med_thru_fltr = med.copy()
        loss = np.mean((result - med_thru_fltr)**2)/2

        rel_error = (np.float64(L)-np.float64(loss))/np.float64(L)
        print(f"Non-spectral loss: {loss:.6f}, Spectral loss: {L:.6f}")
        print(f"Relative Error: {rel_error:e}, Ratio: {L/loss:.6f}")


    # Approximate using hc.affine_resample and finite differences to validate the gradient and hessian from the spectral method.
    # If the GPU is fast enough we may just use this and avoid the spectral hell.
    def forward_model(params, img, kx_unshifted, ky_unshifted, hc, blur=None):
        A, B, dy, dx, a, b, c, d = params

        Lambda = np.array([[a, b],
                        [c, d]], dtype=np.float64)

        # Keep this convention fixed everywhere.
        Lambda_inv = np.linalg.inv(Lambda)
        warped = hc.affine_resample(img, Lambda_inv)

        phasor = np.exp(-2j * np.pi * (ky_unshifted * dy + kx_unshifted * dx))
        if blur is None:
            shifted = np.fft.ifft2(np.fft.fft2(warped) * phasor).real
        else:
            shifted = np.fft.ifft2(np.fft.fft2(warped) * phasor * blur).real

        return A * shifted + B


    def residual(params, img, med, kx_unshifted, ky_unshifted, hc, blur=None):
        #return forward_model(params, img, kx_unshifted, ky_unshifted, hc, blur=blur) - med
        result = forward_model(params, img, kx_unshifted, ky_unshifted, hc, blur=blur)
        mblur = np.fft.ifft2(np.fft.fft2(med)*blur).real if blur is not None else med
        return result - mblur


    def loss_from_residual(r):
        return 0.5 * np.mean(r * r)


    def finite_diff_residual_jacobian(params, img, med, kx_unshifted, ky_unshifted, hc, eps, blur=None):
        """
        eps: array-like of shape (8,), parameter-wise perturbations
        """
        params = np.asarray(params, dtype=np.float64)
        eps = np.asarray(eps, dtype=np.float64)

        r0 = residual(params, img, med, kx_unshifted, ky_unshifted, hc, blur=blur)
        n_params = params.size
        J = np.empty((r0.size, n_params), dtype=np.float64)

        for i in range(n_params):
            dp = np.zeros_like(params)
            dp[i] = eps[i]

            rp = residual(params + dp, img, med, kx_unshifted, ky_unshifted, hc, blur=blur)
            rm = residual(params - dp, img, med, kx_unshifted, ky_unshifted, hc, blur=blur)

            J[:, i] = ((rp - rm) / (2.0 * eps[i])).ravel()

        return r0, J


    def direct_loss_grad_hess(params, img, med, kx_unshifted, ky_unshifted, hc, eps, blur=None):
        r0, Jr = finite_diff_residual_jacobian(
            params, img, med, kx_unshifted, ky_unshifted, hc, eps, blur=blur
        )

        r = r0.ravel()
        N = r.size

        L = 0.5 * np.dot(r, r) / N
        g = (Jr.T @ r) / N
        H_gn = (Jr.T @ Jr) / N   # Gauss–Newton Hessian approximation

        return L, g, H_gn


    def test_affine_spectral_fields_and_grad_hess_optimizer(framenum=75, epochs=10, rate=1.0, LMDamping=1e-12):
        # Identity-ish parameters
        params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
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
        kx0 = np.fft.ifftshift(kx) # Uncenter K for the visualization below
        ky0 = np.fft.ifftshift(ky)

        pm = np.zeros_like(params)  # momentum for Adam
        pv = np.zeros_like(params)   # velocity for Adam
        hm = np.zeros_like(params)
        hv = np.zeros_like(params)
        β1, β2 = 0.9, 0.999
        print(params)

        for epoch in range(epochs):

            # L, J, H = affine_loss_grad_hess(
            #     params, Mhat, m0, f0, F0, Fx, Fy, Fxx, Fxy, Fyy, kx, ky
            # )
            L, J, H = direct_loss_grad_hess(
                params, img, med, kx0, ky0, hc, eps=np.array([1e-6]*8)
            )

            # Whatever we're doing, its unhappy with the affine terms. I suspect that's where the problem lies. Lets force the gradient on their terms to always be zero.
            #J[4:] = 0.0
            #H[4:, :] = 0.0
            #H[:, 4:] = 0.0
            #np.fill_diagonal(H, 1.0)  # Identity Hessian on affine terms to prevent singularity
            # Lets generalize this for arbitrary coordinates we want to zero out.
            #ignore = [4,5,6,7]  # affine parameters
            # ignore = [5,6] # off-diagonal affine parameters b,c which are near zero and causing instability
            # J[ignore] = 0.0
            # H[ignore, :] = 0.0
            # H[:, ignore] = 0.0
            # for i in ignore:
            #     H[i, i] = 1.0

            #Gauss newton step
            try:
               # Lavenberg-Marquardt style damping to improve stability: add a small multiple of the identity to the Hessian.
               H = H + np.outer(J,J)*LMDamping

               #step = -0.5*np.linalg.solve(H, J)
               step = -np.linalg.solve(H, J)
            except np.linalg.LinAlgError:
               print("Hessian is singular, cannot take Newton step.")
               break
            params += step*rate

            # Lets try a gradient rescale based on the diagonal of H.
            #grad_rescale = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-30))
            #params -= rate * grad_rescale * J

            # # It REALLY didn't like that. Lets try adam.
            # pm[:] = β1 * pm + (1 - β1) * J
            # pv[:] = β2 * pv + (1 - β2) * (J ** 2)

            # hm[:] = pm / (1 - β1 ** (epoch + 1))
            # hv[:] = pv / (1 - β2 ** (epoch + 1))

            # grad = hm / (np.sqrt(hv) + 1e-8)
            # params -= rate * grad

            # def eval_loss(params): # Defined above
            # A, B, dy, dx, a, b, c, d = params
            # Λ = np.array([
            #     [a, b],
            #     [c, d]
            # ], dtype=np.float64)
            # iΛ = np.linalg.inv(Λ)
            # rotated = hc.affine_resample(img, Λ)

            # phasor = np.exp(-2j * np.pi * (ky0 * dy + kx0 * dx))
            # result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A + B
            # med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))
            # loss = np.mean((result - med_thru_fltr)**2)/2

            # # rewrite this via the forward model
            result = forward_model(params, img, kx0, ky0, hc)

            cv2_imshow(cv2.resize(result - med, (512, 512), interpolation=cv2.INTER_LANCZOS4),
                #title=f'[{framenum:04d}] loss={loss:.2e} @ epoch={epoch}')
                title=f'[{framenum:04d}] @ epoch={epoch}, loss={L:.16e}')
        print(params)

        globals().update(locals())

    def test_full_affine_spectral_fields_and_grad_hess_optimizer(epochs=10, rate=1.0, LMDamping=1e-12):
        # Identity-ish parameters
        #params = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        # Lets select the existing affine parameter vector.

        seq = sector_data[:32]
        
        params_dset = np.zeros((len(seq), 8), dtype=np.float64)
        #params_dset[:len(delta_coords_affine)] = delta_coords_affine[:]
        #params_dset[len(delta_coords_affine):] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float64)
        params_dset[:,0] = 1.0 # Initialize to A=1, and an identity affine
        params_dset[:,4] = 1.0
        params_dset[:,7] = 1.0

        #img = np.asarray(sector_data[framenum], dtype=np.float64)
        #med = np.asarray(sector_median, dtype=np.float64)
        med = np.mean(seq, axis=0)

        # Centered frequency grids for fftshifted arrays
        ky, kx = centered_k(seq[0].shape)
        kx0 = np.fft.ifftshift(kx) # Uncenter K for the visualization below
        ky0 = np.fft.ifftshift(ky)

        blur_size = 2.0 #pixels
        blur = np.exp(-0.5*(kx**2 + ky**2)*(blur_size*max(kx.max(), ky.max())/seq[0].shape[0])**2)  # Gaussian blur in frequency domain to improve stability

        sector_cache = np.empty((len(seq),) + sector_data[0].shape, dtype=np.float64)
        vlow, vhigh = None,None
        vlow_next, vhigh_next = 1e30, -1e30

        for epoch in range(epochs):

            # For each epoch, lets iterate *one* step for each image, recompute mean, and move on.
            for framenum in range(len(seq)):
                img = seq[framenum]
                for iters in range(2):
                    params = params_dset[framenum]

                    L, J, H = direct_loss_grad_hess(
                        params, img, med, kx0, ky0, hc, eps=np.array([1e-6]*8), blur=blur
                    )

                    #Gauss newton step
                    try:
                        # Lavenberg-Marquardt style damping to improve stability: add a small multiple of the identity to the Hessian.
                        H = H + np.eye(len(J))*LMDamping

                        step = np.linalg.solve(H, J)
                    except np.linalg.LinAlgError:
                        print("Hessian is singular, cannot take Newton step.")
                        break
                    params_dset[framenum] -= step*rate

                # Lets try a gradient rescale based on the diagonal of H.
                #grad_rescale = 1.0 / np.sqrt(np.maximum(np.diag(H), 1e-30))
                #params -= rate * grad_rescale * J

                # # It REALLY didn't like that. Lets try adam.
                # pm[:] = β1 * pm + (1 - β1) * J
                # pv[:] = β2 * pv + (1 - β2) * (J ** 2)

                # hm[:] = pm / (1 - β1 ** (epoch + 1))
                # hv[:] = pv / (1 - β2 ** (epoch + 1))

                # grad = hm / (np.sqrt(hv) + 1e-8)
                # params -= rate * grad

                # def eval_loss(params): # Defined above
                # A, B, dy, dx, a, b, c, d = params
                # Λ = np.array([
                #     [a, b],
                #     [c, d]
                # ], dtype=np.float64)
                # iΛ = np.linalg.inv(Λ)
                # rotated = hc.affine_resample(img, Λ)

                # phasor = np.exp(-2j * np.pi * (ky0 * dy + kx0 * dx))
                # result = np.fft.ifft2(np.fft.fft2(rotated) * phasor).real * A + B
                # med_thru_fltr = hc.affine_resample(med, np.identity(2, dtype=np.float64))
                # loss = np.mean((result - med_thru_fltr)**2)/2

                # # rewrite this via the forward model
                result = forward_model(params, img, kx0, ky0, hc)
                sector_cache[framenum] = result

                _, clow, chigh = sigclip_histogram(result - med, 5.0, 5.0)
                vlow_next = min(vlow_next, clow)
                vhigh_next = max(vhigh_next, chigh)

                cv2_imshow(cv2.resize(result - med, (512, 512), interpolation=cv2.INTER_LANCZOS4),
                    Min=vlow, Max=vhigh,
                    #title=f'[{framenum:04d}] loss={loss:.2e} @ epoch={epoch}')
                    title=f'[{framenum:04d}] @ epoch={epoch}, loss={L:.16e}')

            # Epoch finished, recompute med
            #if epoch == 0:
            med = np.mean(sector_cache, axis=0)
            vlow, vhigh = vlow_next, vhigh_next
            vlow_next, vhigh_next = 1e30, -1e30
            #else:
            #    med = np.median(sector_cache, axis=0)
            globals().update(locals())
        print(params)

        globals().update(locals())

    # validate_affine_spectral_fields_and_grad_hess()
    # # Lets get numpy to print as wide as possible instead of doing "word wrap"
    # with np.printoptions(linewidth=2000):
    #     print(f"      {Ln = }")
    #     print(f"       {J = }")
    #     print(f"{(Ln-J)/J = }")

    #test_affine_spectral_fields_and_grad_hess_optimizer(epochs=100, rate=1e-3, LMDamping=1e-20)
    test_full_affine_spectral_fields_and_grad_hess_optimizer(epochs=100, rate=1.0, LMDamping=1e-20)








def kill_ram_chunk(crop_width=512):
    px0, py0 = 1426, 1616
    #crop_width = 48//2#512//2
    #crop_width = 512//2
    crop_width = crop_width//2
    idx_crop = (slice(py0-crop_width,py0+crop_width), slice(px0-crop_width,px0+crop_width))
    #idx_crop = (slice(None), slice(None))

    # Load one file
    #"""
    camera_ = int(camera)
    ccd_ = int(ccd)
    #files = get_file_list(camera=1, ccd=4)[:lim] # Filter for camera 1, CCD 4
    files = get_file_list(camera=camera_, ccd=ccd_) # Filter for specified camera and CCD
    data, header = fits.getdata(files[0], header=True)
    data = data.astype(np.float64)[idx_crop]
    sz = data.shape
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
    m = np.zeros(sz, dtype=np.float64) # TEMP CROP TO 64X64
    v = np.zeros(sz, dtype=np.float64)
    last_m_hat = np.zeros(sz, dtype=np.complex128)
    px, py = 1426, 1616

    for ii, (data, header) in enumerate(make_file_generator(files)):
        #crop = data[:sz[0],:sz[1]]
        crop = data[idx_crop]
        sector_data[ii] = crop# - np.median(crop, axis=0)[None,:]  # This median removes the column bias. lets ignore that for now.
        sector_headers.append(header)
        #offset_sector_data(sector_data, ii, data)
        if ii > 0:
            #cv2_imshow(tonemap(sector_data[ii]-sector_data[ii-1], bins=512), title=f'tess s4 c1 ccd4 f{ii}')
            curr = sector_data[ii] - sector_data[ii-1]
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

    hc = CUDAHcongridContainer()
    hc.initialize_affine(*sector_data[0].shape)
    hc.initialize_undistort(sector_headers[0], sector_data[0].shape)#, chunk_offset=(1024,1024))

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
    #cuda_hcongrid_container.initialize_undistort(sector_headers[0], sector_data[0].shape)#, chunk_offset=(1024,1024))

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
    
# I need to make a quick adaptation of the above function that just gets *one* picture at full resolution. Lets load this as ffi.
def load_one_full_res_image(camera, ccd, framenum):
    files = get_file_list(camera=camera, ccd=ccd)
    file = files[framenum]
    ffi, header = fits.getdata(file, header=True)
    ffi = ffi.astype(np.float64)
    print(f"{ffi.shape = }, {ffi.dtype = }")
    return ffi # we already have the header

ffi = load_one_full_res_image(camera=1, ccd=1, framenum=75)
