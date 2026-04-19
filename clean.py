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
rawdir = ROOT / "DIA_TEMP" / "raw"
#rawdir = ROOT / "TESS_sector_4"
clndir = ROOT / "DIA_TEMP" / "clean2"
clndir_control = ROOT / "DIA_TEMP" / "clean"

# ensure the output directories exist
ROOT = Path(ROOT)#.mkdir(parents=True, exist_ok=True)
cdedir = Path(cdedir)#.mkdir(parents=True, exist_ok=True)
rawdir = Path(rawdir)#.mkdir(parents=True, exist_ok=True)
clndir = Path(clndir)#.mkdir(parents=True, exist_ok=True)
clndir_control = Path(clndir_control)

ROOT.mkdir(parents=True, exist_ok=True)
cdedir.mkdir(parents=True, exist_ok=True)
rawdir.mkdir(parents=True, exist_ok=True)
clndir.mkdir(parents=True, exist_ok=True)

#sample every how many pixels? usually 32x32 is OK but it can be larger or smaller
pix = 32 # UPDATE HERE FOR BACKGROUND SPACING
axs = 2048 # UPDATE HERE FOR IMAGE AXIS SIZE
###END UPDATE INFORMATION###

#get the image list and the number of files which need reduction
#os.chdir(rawdir) #changes to the raw image direcotory
files = [f for f in rawdir.glob("*.fits") if isfile(join(rawdir, f))] #gets the relevant files with the proper extension
files.sort()
nfiles = len(files)
#os.chdir(cdedir) #changes back to the code directory

# Cull files that don't match the specified camera and CCD
camera, ccd = '1', '4'
def filter_file(f):
	img_data, img_header = fits.getdata(f, header=True)
	return (img_header['CAMERA'] == camera) and (img_header['CCD'] == ccd)
files = list(filter(filter_file, files))

#get the zeroth image for registration
#read in the image
ref, rhead = fits.getdata(os.path.join(rawdir, files[0]), header = True)
rhead['CRPIX1'] = 1001.
rhead['NAXIS1'] = 2048
rhead['NAXIS2'] = 2048

#sample every how many pixels?
bxs = 512 #how big do you want to make the boxes for each image?
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

#begin cleaning
for ii in range(0, nfiles):
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
	if not clndir.exists():
		clndir.mkdir(parents=True, exist_ok=True)

	#only create the files that don't exist
	#if not outpath.exists():
	if True: # DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG DEBUG
		#start the watch
		st = time.time()
		sts = time.strftime("%c")
		print(f'Now cleaning {files[ii]} at {sts}.')

		#read in the image
		orgimg, header = fits.getdata(files[ii], header = True)
		w = WCS(header)
		cut = Cutout2D(orgimg, (1068,1024), (axs, axs), wcs = w)
		bigimg = cut.data

		#update the header
		header['CRPIX1'] = 1001.
		header['NAXIS1'] = 2048
		header['NAXIS2'] = 2048

		#get the holders ready
		res = numpy.zeros(shape=(axs, axs), dtype=float) #holder for the background 'image'
		nboxes = int((axs//bxs)**2)
		bck = numpy.zeros(shape=(nboxes,), dtype=float) #get the holder for the image backgroudn
		sbk = numpy.zeros(shape=(nboxes,), dtype=float) #get the holder for the sigma of the image background

		#remove the flat and the bias
		if (biassub == 1) and (flatdiv == 1):
			bigimg = bigimg - bias #subtract the bias
			bigimg = bigimg/flat #subtract the flat
		
		tsc_0 = time.time()
		tts = 0
		for oo in range(0, axs, bxs):
			for ee in range(0, axs, bxs):
				img = bigimg[ee:ee+bxs, oo:oo+bxs] #split the image into small subsections
				
				#calculate the sky statistics
				cimg, clow, chigh = scipy.stats.sigmaclip(img, low=2.5, high = 2.5) #do a 2.5 sigma clipping
				sky = numpy.median(cimg) #determine the sky value
				sig = numpy.std(cimg) #determine the sigma(sky)

				bck[tts] = sky #insert the image median background
				sbk[tts] = sig #insert the image sigma background
				
		tsc_1 = time.time()
		print(f'Sigma Clip for {files[ii]} finished in {tsc_1 - tsc_0}s.')
		
		tts = 0
		for oo in range(0, axs, bxs):
			for ee in range(0, axs, bxs):
				img = bigimg[ee:ee+bxs, oo:oo+bxs] #split the image into small subsections
				
				#create holder arrays for good and bad pixels
				x = numpy.zeros(shape=(int(sze),), dtype=float)
				y = numpy.zeros(shape=(int(sze),), dtype=float)
				v = numpy.zeros(shape=(int(sze),), dtype=float)
				s = numpy.zeros(shape=(int(sze),), dtype=float)
				nd = 0
	
				#begin the sampling of the "local" sky value
				for jj in range(0, bxs+pix, pix):
					for kk in range(0,bxs+pix, pix):
						il = numpy.amax([jj-lop,0])
						ih = numpy.amin([jj+lop, bxs-1])
						jl = numpy.amax([kk-lop, 0])
						jh = numpy.amin([kk+lop, bxs-1])
						c = img[jl:jh, il:ih]
						#select the median value with clipping
						cc, cclow, cchigh = scipy.stats.sigmaclip(c, low=2.5, high = 2.5) #sigma clip the background
						lsky = numpy.median(cc) #the sky background
						ssky = numpy.std(cc) #sigma of the sky background
						x[nd] = numpy.amin([jj, bxs-1]) #determine the pixel to input
						y[nd] = numpy.amin([kk, bxs-1]) #determine the pixel to input
						v[nd] = lsky #median sky
						s[nd] = ssky #sigma sky
						nd = nd + 1

				#now we want to remove any possible values which have bad sky values
				rj = numpy.where(v <= 0) #stuff to remove
				kp = numpy.where(v > 0) #stuff to keep

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
						rd = math.sqrt((xgood-ygood)**2.+(ygood-ybad)**2.)
						#sort the radii
						pp = sorted(range(len(rd)), key = lambda k:rd[k])
						#use the closest 10 points to get a median
						vnear = vgood[pp[0:9]]
						ave = numpy.median(vnear)
						#insert the good value into the array
						v[idx] = ave

				#now we want to remove any possible values which have bad sigmas
				rjs = numpy.where(s >= 2*sig)
				rj  = rjs[0]
				kps = numpy.where(s < 2*sig)
				kp  = kps[0]

				if (len(rj) > 0):
					#keep only the good points
					xgood = numpy.array(x[kp])
					ygood = numpy.array(y[kp])
					vgood = numpy.array(v[kp])
					sgood = numpy.array(s[kp])

					for jj in range(0, len(rj)):
						#select the bad point
						idx = int(rj[jj])
						xbad = x[idx]
						ybad = y[idx]
						#print xbad, ybad
						#use the distance formula to get the closest points
						rd = numpy.sqrt((xgood-xbad)**2.+(ygood-ybad)**2.)
						#sort the radii
						pp = sorted(range(len(rd)), key = lambda k:rd[k])
						#use the closest 10 points to get a median
						vnear = vgood[pp[0:9]]
						ave = numpy.median(vnear)
						#insert the good value into the array
						v[idx] = ave

				#now we interpolate to the rest of the image with a thin-plate spline	
				xi = numpy.linspace(0, bxs-1, bxs)
				yi = numpy.linspace(0, bxs-1, bxs)
				XI, YI = numpy.meshgrid(xi, yi)
				rbf = Rbf(x, y, v, function = 'thin-plate', smooth = 0.0)
				reshld = rbf(XI, YI)
			
				#now add the values to the residual image
				res[ee:ee+bxs, oo:oo+bxs] = reshld
				tts = tts+1

		#get the median background
		mbck = numpy.median(bck)
		sbck = numpy.median(sbk)
	
		#subtract the sky gradient and add back the median background
		sub = bigimg-res
		sub = sub + mbck

		#align the image
		# NOTE: `hcongrid` was originally used for alignment. If available,
		# uncomment the import at the top and ensure the function is on PATH.
		algn = hcongrid(sub, header, rhead)

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
		shd = fits.PrimaryHDU(algn, header=header)
		shd.writeto(outpath, overwrite=True)

		#stop the watch
		fn = time.time()
		print(f'Background subtraction for {files[ii]} finished in {fn-st}s.')
		
		# clndir_control
		# Find the max abs error between the file we just wrote and the control file.
		# files = [f for f in rawdir.glob("*.fits") if isfile(join(rawdir, f))]
		# orgimg, header = fits.getdata(files[ii], header = True)
		
		# Load the cleaned image we just wrote
		cleaned_img, cleaned_header = fits.getdata(outpath, header=True)
		
		# Load the control image
		control_path = clndir_control / finnme
		control_img, control_header = fits.getdata(control_path, header=True)
		
		print(f"Max abs error vs control image: {numpy.max(numpy.abs(cleaned_img - control_img)) = }")




print('All done! See ya later alliagtor.')
