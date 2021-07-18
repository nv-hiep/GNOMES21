import sys, os
import numpy             as np
import matplotlib.pyplot as plt



'''
Fit the on-/off-source spectra using multiple Gaussian components
(for both CNM and WNM).
Please refer to Heiles & Troland 2003a for the methodology
'''


## Get the index of a given velocity #
 #
 # params list v-axis List of Velocity_axis
 # params float vel Velocity
 #
 # return int idx Index of vel in List of velocity_axis
 # 
 # Author Van Hiep ##
def get_vel_index(v_axis, vel):
  idx = (np.abs(np.array(v_axis)-vel)).argmin()
  return idx

## Get Vrange Indexes ##
 #
 # params 1-D-array v     VLSR
 # params float     lowv  Lower limit
 # params float     upv   Upper limit
 #
 # return list
 #
 # version 01/2017 
 # author Nguyen Van Hiep ##
def get_vrange_id(v, lowv, upv):
  vmax = get_vel_index(v, lowv)
  vmin = get_vel_index(v, upv)
  return [vmin, vmax]



##
 # Calculate the T_exp from the contributions of both CNM and WNM components
 #
 # Params:
 # xdata : VLSR (km/s)
 # zrocnm: baseline for on-source spectrum: zrocnm + sum(Gaussians)
 # hgtcnm: Amplitudes of CNM components
 # cencnm: Central VLSR of CNM components
 # widcnm: VLSR width of CNM components
 # tspincnm: Tspin of CNM components
 # ordercnm: Order of CNM components
 #
	 # continuum: baseline for off-source spectrum: continuum + sum(Gaussians)
 # hgtwnm: Amplitudes of WNM components
 # cenwnm: Central VLSR of WNM components
 # widwnm: VLSR width of WNM components
 # fwnm: F-value of WNM components (see HT03a for details)
 #
 # Return:
 # tb_cont: Continuum temperature [K], see below
 # tb_wnm_tot: The contribution of WNM components to the total T_exp [K]
 # tb_cnm_tot: The contribution of CNM components to the total T_exp [K]
 # tb_tot:     the total T_exp [K]
 # exp_tausum: e^(-tau)
 #
 # version 08/2016 
 # author Nguyen Van Hiep ##
def tb_exp(xdata,
	       zrocnm, hgtcnm, cencnm, widcnm, tspincnm, ordercnm,
	       continuum, hgtwnm, cenwnm, widwnm, fwnm):
	
	#ZEROTH STEP IS TO REARRANGE CLOUDS IN ORDER OF 'ORDER'.
	# zro1 = zrocnm
	# hgt1 = hgtcnm[ordercnm]  # od = [0,1,2,3],  q1 = a[od], q1 = [0,2,3,4]
	# cen1 = cencnm[ordercnm]
	# wid1 = widcnm[ordercnm]
	# tspin1 = tspincnm[ordercnm]

	zro1   = zrocnm
	hgt1   = hgtcnm[ordercnm]
	cen1   = cencnm[ordercnm]
	wid1   = widcnm[ordercnm]
	tspin1 = tspincnm[ordercnm]

	#FIRST STEP IS TO CALCULATE THE OPACITY OF EACH COLD CLOUD...
	nchnl  = len(xdata)
	nrcnm  = len(hgt1)
	taucnm = np.zeros( (nrcnm, nchnl), dtype='float64' )

	for nrc in range(nrcnm):
		tau1nrc = gcurv(xdata, zro1, hgt1[nrc], cen1[nrc], wid1[nrc])
		taucnm[nrc, :] = tau1nrc


	tausum = np.squeeze(taucnm) if nrcnm == 1 else taucnm.sum(0)

	exp_tausum = np.exp(-tausum)
	# exp_tausum = exp_tausum.reshape(nchnl,)

	##********** CALCULATE THE WNM CONTRIBUTION ********************
	##  EXPRESS THE WNM CONTRIBUTION AS A SUM OF GAUSSIANS:
	##	FWNM, ZROWNM, HGTWNM, CENWNM, WIDWNM
	tb_cont    = continuum * exp_tausum

	tb_wnm_tot = np.zeros(nchnl, dtype='float64')
	nrwnm      = len(hgtwnm)
	for nrw in range(nrwnm):
		tb_wnm_nrw = gcurv(xdata, 0., hgtwnm[nrw], cenwnm[nrw], widwnm[nrw])
		tb_wnm_tot = tb_wnm_tot + tb_wnm_nrw*(fwnm[nrw] + (1.-fwnm[nrw])*exp_tausum)

	#*************** NEXT CALCULATE THE CNM CONTRIBUTION ****************

	tb_cnm_tot = np.zeros(nchnl, dtype='float64')

	# BRIGHT TEMP OF EACH CNM CLUMP:
	tbclump = np.zeros((nrcnm, nchnl), dtype='float64')
	for nrc in range(nrcnm):
		tbclump[nrc, :] = tspin1[nrc] * (1. - np.exp(-taucnm[nrc, :]))

	## Cho nay dung' ro`i, cong lai ca 1->m roi Tru` di tau_m. Lam` the' nay` de? tranh' di "[]" luc' nrc=0
	for nrc in range(nrcnm):
		temp        = np.reshape(taucnm[0:nrc+1, :], (nrc+1, nchnl))
		tausum_nrc  = temp.sum(0)
		exp_tau_nrc = np.exp(taucnm[nrc, :] - tausum_nrc)
		tb_cnm_tot  = tb_cnm_tot + tspin1[nrc] * (1. - np.exp(-taucnm[nrc, :]) ) * exp_tau_nrc
	## Endfor: nrc

	tb_tot = tb_cont + tb_cnm_tot + tb_wnm_tot

	return tb_cont, tb_wnm_tot, tb_cnm_tot, tb_tot, exp_tausum
	




## Multiple (N) Gaussians + offset. ##
 #
 # params list  v    VLSR
 # params float zr   estimated constant zero offset of the data points.
 # params list  h    the array of N estimated heights of the Gaussians.
 # params list  v0   the array of N estimated centers of the Gaussians.
 # params list  w    the array of N estimated halfwidths of the Gaussians.
 #
 # return 1-D-array  tf  The calculated points.
 #
 # version 01/2017 
 # author Nguyen Van Hiep ##
def gcurv(v, zr, h, v0, w):
	'''
	FWHM = 2 * sqrt(2*ln2) * sigma
	-> sigma = FWHM / ( 2 * sqrt(2*ln2) )

	Gaussian function: A * exp{ - [ (x-x0) / (sqrt(2) * sigma) ]**2 }
	                 = A * exp{ - [ (x-x0) / (FWHM / (2*sqrt(ln2)) ) ]**2 }
	                 = A * exp{ - [ (x-x0) / (FWHM / (2*sqrt(ln2)) ) ]**2 }
	                 = A * exp{ - [ (x-x0) / (0.60056120439 * FWHM ]**2 }
	'''
	dp600 = np.float64(0.60056120) # For converting between FWHM and sigma
	
	if np.isscalar(v):
		v  = np.array([v], dtype='float64')
	
	if np.isscalar(h):
		h  = np.array([h], dtype='float64')
	
	if np.isscalar(v0):
		v0 = np.array([v0], dtype='float64')
	
	if np.isscalar(w):
		w  = np.array([w], dtype='float64')

	# DETERMINE NUMBER OF GAUSSIANS...
	ng = len(h)
	
	tf = 0.*v + zr
	for i in range(ng):
		if (w[i] > 0.):
			tf = tf + h[i]*np.exp(- ( (v-v0[i])/(dp600*w[i]))**2)

	return tf



##  Fit the Expected Emission Profile ##
 #
 #+
 #NAME:
 #   GFITFLEX_EXP
 #
 #PURPOSE:
 #    Fit multiple (N) exp(-sum of Gaussians) to a one-d array of data points, 
 #	keeping any arbitrary set of parameters fixed and not included
 #	in the fit.
 #
 #CALLING SEQUENCE:
 #    GFITFLEX_EXP, look, xdata, tdata, zro0, hgt0, cen0, wid0, $
 #	zro0yn, hgt0yn, cen0yn, wid0yn, $
 #	tfit, sigma, zro1, hgt1, cen1, wid1, $
 #	sigzro1, sighgt1, sigcen1, sigwid1, cov
 #
 #INPUTS:
 #     look: if >=0, plots the iteratited values for the Gaussian
 #     whose number is equal to look. Then it prompts you to plot 
 #     a different Gaussian number.
 #
 #     xdata: the x-values at which the data points exist.
 #     tdata: the data points.
 #     xrange: 2n-element vector: 2 values for each of n index ranges
 #	specifying which indices of tdata to include in the fit.
 #
 #     zro0: the estimated constant zero offset of the data points.
 #     hgt0: the array of N estimated heights of the Gaussians.
 #     cen0: the array of N estimated centers of the Gaussians.
 #     wid0: the array of N estimated halfwidths of the Gaussians.
 #
 #     zr0yn: if 0, does not fit zero level # if 1, it does.
 #     hgt0yn: array of N 0 or 1 # 0 does not fit the hgt, 1 does.
 #     cen0yn: array of N 0 or 1 # 0 does not fit the hgt, 1 does.
 #     wid0yn: array of N 0 or 1 # 0 does not fit the hgt, 1 does.
 #
 #OUTPUTS:
 #     tfita: the fitted values of the data at the points in xdata.
 #     sigma: the rms of the residuals.
 #     zro1: the fitted zero offset (held constant if zro0yn=0).
 #     hgt1: the array of N fitted heights. 
 #     cen1: the array of N fitted centers.
 #     wid1: the array of N fitted half-power widths.
 #     sigzro1: the error of the fitted zero offset # zero if zr0yn=0.
 #     sighgt1: the array of errors of the N fitted heights # zero if hgt0yn=0).
 #     sigcen1: the array of errors of the N fitted centers # zero if cen0yn=0).
 #     sigwid1: the array of errors of the N fitted widths # zero if wid0yn=0).
 #     problem: 0, OK # -1, excessive width # -2, >50 loops # -3, negative sigmas, 4, bad derived values.
 #     cov: the normalized covariance matrix of the fitted coefficients.
 #
 #RESTRICTIONS:
 #    The data and x values should be in asympototic x order, 
 #    either increasing or decreasing.
 #    Gaussians are not an orthogonal set of functions! 
 #    This doesn't matter for many cases # convergence is unique UNLESS...
 #    Convergence is NOT unique when Gaussians are close together or when
 #    multiple Gaussians lie within a single peak. In these cases, you
 #    can get different outputs from different inputs.
 #    And sometimes in these cases the fits will not converge!
 #
 #    This procedure uses the classical nonlinear least squares technique,
 #    which utilizes analytically-calculated derivatives, to iteratively
 #    solve for the least squares coefficients. Some criteria on the
 #    parameters used to update the iterated coefficients are used to
 #    make the fit more stable (and more time-consuming). The number
 #    of iterations is limited to 50 # if you need more, enter the routing
 #    again, using the output parameters as input for the next attampt.
 #
 #EXAMPLE:
 #    You have two Gaussians that are well-separated. This counts as an
 #	easy case # for the estimated parameters, you need not be accurate
 #	at all. The heights are hgt0=[1.5, 2.5], the centers cen0=[12., 20.],
 #	and the widths are [5., 6.]. You wish to hold the width of the second
 #    Gaussian fixed in the fit, so you set wid0yn=[1,0]. 
 #    There are 100 data points (tdata) at 
 #	100 values of x (xdata) and you want to fit indices 25-75 and
 #	77-80 only, so
 #	you set xrange=[50,75,77,80]. 
 #    You don't wish to see plots of the iterations,
 #	you don't care about the uncertainties, but you want the fitted
 #	 points and also the rms of the residuals.
 #
 #    If you have two Gaussians that are mixed, you must be careful in
 #    your estimates!
 #
 #RELATED PROCEDURES:
 #	GCURV
 #HISTORY:
 #	Original GFIT Written by Carl Heiles. 21 Mar 1998.
 #	FLEX options added 4 feb 00.
 #-
 #
 # version 01/2017 
 # author Nguyen Van Hiep (translation) ##
def texp_fit(look, xdataa, tdataa, xindxrange,
	     zrocnm, hgtcnm, cencnm, widcnm, tspincnm, ordercnm,
	     zrocnmyn, hgtcnmyn, cencnmyn, widcnmyn, tspincnmyn,
	     zrownm, hgtwnm, cenwnm, widwnm, fwnm,
	     zrownmyn, hgtwnmyn, cenwnmyn, widwnmyn, fwnmyn,
	     nloopmax = 50, halfasseduse = 0.5):

	dp600    = np.float64(0.60056120)
	nr_of_ns = int(len(xindxrange)/2)  # you can set like xindxrange=50,75,77,80
	
	datasize = 0
	for nnr in range(nr_of_ns):
		datasize = datasize + xindxrange[2*nnr+1]-xindxrange[2*nnr]+1

	xdata = np.zeros(datasize, dtype='float64')
	tdata = np.zeros(datasize, dtype='float64')

	## Data ##
	dtsiz = 0
	for nnr in range(nr_of_ns):
		dtsiz1              = dtsiz + xindxrange[2*nnr+1]-xindxrange[2*nnr] +1
		xdata[dtsiz:dtsiz1] = xdataa[xindxrange[2*nnr]:xindxrange[2*nnr+1]+1]
		tdata[dtsiz:dtsiz1] = tdataa[xindxrange[2*nnr]:xindxrange[2*nnr+1]+1]
		dtsiz               = dtsiz1
	## Endfor

	# AX1 IS THE PERCENTAGE OF CHANGE THAT WE ALLOW# 1% IS THE DEFAULT...
	ax1       = 0.01  # Change in each param after 1 loop

	# HALFASSED IS THE MULTIPLIER FOR THE CORRECTIONS IN NONLI!=AR REGIME.
	halfassed = halfasseduse

	#A NONZERO PROBLEM INDICATES A PROBLEM...
	problem   = 0

	#DFSTOP IS THE MAXIMUM WIDTH WE ALLOW, = 80% of the total window...
	dfstop    = 0.8*abs(xdata[datasize-1]-xdata[0])

	# THE OUTPUT GAUSSIAN PARAMETERS# SCALE WID FROM FWHM TO 1/E...
	# THESE ARE THE SAME AS THE PARAMETERS THAT ARE ITERATED.
	zrocnm1   = np.float64(zrocnm) # Scalar
	hgtcnm1   = np.array(hgtcnm, dtype='float64')
	cencnm1   = np.array(cencnm, dtype='float64')
	widcnm1   = dp600*np.array(widcnm, dtype='float64')
	tspincnm1 = np.array(tspincnm, dtype='float64')

	zrownm1   = np.float64(zrownm) # Scalar 
	hgtwnm1   = np.array(hgtwnm, dtype='float64')
	cenwnm1   = np.array(cenwnm, dtype='float64')
	widwnm1   = dp600*np.array(widwnm, dtype='float64')
	fwnm1     = np.array(fwnm, dtype='float64')

	nloop     = 0
	nloopn    = 0

	# get Number OF CNM AND WNM GAUSSIANS TO FIT...
	ngaussians_cnm = len(hgtcnm)
	ngaussians_wnm = len(hgtwnm)

	## get NR OF PARAMETERS TO FIT...
	nparams = zrocnmyn + sum(hgtcnmyn) + sum(cencnmyn) + sum(widcnmyn) + sum(tspincnmyn) + \
		      zrownmyn + sum(hgtwnmyn) + sum(cenwnmyn) + sum(widwnmyn) + sum(fwnmyn) 


	# TOTAL (MAXIMUM) NUMBER OF PARAMGERS THAT WOULD BE FIT IF ALL YesNo'S = 1...
	nparams_max = 2 + 4*(ngaussians_cnm + ngaussians_wnm)

	## EQUATION-OF-CONDITION ARRAY, S AND ITS COUNTERPART SFULL...
	datasize = int(datasize)
	nparams  = int(nparams)

	s          = np.zeros((datasize,nparams), dtype='float64')
	sfull      = np.zeros((datasize,nparams_max), dtype='float64')
	afull      = np.zeros(nparams_max, dtype='float64')
	sfull_to_s = [0]*nparams
	sigarraya  = np.zeros(nparams_max, dtype='float64')

	# RELATIONSHIP BETWEEN COLS IN S AND SFULL...
	# BEGIN WITH THE CNM PARAMETERS...
	scol     = 0
	sfullcol = 0

	if (zrocnmyn != 0):
		sfull_to_s[scol] = int(sfullcol)
		scol             = scol + 1 

	sfullcol = sfullcol + 1

	for ng in range(ngaussians_cnm):
		if (hgtcnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 

		sfullcol = sfullcol + 1

		if (cencnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 

		sfullcol = sfullcol + 1

		if (widcnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 

		sfullcol = sfullcol + 1

		if (tspincnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 

		sfullcol = sfullcol + 1

	## THEN THE WNM PARAMETERS...
	if (zrownmyn != 0):
		sfull_to_s[scol] = int(sfullcol)
		scol             = scol + 1 
	
	sfullcol = sfullcol + 1

	## when ONLY 1 single component, so hgtwnmyn=1, need to be a list hgtwnmyn=[1]
	if(type(hgtwnmyn) is int):
		hgtwnmyn = [hgtwnmyn]
		cenwnmyn = [cenwnmyn]
		widwnmyn = [widwnmyn]
		fwnmyn   = [fwnmyn]

	for ng in range(ngaussians_wnm): 
		if (hgtwnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 
		
		sfullcol = sfullcol + 1

		if (cenwnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 
		
		sfullcol = sfullcol + 1

		if (widwnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 
		
		sfullcol = sfullcol + 1

		if (fwnmyn[ng] != 0):
			sfull_to_s[scol] = int(sfullcol)
			scol             = scol + 1 
		
		sfullcol = sfullcol + 1

	### START LOOPING ###
	redoit = 1
	while (redoit == 1):
		nloop  = nloop  + 1
		nloopn = nloopn + 1

		sfullcol = 0

		# EVALUATE CONSTANT DERIVATIVE FOR CNM:
		xdel    = np.float64(0.0000025)
		zrocnm1 = zrocnm1 + xdel
		tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
			zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
		        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
		        

		zrocnm1 = zrocnm1 - 2.*xdel
		tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
			zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
		        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
		        

		zrocnm1   = zrocnm1 + xdel
		zrocnmder = (tb_totplus - tb_totminus)/(2.*xdel)

		sfull[:,sfullcol] = zrocnmder				#THE CONSTANT
		sfullcol          = sfullcol + 1

		# WORK THROUGH CNM GAUSSIANS...
		for ng in range(ngaussians_cnm):
			#EVALUATE HGT DERIVATIVE:
			xdel = np.float64(0.0000025)
			hgtcnm1[ng] = hgtcnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			hgtcnm1[ng] = hgtcnm1[ng] -2.* xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			hgtcnm1[ng] = hgtcnm1[ng] + xdel
			hgtder      = (tb_totplus - tb_totminus)/(2.*xdel)

			## EVALUATE CEN DERIVATIVE:
			xdel        = np.float64(0.0000025)*widcnm1[ng]
			cencnm1[ng] = cencnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			cencnm1[ng] = cencnm1[ng] - 2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			cencnm1[ng] = cencnm1[ng] + xdel
			cender      = (tb_totplus - tb_totminus)/(2.*xdel)

			## EVALUATE WID DERIVATIVE:
			xdel        = np.float64(0.0000025)*widcnm1[ng]
			widcnm1[ng] = widcnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			widcnm1[ng] = widcnm1[ng] - 2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			widcnm1[ng] = widcnm1[ng] + xdel
			widder      = (tb_totplus - tb_totminus)/(2.*xdel)

			## EVALUATE TSPIN DERIVATIVE:
			xdel        = np.float64(0.0000025)*tspincnm1[ng]
			# xdel        = np.float64(0.00025) * 100.0 ## Different
			tspincnm1[ng] = tspincnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			tspincnm1[ng] = tspincnm1[ng] -2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			tspincnm1[ng] = tspincnm1[ng] + xdel
			tspinder      = (tb_totplus - tb_totminus)/(2.*xdel)

			sfull[:, sfullcol] = hgtder     #HGT-*/41-*/7
			sfullcol           = sfullcol + 1

			sfull[:, sfullcol] = cender     #CNTR
			sfullcol           = sfullcol + 1

			sfull[:, sfullcol] = widder     #WIDTH
			sfullcol           = sfullcol + 1

			sfull[:, sfullcol] = tspinder   #TSPIN
			sfullcol           = sfullcol + 1

		## EVALUATE CONSTANT DERIVATIVE FOR WNM:
		xdel = np.float64(0.0000025 * 10.)
		zrownm1 = zrownm1 + xdel
		tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
			zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
		        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
		        
		zrownm1 = zrownm1 - 2.*xdel
		tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
			zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
		        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
		        
		zrownm1   = zrownm1 + xdel
		zrownmder = (tb_totplus - tb_totminus)/(2.*xdel)

		sfull[:, sfullcol] = zrownmder				#THE CONSTANT
		sfullcol           = sfullcol + 1

		# WORK THROUGH wnm GAUSSIANS...
		for ng in range(ngaussians_wnm):
			## EVALUATE HGT DERIVATIVE:
			xdel = np.float64(0.0000025)
			# xdel = np.float64(0.000025) * 10. # changed
			hgtwnm1[ng] = hgtwnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			hgtwnm1[ng] = hgtwnm1[ng] -2.* xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			hgtwnm1[ng] = hgtwnm1[ng] + xdel
			hgtder      = (tb_totplus - tb_totminus)/(2.*xdel)

			# EVALUATE CEN DERIVATIVE:
			xdel = np.float64(0.0000025)*widwnm1[ng]
			cenwnm1[ng] = cenwnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			cenwnm1[ng] = cenwnm1[ng] - 2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			cenwnm1[ng] = cenwnm1[ng] + xdel
			cender      = (tb_totplus - tb_totminus)/(2.*xdel)

			# EVALUATE WID DERIVATIVE:
			xdel = np.float64(0.0000025)*widwnm1[ng]
			widwnm1[ng] = widwnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			widwnm1[ng] = widwnm1[ng] - 2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			widwnm1[ng] = widwnm1[ng] + xdel
			widder = (tb_totplus - tb_totminus)/(2.*xdel)

			#EVALUATE F DERIVATIVE:
			# xdel = np.float64(0.0000025) * fwnm1[ng]
			xdel = np.float64(0.0000025)
			fwnm1[ng] = fwnm1[ng] + xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totplus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			fwnm1[ng] = fwnm1[ng] -2.*xdel
			tb_cont, tb_wnm_tot, tb_cnm_tot, tb_totminus, exp_tau_sum = tb_exp(xdata, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
			        zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)
			        
			fwnm1[ng] = fwnm1[ng] + xdel
			fder = (tb_totplus - tb_totminus)/(2.*xdel)
			sfull[ :, sfullcol] = hgtder #HGT
			sfullcol            = sfullcol + 1
			sfull[ :, sfullcol] = cender #CNTR
			sfullcol            = sfullcol + 1
			sfull[ :, sfullcol] = widder #WIDTH
			sfullcol            = sfullcol + 1
			sfull[ :, sfullcol] = fder      #f
			sfullcol            = sfullcol + 1

		s = sfull[:,sfull_to_s]

		# CALCULATE T_PREDICTED...
		tb_cont, tb_wnm_tot, tb_cnm_tot, t_predicted, exp_tau_sum = tb_exp(xdata, \
			zrocnm1, hgtcnm1, cencnm1, widcnm1/dp600, tspincnm1, ordercnm, \
		    zrownm1, hgtwnm1, cenwnm1, widwnm1/dp600, fwnm1)

		## CREATE AND SOLVE THE NORMAL EQUATION MATRICES...
		t   = tdata - t_predicted
		ss  = np.dot(np.transpose(s),s)
		st  = np.dot(np.transpose(s), np.transpose(t))
		ssi = np.linalg.inv(ss)
		a   = np.dot(ssi,st)
		afull[sfull_to_s] = np.squeeze(a)

		## CHECK THE DERIVED CNM PARAMETERS...
		## THE CNM AMPLITUDES...
		delt    = afull[ [x+1 for x in (x*4 for x in list(range(ngaussians_cnm)))]  ]
		adelt   = [abs(x) for x in delt]
		hgtcnm1 = [abs(x) for x in hgtcnm1]

		for i in range(len(adelt)):
			if(0.2*hgtcnm1[i] < adelt[i]):
				adelt[i] = 0.2*hgtcnm1[i]

		delthgtcnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				delthgtcnm.append(-adelt[i])
			else:
				delthgtcnm.append(adelt[i])

		## CNM CENTERS
		delt    = afull[ [x+2 for x in (x*4 for x in list(range(ngaussians_cnm)))]  ]
		adelt   = [abs(x) for x in delt]
		widcnm1 = [abs(x) for x in widcnm1]

		for i in range(len(adelt)):
			if(0.2*widcnm1[i] < adelt[i]):
				adelt[i] = 0.2*widcnm1[i]

		deltcencnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltcencnm.append(-adelt[i])
			else:
				deltcencnm.append(adelt[i])

		## CNM WIDTHS
		delt    = afull[ [x+3 for x in (x*4 for x in list(range(ngaussians_cnm)))]  ]
		adelt   = [abs(x) for x in delt]
		widcnm1 = [abs(x) for x in widcnm1]

		for i in range(len(adelt)):
			if(0.2*widcnm1[i] < adelt[i]):
				adelt[i] = 0.2*widcnm1[i]

		deltwidcnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltwidcnm.append(-adelt[i])
			else:
				deltwidcnm.append(adelt[i])

		## CNM Tex
		delt   = afull[ [x+4 for x in (x*4 for x in list(range(ngaussians_cnm)))]  ]
		adelt  = [abs(x) for x in delt]
		tscnm1 = [abs(x) for x in tspincnm1]

		for i in range(len(adelt)):
			if(0.2*tscnm1[i] < adelt[i]):
				adelt[i] = 0.2*tscnm1[i]

		delttspincnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				delttspincnm.append(-adelt[i])
			else:
				delttspincnm.append(adelt[i])
		
		## CHECK FOR CONVERGENCE AND IF CNM PARAMETERS are REASONABLE ##
		hgtf   = np.absolute(np.array(delthgtcnm)/hgtcnm1)
		cenf   = np.absolute(np.array(deltcencnm)/widcnm1)
		widf   = np.absolute(np.array(deltwidcnm)/widcnm1)
		tspinf = np.absolute(np.array(delttspincnm)/tspincnm1)

		redoit = 0
		if (max(hgtf) > ax1):
			redoit = 1
		if (max(cenf) > ax1):
			redoit = 1
		if (max(widf) > ax1):
			redoit = 1
		if (max(tspinf) > ax1):
			redoit = 1

		## CHECK THE DERIVED CNM PARAMETERS...
		## THE WNM AMPLITUDES...
		delt    = afull[ [x+4*ngaussians_cnm+2 for x in (x*4 for x in list(range(ngaussians_wnm)))]  ]
		adelt   = [abs(x) for x in delt]
		hgtwnm1 = [abs(x) for x in hgtwnm1]

		for i in range(len(adelt)):
			if(0.2*hgtwnm1[i] < adelt[i]):
				adelt[i] = 0.2*hgtwnm1[i]

		delthgtwnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				delthgtwnm.append(-adelt[i])
			else:
				delthgtwnm.append(adelt[i])

		## THE WNM CENTERS
		delt    = afull[ [x+4*ngaussians_cnm+3 for x in (x*4 for x in list(range(ngaussians_wnm)))]  ]
		adelt   = [abs(x) for x in delt]
		widwnm1 = [abs(x) for x in widwnm1]

		for i in range(len(adelt)):
			if(0.2*widwnm1[i] < adelt[i]):
				adelt[i] = 0.2*widwnm1[i]

		deltcenwnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltcenwnm.append(-adelt[i])
			else:
				deltcenwnm.append(adelt[i])

		## THE WNM WIDTHS
		delt    = afull[ [x+4*ngaussians_cnm+4 for x in (x*4 for x in list(range(ngaussians_wnm)))]  ]
		adelt   = [abs(x) for x in delt]
		widwnm1 = [abs(x) for x in widwnm1]

		for i in range(len(adelt)):
			if(0.2*widwnm1[i] < adelt[i]):
				adelt[i] = 0.2*widwnm1[i]

		deltwidwnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltwidwnm.append(-adelt[i])
			else:
				deltwidwnm.append(adelt[i])

		## THE WNM FRACTIONs
		delt    = afull[ [x+4*ngaussians_cnm+5 for x in (x*4 for x in list(range(ngaussians_wnm)))]  ]
		adelt   = [abs(x) for x in delt]
		fwnm1   = [abs(x) for x in fwnm1]

		for i in range(len(adelt)):
			if(0.2*fwnm1[i] < adelt[i]):
				adelt[i] = 0.2*fwnm1[i]

		deltfwnm = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltfwnm.append(-adelt[i])
			else:
				deltfwnm.append(adelt[i])

		## CHECK FOR CONVERGENCE AND IF WNM PARAMETERS are REASONABLE ##
		hgtwnmf = np.absolute(np.array(delthgtwnm)/hgtwnm1)
		cenwnmf = np.absolute(np.array(deltcenwnm)/widwnm1)
		widwnmf = np.absolute(np.array(deltwidwnm)/widwnm1)

		if(any(x==0. for x in fwnm1)):
			fwnmf = [0.,0.,0.]
		else:
			fwnmf = np.absolute(np.array(deltfwnm)/fwnm1)

		if (max(hgtwnmf) > ax1):
			redoit = 1
		if (max(cenwnmf) > ax1):
			redoit = 1
		if (max(widwnmf) > ax1):
			redoit = 1
		if (max(fwnmf) > ax1):
			redoit = 1

		# INCREMENT THE PARAMETERS...
		# halfassed = 0.5
		# halfassed = 0.4
		if(redoit == 0):
			halfassed = 1.0

		zrocnm1   = zrocnm1   + afull[0]               * halfassed
		hgtcnm1   = hgtcnm1   + np.array(delthgtcnm)   * halfassed
		cencnm1   = cencnm1   + np.array(deltcencnm)   * halfassed
		widcnm1   = widcnm1   + np.array(deltwidcnm)   * halfassed
		tspincnm1 = tspincnm1 + np.array(delttspincnm) * halfassed

		zrownm1 = zrownm1 + afull[4*ngaussians_cnm + 1] * halfassed
		hgtwnm1 = hgtwnm1 + np.array(delthgtwnm)        * halfassed
		cenwnm1 = cenwnm1 + np.array(deltcenwnm)        * halfassed
		widwnm1 = widwnm1 + np.array(deltwidwnm)        * halfassed
		fwnm1   = fwnm1   + np.array(deltfwnm)          * halfassed

		# CHECK TO SEE IF WIDTH IS TOO BIG..but ignore if these params are fixed.
		if ((np.max(np.array(widcnmyn)*widcnm1) > dfstop)  or \
			(np.min(np.array(widcnmyn)*widcnm1) < 0.)     or \
			(np.max(np.array(widwnmyn)*widwnm1) > dfstop) or \
			(np.min(np.array(widwnmyn)*widwnm1) < 0.) ):
		    problem = -1
		    break

	   	# nloop = 200
		if (nloop >= nloopmax-1):
			problem = -2
			break
	## ENDLOOP Here
	
	## IF WE GET THIS FAR, THE FIT IS FINISHED AND SUCCESSFUL...
	## CONVERT THE 1/E WIDTHS TO HALFWIDTHS...
	widcnm1 = widcnm1/dp600
	widwnm1 = widwnm1/dp600

	## DERIVE THE FITTED POINTS, RESIDUALS, THE ERRORS IN DERIVED COEFFICIENTS...
	## NOTE THAT THE WIDTHS HAVE BEEN CONVERTED TO HALFWIDTHS HERE, SO THE
	## 0.6 FACTORS ARE NOT REQUIRED...
	tb_cont, tb_wnm_tot, tb_cnm_tot, t_predicted, exp_tau_sum = tb_exp(xdata, \
		zrocnm1, hgtcnm1, cencnm1, widcnm1, tspincnm1, ordercnm, \
	    zrownm1, hgtwnm1, cenwnm1, widwnm1, fwnm1)
	        
	resid     = tdata - t_predicted
	resid2    = np.square(resid)
	sigsq     = resid2.sum()/(datasize - nparams)
	sigma     = sigsq**0.5

	ltemp     = list(range(nparams))
	ltemp     = [x*(nparams+1) for x in ltemp]
	ssi_temp  = ssi.ravel()
	sigarray  = sigsq*ssi_temp[ltemp]

	countsqrt = 0
	indxsqrt  = []
	jj        = 0
	for x in np.nditer(sigarray):
		if (x<0.):
			countsqrt = countsqrt + 1
			indxsqrt.append(jj)

		jj = jj + 1

	sigarray = np.sqrt( abs(sigarray))

	## TEST FOR NEG SQRTS...
	if (countsqrt != 0):
		sigarray[indxsqrt] = -sigarray[indxsqrt]
		problem = -3

	## TEST FOR INFINITIES, ETC...
	countbad = 0
	indxbad  = []
	kk       = 0
	for x in np.nditer(a):
		if not np.isfinite(x):
			countbad = countbad + 1
			indxbad.append(kk)

		kk = kk + 1

	if (countbad != 0):
		problem = -4

	sigarraya[sfull_to_s] = sigarray
	sigzrocnm1            = sigarraya[0]
	temp_list             = [x*4 for x in list(range(ngaussians_cnm))]
	sighgtcnm1            = sigarraya[ [x+1 for x in temp_list] ]
	sigcencnm1            = sigarraya[ [x+2 for x in temp_list] ]
	sigwidcnm1            = sigarraya[ [x+3 for x in temp_list] ]/dp600
	sigtspincnm1          = sigarraya[ [x+4 for x in temp_list] ]

	temp_list             = [x*4 for x in list(range(ngaussians_wnm))]
	sigzrownm1            = sigarraya[ 4*ngaussians_cnm + 1]
	sighgtwnm1            = sigarraya[ [x+4*ngaussians_cnm+2 for x in temp_list] ]
	sigcenwnm1            = sigarraya[ [x+4*ngaussians_cnm+3 for x in temp_list] ]
	sigwidwnm1            = sigarraya[ [x+4*ngaussians_cnm+4 for x in temp_list] ]/dp600
	sigfwnm1              = sigarraya[ [x+4*ngaussians_cnm+5 for x in temp_list] ]


	## DERIVE THE NORMALIZED COVARIANCE ARRAY
	temp_list = [x*(nparams+1) for x in list(range(nparams))]
	ssi_temp  = ssi.ravel()
	doug      = ssi_temp[temp_list]
	doug_temp = doug[np.newaxis]
	doug_t    = doug[np.newaxis].T
	doug      = np.dot(doug_t,doug_temp)
	cov       = ssi/np.sqrt(doug)

	tb_cont, tb_wnm_tot, tb_cnm_tot, tfita, exp_tau_sum = tb_exp(xdata, \
		zrocnm1, hgtcnm1, cencnm1, widcnm1, tspincnm1, ordercnm, \
	    zrownm1, hgtwnm1, cenwnm1, widwnm1, fwnm1)

	

	# ## Plot Absoprtion line
	# plt.plot(xdata,tdata, 'b-', linewidth=1, label='data')
	# plt.plot(xdata,tfita, 'r-', linewidth=1, label='fit')
	# plt.title('3C98', fontsize=30)
	# plt.ylabel('$T_{b} [K]$', fontsize=35)
	# plt.xlabel('$V_{lsr} (km/s)$', fontsize=35)
	# # plt.xlim(0.0, 2.0)
	# # plt.xlim(-1.0, 6.0)
	# plt.grid(True)
	# plt.tick_params(axis='x', labelsize=18)
	# plt.tick_params(axis='y', labelsize=15)

	# # plt.text(0.0, 3.2, r'$f = [0.32\pm0.06]\cdot log_{10}(N^*_{HI}/10^{20}) + [0.81\pm0.05]$, Lee et al.', color='blue', fontsize=20)
	# plt.legend(loc='upper left', fontsize=18)
	# plt.show()
	        
	

	return tfita, sigma, resid,\
			zrocnm1, hgtcnm1, cencnm1, widcnm1, tspincnm1, \
			sigzrocnm1, sighgtcnm1, sigcencnm1, sigwidcnm1, sigtspincnm1, \
			zrownm1, hgtwnm1, cenwnm1, widwnm1, fwnm1, \
			sigzrownm1, sighgtwnm1, sigcenwnm1, sigwidwnm1, sigfwnm1, \
			cov, problem, nloop, \
	        tb_cont, tb_wnm_tot, tb_cnm_tot, \
	        exp_tau_sum, nloopmax, halfasseduse










##  Fit on-source spectrum: tau ##
 #
 # params...
 # return... 	
 #
 # version 01/2017 
 # author Nguyen Van Hiep ##
def emt_fit(look, xdataa, tdataa, xindxrange,
	    zro0,   hgt0,   cen0,   wid0,
	    zro0yn, hgt0yn, cen0yn, wid0yn):

	dp600    = np.float64(0.60056120)           # To convert between FWHM and 1/e-width
	hgt0     = np.array(hgt0, dtype='float64')
	cen0     = np.array(cen0, dtype='float64')
	wid0     = np.array(wid0, dtype='float64')

	nr_of_ns = int(len(xindxrange)/2)
	datasize = 0
	for nnr in range(nr_of_ns):
		datasize = datasize + xindxrange[2*nnr+1]-xindxrange[2*nnr]+1

	xdata = np.zeros(datasize, dtype='float64')
	tdata = np.zeros(datasize, dtype='float64')

	## Data ##
	dtsiz = 0
	for nnr in range(nr_of_ns):
		dtsiz1              = dtsiz + xindxrange[2*nnr+1]-xindxrange[2*nnr] + 1
		print('Number of bins: ', dtsiz1)
		xdata[dtsiz:dtsiz1] = xdataa[xindxrange[2*nnr]:xindxrange[2*nnr+1]+1]
		tdata[dtsiz:dtsiz1] = tdataa[xindxrange[2*nnr]:xindxrange[2*nnr+1]+1]
		dtsiz               = dtsiz1

	# AX1 IS THE PERCENTAGE OF CHANGE THAT WE ALLOW# 1% IS THE DEFAULT...
	ax1 = np.float64(0.01)

	# HALFASSED IS THE MULTIPLIER FOR THE CORRECTIONS IN NONLI!=AR REGIME.
	# if (halfasseduse == No!=):
	halfassed = np.float64(0.5)

	#if (nloopmax == No!=):
	nloopmax = 350

	#A NONZERO PROBLEM INDICATES A PROBLEM...
	problem = 0

	#DFSTOP IS THE MAXIMUM WIDTH WE ALLOW, = 80% of the total window...
	dfstop = 0.8*abs(xdata[datasize-1]-xdata[0])

	# THE OUTPUT GAUSSIAN PARAMETERS# SCALE WID FROM FWHM TO 1/E...
	# THESE ARE THE SAME AS THE PARAMETERS THAT ARE ITERATED.
	zro1   = zro0 # a scalar
	hgt1   = hgt0
	cen1   = cen0
	wid1   = dp600*wid0

	nloop  = 0
	nloopn = 0

	# get Number OF GAUSSIANS TO FIT...
	ngaussians = len(hgt0)

	## get NR OF PARAMETERS TO FIT...
	nparams = zro0yn + sum(hgt0yn) + sum(cen0yn) + sum(wid0yn)

	# TOTAL NR OF PARAMGERS THAT WOULD BE FIT IF ALL YesNo'S = 1...
	nparams_max = 1 + 3*ngaussians

	datasize = int(datasize)
	nparams  = int(nparams)

	## EQUATION-OF-CONDITION ARRAY, S AND ITS COUNTERPART SFULL...
	s          = np.zeros((datasize,nparams), dtype='float64')
	sfull      = np.zeros((datasize,nparams_max), dtype='float64')
	afull      = np.zeros(nparams_max, dtype='float64')
	sfull_to_s = [0]*nparams
	s_to_sfull = [0]*nparams_max
	sigarraya  = np.zeros(nparams_max, dtype='float64')

	# RELATIONSHIP BETWEEN COLS IN S AND SFULL...
	scol     = 0
	sfullcol = 0

	if (zro0yn != 0):
		s_to_sfull[0]    = int(scol)
		sfull_to_s[scol] = 0
		scol             = scol + 1

	for ng in range(ngaussians):
		if (hgt0yn[ng] != 0):
			s_to_sfull[3*ng+1] = int(scol)
			sfull_to_s[scol]   = 3*ng + 1
			scol               = scol + 1

		if (cen0yn[ng] != 0):
			s_to_sfull[3*ng+2] = int(scol)
			sfull_to_s[scol]   = 3*ng + 2
			scol               = scol + 1

		if (wid0yn[ng] != 0):
			s_to_sfull[3*ng+3] = int(scol)
			sfull_to_s[scol]   = 3*ng + 3
			scol               = scol + 1

	### Start looping ###
	redoit = 1
	while (redoit == 1):
		nloop  = nloop  + 1
		nloopn = nloopn + 1

		## FIRST DEFINE SFULL...
		sum_of_gaussians = gcurv(xdata, zro1, hgt1, cen1, wid1/dp600)
		t_predicted      = np.exp(-sum_of_gaussians)
		expfactor        = -t_predicted
		sfull[:,0]       = expfactor 			## THE CONSTANT

		for ng in range(ngaussians):
			xdel                 = (xdata - cen1[ng])/wid1[ng]
			edel                 = np.exp(-xdel*xdel)
			sum1                 = edel
			sum2                 = edel*xdel
			sum3                 = sum2*xdel
			sum6                 = np.float64(2.0)*hgt1[ng]/wid1[ng]
			sfull[ :, (3*ng+1) ] = expfactor*sum1          ## HGT
			sfull[ :, (3*ng+2) ] = expfactor*sum2*sum6     ## CNTR
			sfull[ :, (3*ng+3) ] = expfactor*sum3*sum6     ## WIDTH

		s = sfull[:,sfull_to_s]

		## CREATE AND SOLVE THE NORMAL EQUATION MATRICES...
		t   = tdata-t_predicted
		ss  = np.dot(np.transpose(s),s)
		st  = np.dot(np.transpose(s), np.transpose(t))
		ssi = np.linalg.inv(ss)
		a   = np.dot(ssi,st)
		afull[sfull_to_s] = np.squeeze(a)

		## CHECK THE DERIVED CNM PARAMETERS...
		## THE AMPLITUDES...
		delt  = afull[ [x+1 for x in (x*3 for x in list(range(ngaussians)))]  ]
		adelt = [abs(x) for x in delt]
		hgt1  = [abs(x) for x in hgt1]

		for i in range(len(adelt)):
			if(0.2*hgt1[i] < adelt[i]):
				adelt[i] = 0.2*hgt1[i]

		delthgt = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				delthgt.append(-adelt[i])
			else:
				delthgt.append(adelt[i])

		## CENTERS
		delt  = afull[ [x+2 for x in (x*3 for x in list(range(ngaussians)))]  ]
		adelt = [abs(x) for x in delt]
		wid1  = [abs(x) for x in wid1]

		for i in range(len(adelt)):
			if(0.2*wid1[i] < adelt[i]):
				adelt[i] = 0.2*wid1[i]

		deltcen = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltcen.append(-adelt[i])
			else:
				deltcen.append(adelt[i])

		## WIDTHS
		delt  = afull[ [x+3 for x in (x*3 for x in list(range(ngaussians)))]  ]
		adelt = [abs(x) for x in delt]
		wid1  = [abs(x) for x in wid1]

		for i in range(len(adelt)):
			if(0.2*wid1[i] < adelt[i]):
				adelt[i] = 0.2*wid1[i]

		deltwid = []
		for i in range(len(delt)):
			if(delt[i] < 0.):
				deltwid.append(-adelt[i])
			else:
				deltwid.append(adelt[i])

		## CHECK FOR CONVERGENCE AND IF CNM PARAMETERS are REASONABLE ##
		delthgt = np.array(delthgt)
		deltcen = np.array(deltcen)
		deltwid = np.array(deltwid)
		hgtf    = np.absolute(delthgt/hgt1)
		cenf    = np.absolute(deltcen/wid1)
		widf    = np.absolute(deltwid/wid1)

		redoit = 0
		if (max(hgtf) > ax1):
			redoit = 1
		if (max(cenf) > ax1):
			redoit = 1
		if (max(widf) > ax1):
			redoit = 1

		# INCREMENT THE PARAMETERS...
		if(redoit == 0):
			halfassed = 1.0

		zro1   = zro1 + halfassed * afull[0]
		hgt1   = hgt1 + halfassed * delthgt
		cen1   = cen1 + halfassed * deltcen
		wid1   = wid1 + halfassed * deltwid

		# CHECK TO SEE IF WIDTH IS TOO BIG..but ignore if these params are fixed.
		if ((max(wid0yn*wid1) > dfstop)  or \
			(min(wid0yn*wid1) < 0.)  ):
		    problem = -1
		    break

		if (nloop >= nloopmax-1):
			problem = -2
			break
	
	## CONVERT THE 1/E WIDTHS TO HALFWIDTHS...
	wid1 = wid1/dp600

	## DERIVE THE FITTED POINTS, RESIDUALS, THE ERRORS IN DERIVED COEFFICIENTS...
	## NOTE THAT THE WIDTHS HAVE BEEN CONVERTED TO HALFWIDTHS HERE, SO THE
	## 0.6 FACTORS ARE NOT REQUIRED...
	sum_of_gaussians = gcurv(xdata, zro1, hgt1, cen1, wid1)
	t_predicted      = np.exp(-sum_of_gaussians)
	        
	resid  = tdata - t_predicted
	resid2 = resid**2
	ressum = resid2.sum()
	sigsq  = ressum/(datasize - nparams)
	sigma  = sigsq**0.5

	ltemp    = list(range(nparams))
	ltemp    = [x*(nparams+1) for x in ltemp]
	ssi_temp = ssi.ravel()
	sigarray = sigsq*ssi_temp[ltemp]

	countsqrt = 0
	indxsqrt  = []
	jj        = 0
	for x in np.nditer(sigarray):
		if (x<0.):
			countsqrt = countsqrt + 1
			indxsqrt.append(jj)
		jj = jj + 1

	sigarray = np.sqrt( abs(sigarray))

	## TEST FOR NEG SQRTS...
	if (countsqrt != 0):
		sigarray[indxsqrt] = -sigarray[indxsqrt]
		problem = -3

	## TEST FOR INFINITIES, ETC...
	countbad = 0
	indxbad  = []
	kk       = 0
	for x in np.nditer(a):
		if not np.isfinite(x):
			countbad = countbad + 1
			indxbad.append(kk)

		kk = kk + 1

	if (countbad != 0):
		problem = -4

	sigarraya[sfull_to_s] = sigarray
	sigzro1               = sigarraya[0]
	temp_list             = [x*3 for x in list(range(ngaussians))]
	sighgt1               = sigarraya[ [x+1 for x in temp_list] ]
	sigcen1               = sigarraya[ [x+2 for x in temp_list] ]
	sigwid1               = sigarraya[ [x+3 for x in temp_list] ]/dp600

	## DERIVE THE NORMALIZED COVARIANCE ARRAY
	temp_list = [x*(nparams+1) for x in list(range(nparams))]
	ssi_temp  = ssi.ravel()
	doug      = ssi_temp[temp_list]
	doug_temp = doug[np.newaxis]
	doug_t    = doug[np.newaxis].T
	doug      = np.dot(doug_t,doug_temp)
	cov       = ssi/np.sqrt(doug)

	sum_of_gaussians = gcurv(xdata, zro1, hgt1, cen1, wid1)
	tfita            = np.exp(-sum_of_gaussians)

	
	## Plot Absoprtion line
	if look > 0:
		plt.figure(figsize=(10,8))
		plt.plot(xdata, tdata, 'b-', linewidth=1, label='data')
		plt.plot(xdata, tfita, 'r-', linewidth=1, label='fit')
		plt.title('<...>', fontsize=30)
		plt.ylabel(r'$e^{-\tau} [K]$', fontsize=35)
		plt.xlabel('$V_{lsr} (km/s)$', fontsize=35)
		# plt.xlim(0.0, 2.0)
		# plt.xlim(-1.0, 6.0)
		plt.grid(True)
		plt.tick_params(axis='x', labelsize=18)
		plt.tick_params(axis='y', labelsize=15)

		# plt.text(0.0, 3.2, r'$f = [0.32\pm0.06]\cdot log_{10}(N^*_{HI}/10^{20}) + [0.81\pm0.05]$, Lee et al.', color='blue', fontsize=20)
		plt.legend(loc='upper left', fontsize=18)
		plt.show()
	        
	return tfita, sigma, resid,\
			zro1, hgt1, cen1, wid1,\
			sigzro1, sighgt1, sigcen1, sigwid1,\
			cov, problem,\
			nparams