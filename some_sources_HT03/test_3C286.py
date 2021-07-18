import sys, os
sys.path.insert(0, os.getenv("HOME")+'/Phd@MQ/projects/Dark') # add folder of Class

from common.myimport import *
from common.gfitflex import gfit
# from common.gfit_oh  import gfit


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
	#DETERMINE NR OF GAUSSIANS...
	ng = len(h)

	v  = np.array(v)
	h  = np.array(h)
	v0 = np.array(v0)
	w  = np.array(w)

	tf = 0.*v + zr
	for i in range(ng):
		if (w[i] > 0.):
			tf = tf + h[i]*np.exp(- ( (v-v0[i])/(0.6005612*w[i]))**2) # 0.6005612 - 1/e width

	return tf


## ============= MAIN ================ ##
## Class
fit  = gfit()

src  = '3C286'
print 'Fitting...' + src

data = readsav('../idl/gfit_idl_claire/3C286.sav')
# % RESTORE: Restored variable: VLSR.
# % RESTORE: Restored variable: SPEC1.
# % RESTORE: Restored variable: EM_SPEC.
# % RESTORE: Restored variable: SIGMA.
# % RESTORE: Restored variable: EMSIGMA.
# % RESTORE: Restored variable: PSR1.
# % RESTORE: Restored variable: VLSREM.
# % RESTORE: Restored variable: CONT.
# % RESTORE: Restored variable: RMS.

## ABS
cols    = ['id', 'src', 'xdata', 'taudata']
format  = ['i', 's', 'f', 'f']
absDat  = txtDat.readcol(fname='3c286_ABS_data.txt', cols=cols, fmt=format, skip=0, asarray=False)
# vlsr    = data.vlsr[:,0]
# tau     = data.spec1[:,0,0]

vlsr    = np.array(absDat['xdata'], dtype='float64')
tau     = np.array(absDat['taudata'], dtype='float64')

sigtau  = data.sigma
cont    = data.cont

# print cont

# sys.exit()

## EM - Set the velocity range for the emission spectra (to trim Carl's data)
cols    = ['id', 'src', 'xdataem', 'em_spec']
format  = ['i', 's', 'f', 'f']
emDat   = txtDat.readcol(fname='3c286_EM_data.txt', cols=cols, fmt=format, skip=0, asarray=False)

vlsrem  = emDat['xdataem'] # data.vlsrem
vmin, \
vmax    = get_vrange_id(vlsrem, -100., 100.)
Te      = emDat['em_spec'] # data.em_spec
xdataem = np.array(vlsrem[vmin:vmax+1], dtype='float64')
Te      = np.array(Te[vmin:vmax+1], dtype='float64')

# plt.plot(xdataem, Te)
# plt.show()

# plt.plot(vlsr, tau)
# plt.show()


# sys.exit()

# Retrieve initial Gaussian guesses for the absorption spectrum
zro0   = 0.0
hgt0   = [0.0065, 0.0066, 0.009]
cen0   = [-28.48, -14.4, -7.3]
wid0   = [2.32,    4.66,   5.0]

hgt0   = [1., 1., 1.]
cen0   = [-30., -15., -5.]
wid0   = [1., 1., 1.]
look   = 0

nrg    = len(hgt0)
zro0yn = 0
hgt0yn = [1]*nrg
cen0yn = [1]*nrg
wid0yn = [1]*nrg
corder = 'no'

## WNM
tspin0 = [0.]*nrg
order0 = list(range(nrg))

zro0yn   = 0
tau0yn   = [1]*nrg
cenc0yn  = [1]*nrg
wid0yn   = [1]*nrg
tspin0yn = [0]*nrg

zrownm = 1.
hgtwnm = [0.003]
cenwnm = [-10.4]
widwnm = [21.46]
fwnm   = [0.5]

zrownmyn = 1
hgtwnmyn = 0
cenwnmyn = 0
widwnmyn = 0
fwnmyn   = 0

## Fit these guesses
# tau0  = fit.gcurv(vlsr, zro0, hgt0, cen0, wid0)
# tfit0 = np.exp(-tau0)

## ABS line - From Guesses
# plt.plot(vlsr, tfit0, 'b-', linewidth=2, label='data, Tau abs line')
# plt.title(src, fontsize=30)
# plt.ylabel(r'$\tau$', fontsize=35)
# plt.xlabel('$V_{lsr} (km/s)$', fontsize=35)
# # plt.xlim(0.0, 2.0)
# # plt.xlim(-1.0, 6.0)
# plt.grid(True)
# plt.tick_params(axis='x', labelsize=18)
# plt.tick_params(axis='y', labelsize=15)
# plt.legend(loc='upper left', fontsize=18)
# plt.show()

# tfita, sigma, \
# zro1, hgt1, cen1, wid1,\
# sigzro1, sighgt1, sigcen1, sigwid1,\
# cov, problem,\
# nparams = fit.abfit(look, vlsr, tau, [0, len(tau)-1], zro0, hgt0, cen0, wid0, zro0yn, hgt0yn, cen0yn, wid0yn)

tfita, sigma, resida,\
zro1, hgt1, cen1, wid1,\
sigzro1, sighgt1, sigcen1, sigwid1,\
cov, problem,\
nparams = fit.fit(look, vlsr, tau, [0, len(tau)-1],\
			      zro0, hgt0, cen0, wid0,\
				  zro0yn, hgt0yn, cen0yn, wid0yn)

print 'Absorption line: problem...', problem

print '1. sigma ', sigma
print '2. Zro ', zro1
print '3. tau ', hgt1
print '\t\t', sighgt1
print '4. cen ', cen1
print '\t\t', sigcen1
print '5. wid ', wid1
print '\t\t', sigwid1

print ''
print ''

zrownm1   = 0.0
hgtwnm1   = [1,1]
cenwnm1   = [-5,-20]
widwnm1   = [10,10]
look      = 0
nrgwnm    = len(hgtwnm1)
zrownm1yn = 1
hgtwnm1yn = [1]*nrgwnm
cenwnm1yn = [1]*nrgwnm
widwnm1yn = [1]*nrgwnm
fwnm1     = [0]*nrgwnm
fwnm1yn   = [0]*nrgwnm

order1    = list(range(nrg))
tspin1    = [30.]*nrg
tspin1yn  = [1]*nrg

zrocnm1   = 0.
hgtcnm1   = hgt1
cencnm1   = cen1
widcnm1   = wid1
zrocnm1yn = 0
hgtcnm1yn = [0]*nrg
cencnm1yn = [0]*nrg
widcnm1yn = [0]*nrg

look      = -1
xindxrange= [0,len(xdataem)-1]

## ---Parameters within tbgfitflex_exp.pro, sets number of loops (nloopmax)
## ---and the fractional change in each parameter per loop iteration
nloopmax     = 100
halfasseduse = 0.2

## ---Compute Tsky at the source position [**predict_sync method currently not working, 
## ---so I just set a generic value here, will fix in the future]
## @galactic_coordinates.pro
## print, 'gl gb', gl, ' ', gb
## tsky=predict_sync(gl,gb, nu=1.4, /interp)+2.725 
tsky  = 2.8 ## 3.41
tdata = Te + tsky

simple = False
if(simple):
	tfite, sigma, reside,\
	zrocnm1, hgtcnm1, cencnm1, widcnm1, tspincnm1, \
	sigzrocnm1, sighgtcnm1, sigcencnm1, sigwidcnm1, sigtspincnm1, \
	zrownm1, hgtwnm1, cenwnm1, widwnm1, fwnm1, \
	sigzrownm1, sighgtwnm1, sigcenwnm1, sigwidwnm1, sigfwnm1, \
	cov, problem, nloop, \
	tb_cont, tb_wnm_tot, tb_cnm_tot, \
	exp_tau_sum, nloopmax, halfasseduse = fit.efit(look, xdataem, tdata, xindxrange, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1, tspin1, order1, \
				zrocnm1yn, hgtcnm1yn, cencnm1yn, widcnm1yn, tspin1yn, \
				cont, hgtwnm1, cenwnm1, widwnm1, fwnm1, \
				1, hgtwnm1yn, cenwnm1yn, widwnm1yn, fwnm1yn, nloopmax=nloopmax, halfasseduse=0.2)

	tb_tot_fit = tb_cnm_tot+tb_wnm_tot+tb_cont-tsky
	tb_tot_fit = tfite-tsky
	tdata      = tdata-tsky

else:
	print 'Not Simple!!'
	nrgcnm = nrg
	if( (nrgcnm == 0) or (nrgcnm == 1) ):
		orders = [0]
	else:
		import itertools
		## Orders of (peaks) components
		orders = list(itertools.permutations(range(nrgcnm)))
		orders = [list(x) for x in orders]
	## Endif for orders

	f_pom      = np.zeros( (3**nrgwnm, nrgwnm), dtype='float64')                               #fltarr(nrgwnm,3^nrgwnm)
	sigmas     = np.zeros( 3**nrgwnm*np.math.factorial(nrgcnm), dtype='float64')             #fltarr(3^nrgwnm*factorial(nrgcnm))
	Tspins     = np.zeros( (3**nrgwnm*np.math.factorial(nrgcnm), nrgcnm), dtype='float64') ## fltarr(nrgcnm, 3^nrgwnm*factorial(nrgcnm))
	Tspins_err = np.zeros( (3**nrgwnm*np.math.factorial(nrgcnm), nrgcnm), dtype='float64') ## fltarr(nrgcnm, 3^nrgwnm*factorial(nrgcnm))

	sigmatemp  = np.zeros( np.math.factorial(nrgcnm), dtype='float64')                       ## fltarr(factorial(nrgcnm))
	orders_all = np.zeros( (3**nrgwnm*np.math.factorial(nrgcnm), nrgcnm), dtype='float64')  ## fltarr(nrgcnm,3^nrgwnm*factorial(nrgcnm))

	for k in range(nrgwnm):
		f_pom[:,k] = np.mod( np.arange(3**nrgwnm)/3**k, 3 )

	f_pom = f_pom*0.5

	# [[0.  0. ]
	#  [0.5 0. ]
	#  [1.  0. ]
	#  [0.  0.5]
	#  [0.5 0.5]
	#  [1.  0.5]
	#  [0.  1. ]
	#  [0.5 1. ]
	#  [1.  1. ]]

	for j in range(3**nrgwnm):
		fwnm1 = f_pom[j,:]
		print 'Starting orders loop: ', j
		print str(100.*round( float(j)/float(3**nrgwnm),2 ) ) + '%'

		for oval in range(np.math.factorial(nrgcnm)):
			order1 = orders[oval]

			look = -1

			tfita, sigmaw, reside,\
			zrocnm2, hgtcnm2, cencnm2, widcnm2, tspin2, \
			sigzrocnm2, sighgtcnm2, sigcencnm2, sigwidcnm2, sigtspin2, \
			zrownm2, hgtwnm2, cenwnm2, widwnm2, fwnm2, \
			sigzrownm2, sighgtwnm2, sigcenwnm2, sigwidwnm2, sigfwnm2, \
			cov, problem, nloop, \
			tb_cont, tb_wnm_tot, tb_cnm_tot, \
			exp_tau_sum, nloopmax, halfasseduse = fit.efit(look, xdataem, tdata, xindxrange, \
						zrocnm1, hgtcnm1, cencnm1, widcnm1, tspin1, order1, \
						zrocnm1yn, hgtcnm1yn, cencnm1yn, widcnm1yn, tspin1yn, \
						cont, hgtwnm1, cenwnm1, widwnm1, fwnm1, \
						1, hgtwnm1yn, cenwnm1yn, widwnm1yn, fwnm1yn, nloopmax=nloopmax, halfasseduse=0.2)

			print '>>>> Results: <<<<<<'
			print 'fwnm is: ', f_pom[j, :]
			print 'order1 is:', order1
			print 'Tspin is: ', tspin2
			print '   - sigTsin:', sigtspin2
			print ''
			print ''

			sigmas[j*np.math.factorial(nrgcnm)+oval]        = sigmaw
			Tspins[j*np.math.factorial(nrgcnm)+oval, :]     = tspin2
			Tspins_err[j*np.math.factorial(nrgcnm)+oval, :] = sigtspin2
		## Endfor oval
	## Endfor j

	tspin_final      = np.zeros( nrgcnm, dtype='float64')
	tspin_err_final  = np.zeros( nrgcnm, dtype='float64')
	F                = (np.math.factorial(nrgcnm))*(3**nrgwnm)
	for j in range(nrgcnm):
		w         = (1.0/sigmas)**2

		tspin_val = np.sum(w*Tspins[:,j])/np.sum(w)

		tspin_err = ( np.sum( w*((Tspins[:,j]-tspin_val)**2+Tspins_err[:,j]**2) )/np.sum(w) )*(F/(F-1.))
		tspin_err = np.sqrt(tspin_err)

		tspin_final[j]     = tspin_val
		tspin_err_final[j] = tspin_err
	## Endfor

	print '---------------------------'
	print 'Final:'
	for jj in range(nrgcnm):
		print 'Tspin is: ', tspin_final[jj], '+/-', tspin_err_final[jj]
	## Endfor jj

	m = np.argmin(sigmas)
	print 'where sigmas is min:', m

	fwnm1   = f_pom[ int( np.floor( float(m)/np.math.factorial(nrgcnm) ) ), :]
	order1  = orders[ int(np.mod(float(m), np.math.factorial(nrgcnm) ) )]
	fwnm1yn = np.zeros( nrgwnm, dtype='float64')

	print 'fwnm: ', fwnm1
	print 'order1: ', order1

	look = -1

	tfita, sigmaw, reside,\
	zrocnm2, hgtcnm2, cencnm2, widcnm2, tspin2, \
	sigzrocnm2, sighgtcnm2, sigcencnm2, sigwidcnm2, sigtspin2, \
	zrownm2, hgtwnm2, cenwnm2, widwnm2, fwnm2, \
	sigzrownm2, sighgtwnm2, sigcenwnm2, sigwidwnm2, sigfwnm2, \
	cov, problem, nloop, \
	tb_cont, tb_wnm_tot, tb_cnm_tot, \
	exp_tau_sum, nloopmax, halfasseduse = fit.efit(look, xdataem, tdata, xindxrange, \
				zrocnm1, hgtcnm1, cencnm1, widcnm1, tspin1, order1, \
				zrocnm1yn, hgtcnm1yn, cencnm1yn, widcnm1yn, tspin1yn, \
				cont, hgtwnm1, cenwnm1, widwnm1, fwnm1, \
				1, hgtwnm1yn, cenwnm1yn, widwnm1yn, fwnm1yn, nloopmax=nloopmax, halfasseduse=0.2)

	tb_tot_fit = tfita-tsky
	tdata      = tdata-tsky
## End else

# print zrocnm1, hgtcnm1, cencnm1, widcnm1, tspincnm1
print 'Expected Emission profile: problem...', problem
print 'Nloop', nloop
print 'xindxrange', xindxrange

print len(xindxrange)
print len(tdata)
print len(tb_tot_fit)

# print '1. sigma ', sigmaw
# print '2. Zro ', zrocnm2
# print '3. tau ', hgtcnm2
# print '\t\t', sighgt1
# print '4. cen ', cen1
# print '\t\t', sigcen1
# print '5. wid ', wid1
# print '\t\t', sigwid1
# print '6. Tspin ', tspin1
# print '\t\t', sigtspin1

# print ''
# print ''

# print '7. Tau-WNM ', hgtwnm1
# print '\t\t', sighgtwnm1
# print '8. V0-WNM ', cenwnm1
# print '\t\t', sigcenwnm1
# print '9. Width-WNM ', widwnm1
# print '\t\t', sigwidwnm1
# print '10. fwnm1-WNM ', fwnm1
# print '\t\t', sigfwnm1


plt.plot(xdataem, tdata)
plt.plot(xdataem, tb_tot_fit, 'r-')
plt.show()

## Calculating column densities ##
NH_cnm       = 1.064467 * 0.0183 * 1. * hgt1 * wid1 * tspin2
pom          = sighgt1**2*(0.0194797*wid1*tspin2)**2 + sigwid1**2*(0.0194797*hgt1*tspin2)**2 + sigtspin2*(0.0194797*hgt1*wid1)**2
delta_NH_cnm = np.sqrt(pom)
print 'NH_cnm: ', NH_cnm, '+/-', delta_NH_cnm

NH_wnm       = 1.064467 * 0.0183 * 1. * hgtwnm2 * widwnm2
pomw         = NH_wnm**2*( (sighgtwnm2/hgtwnm2)**2 + (sigwidwnm2/widwnm2)**2 )
delta_NH_wnm = np.sqrt(pomw)
print 'NH_wnm: ', NH_wnm, '+/-', delta_NH_wnm

## solving for Tkmax
Tkmax    = 21.855*wid1**2
sigTkmax = (sigwid1/wid1)*2*Tkmax
Tkmaxw   = 21.855*(widwnm2)**2

print 'Sigma for the fit: ', sigmaw
for nr in range(nrg):
	print src, ' ', nr
	print '>Tkmax:', Tkmax[nr]
	print '>>> sigTkmax: ', sigTkmax[nr]
	print ''
	print '>Tspin[K]: ', tspin2[nr]
	print '>>sigTspin[K]: ', sigtspin2[nr]
## endfor

print ''

print 'cenwnm2'
print cenwnm2
print 'widwnm2'
print widwnm2
print 'hgtwnm2'
print hgtwnm2 
print ''

sys.exit()

## ABS line fit & residuals
# fig1 = plt.figure(1)
# frame1=fig1.add_axes((.1,.3,.8,.7))
# plt.plot(vlsr, tfita, 'r-', linewidth=2, label='data, fit')
# plt.plot(vlsr,tau, 'b-', linewidth=2, label='data, Absorption line')
# plt.ylabel(r'$\tau$', fontsize=35)
# plt.tick_params(axis='y', labelsize=15)
# plt.tick_params(axis='x', labelsize=18, labelbottom='off')
# plt.legend(loc='upper left', fontsize=18)
# plt.grid()

# frame2=fig1.add_axes((.1,.1,.8,.2))
# difference = tau - tfita
# plt.plot(vlsr, difference, 'r-', linewidth=2, label='')
# frame2.set_ylabel('$Residual$',fontsize=20)
# plt.xlabel('$V_{lsr} (km/s)$', fontsize=35)
# plt.tick_params(axis='x', labelsize=18)
# plt.tick_params(axis='y', labelsize=15)
# # plt.legend(loc='upper left', fontsize=18)
# plt.axhline(y=sigma, xmin=-100, xmax=100, c='k', ls='-.', linewidth=3)
# plt.axhline(y=-sigma, xmin=-100, xmax=100, c='k', ls='-.', linewidth=3)
# plt.grid()
# plt.show()

## ABS line fit & residuals
## Plot
plt.rc('font', weight='bold')
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rc('axes', linewidth=2)

# fig          = plt.figure(figsize=(8,8))
# ax           = fig.add_subplot(111); #ax.set_rasterized(True)                                 
# major_xticks = np.arange(-20., 20., 5.)                                              
# minor_xticks = np.arange(-20., 20., 1.)
# major_yticks = np.arange(0.5, 5., 0.5)                                              
# minor_yticks = np.arange(0.5, 5., 0.1)

fig1 = plt.figure(figsize=(10.5,12))

frame1=fig1.add_axes((.1,.4,.8,.56))
ymin = min(tfita)
ymax = max(tfita)
plt.plot(vlsr, tau, 'k-', linewidth=1, label='')
plt.plot(vlsr, tfita, 'k-', linewidth=2, label='')
plt.plot(vlsr, resida+0.998*ymin, 'k-', linewidth=1, label='')
# plt.title(src, fontsize=30)
plt.ylabel(r'$e^{-\tau}$', fontsize=20)
plt.xlim([-80., 80.])

plt.ylim([ymin-0.004*ymin, ymax+0.002*ymax])
# plt.axhline(y=0.991356, xmin=-100, xmax=100, c='k', ls='--', linewidth=1)
plt.tick_params(axis='y', labelsize=15)
plt.tick_params(axis='x', labelsize=18, labelbottom='off')
plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=5)

yy1 = 0.996
plt.text(-78.0, yy1, '$\mathrm{'+src+'}$', fontsize=14, fontweight='bold')
plt.text(-78.0, yy1-0.0005, '$\mathrm{(L,B)\ \ =(???.??,\ \ ???.??)}$', fontsize=14, fontweight='bold')

plt.text(-78.0, yy1-0.0015, '$\mathrm{HI}$', fontsize=14, fontweight='bold')
plt.text(-78.0, yy1-0.002, r'$\tau\ \ \ \ \ \ \ \ \ =[--,\ \ --]$', fontsize=14, fontweight='bold')
plt.text(-78.0, yy1-0.0025, '$\mathrm{VLSR}\ \ =[--,\ \ --]$', fontsize=14, fontweight='bold')
plt.text(-78.0, yy1-0.003, '$\mathrm{FWHM}=[--,\ \ --]$', fontsize=14, fontweight='bold')
plt.text(-78.0, yy1-0.0035, '$\mathrm{T_{ex}}\ \ \ \ \ \ =[--,\ \ --]$', fontsize=14, fontweight='bold')

# plt.legend(loc='upper left', fontsize=18)
plt.grid(False)

frame2=fig1.add_axes((.1,.1,.8,.294))
ymin = min(tb_tot_fit)
ymax = max(tb_tot_fit)
plt.plot(xdataem,tdata, 'k-', lw=1, label='')
plt.plot(xdataem, tb_tot_fit,'k-', lw=2, label='')
plt.plot(xdataem, reside-0.5, 'k-', linewidth=1, label='')
plt.ylabel(r'$\rm{T}_{\rm{exp}}\ (K)$',fontsize=20)
plt.xlabel('$\mathrm{VLSR\ (km/s)}$', fontsize=20)
plt.tick_params(axis='x', labelsize=18, pad=8)
plt.tick_params(axis='y', labelsize=15)
# plt.axhline(y=3.525-tbg, xmin=-100, xmax=100, c='k', ls='--', linewidth=1)
plt.xlim([-80., 80.])
plt.ylim([-1.0, ymax+0.01*ymax])
# plt.ylim([3.47-tbg, 3.595-tbg])
# plt.legend(loc='upper left', fontsize=18)
plt.grid(False)
plt.tick_params(which='both', width=2)
plt.tick_params(which='major', length=5)

# plt.tight_layout()
plt.savefig('abs_em_HI_'+src+'.png', format='png', dpi=400)
plt.show()