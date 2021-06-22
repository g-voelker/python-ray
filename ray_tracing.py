import numpy as np
import matplotlib.pyplot as plt
import lib as lib
import time

plt.style.use('ggplot')

start = time.time()

# set background parameters
NN = 0.01
sigma = 500
vref = 100
xcenter = 1500
zcenter = 1500
nray = 20

# set time axis
dt = 10
nt_max = 3600
tt = np.linspace(0, nt_max * dt, nt_max + 1)

# set initial values
xx_init = np.linspace(100, 2900, nray)
zz_init = 100
kk_init = 2 * np.pi / 5e3
ll = 2 * np.pi / 5e3
mm_init = -2 * np.pi / 5e3

# compile tuple of model parameters
param = (sigma, vref, ll, xcenter, zcenter)

# set arrays for integration
int_xx = np.zeros((nt_max + 1, nray))
int_zz = np.zeros((nt_max + 1, nray))
int_kk = np.zeros((nt_max + 1, nray))
int_mm = np.zeros((nt_max + 1, nray))

# set initial values in integration arrays
int_xx[0] = xx_init
int_zz[0] = zz_init
int_kk[0] = kk_init
int_mm[0] = mm_init

# integrate in time
for nt in range(1, nt_max + 1):
  
    # set the last model state
    state = np.array([int_xx[nt - 1],
                      int_zz[nt - 1],
                      int_kk[nt - 1],
                      int_mm[nt - 1]])

    # use the last model state to integrate to the new state
    state = lib.RK3(dt, state, param, bvf=NN)

    # set variables from model state
    int_xx[nt], int_zz[nt], int_kk[nt], int_mm[nt] = state

    # print out progress in %
    print('progress: {0:.2f}%'.format(nt / nt_max * 100), end='\r')

end = time.time()

print('time needed: {0:.2f}s'.format(end - start))


##
#  plot trajectory

# set spatial arrays for plot of background jet
xx_bg = np.linspace(0, 2 * xcenter, 100) * np.ones((100, 100))
zz_bg = (np.linspace(0, 2 * xcenter, 100) * np.ones((100, 100))).T

# calculate background jet
# v0 = vref * np.exp(-((xx_bg - xcenter)**2 + (zz_bg - zcenter)**2) / 2 / sigma**2)


fig, ax = plt.subplots(1, 3, figsize=(10, 4))

# plot wavenumber evolution
ax[0].plot(tt / 3600, int_kk)
ax[1].plot(tt / 3600, int_mm)

# plot jet and trajectory above
image = ax[2].pcolormesh(xx_bg / 1000, zz_bg / 1000, lib.mean_v(xx_bg, zz_bg, param), vmin=-np.abs(vref), vmax=np.abs(vref),
                         shading='gouraud')
ax[2].plot(int_xx / 1000, int_zz / 1000, '-', color='w')

# set limits for map of trajectory
ax[2].set_xlim(0, 2 * xcenter / 1000)
ax[2].set_ylim(0, 2 * zcenter / 1000)

# set labels
ax[0].set_xlabel('t (h)')
ax[0].set_ylabel('wavenumber $k$ (m$^{-1}$)')
ax[1].set_xlabel('t (h)')
ax[1].set_ylabel('wavenumber $m$ (m$^{-1}$)')
ax[2].set_xlabel('x (km)')
ax[2].set_ylabel('z (km)')

# some layout settings and colorbar workaround
fig.tight_layout(rect=(0, 0, 1.05, 1))
plt.colorbar(image, ax=list(ax), label='background velocity $V_0$ (m/s)')
fig.tight_layout(rect=(0, 0, .88, 1), w_pad=0)

# output / show plot
plt.show()



