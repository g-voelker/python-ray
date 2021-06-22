import numpy as np


def mean_v(xx, zz, param):
    sigma, vref, ll, xcenter, zcenter = param
    return vref * np.exp(-((xx - xcenter)**2 + (zz - zcenter)**2) / 2 / sigma**2)
    # return vref * np.exp(-(zz - zcenter)**2 / 2 / sigma**2)


def omega(kk, ll, mm, bvf=0.01):
    return np.sqrt((bvf ** 2 * (kk**2 + ll**2) / mm**2))


def cgx(kk, ll, mm, bvf=0.01):
    om = omega(kk, ll, mm, bvf=bvf)
    return kk * bvf**2 / om / mm**2


def cgz(kk, ll, mm, bvf=0.01):
    om = omega(kk, ll, mm, bvf=bvf)
    return - bvf**2 / om * (kk**2 + ll**2) / mm**3


def dxdt(kk, mm, xx, zz, param, bvf=0.01):
    ll = param[2]
    return cgx(kk, ll, mm, bvf=bvf)


def dzdt(kk, mm, xx, zz, param, bvf=0.01):
    ll = param[2]
    return cgz(kk, ll, mm, bvf=bvf)


def dkdt(kk, mm, xx, zz, param, bvf=0.01):
    ll = param[2]
    dx = 1
    dvdx = (mean_v(xx + dx, zz, param) - mean_v(xx - dx, zz, param)) / 2 / dx
    return  - ll * dvdx

def dmdt(kk, mm, xx, zz, param, bvf=0.01):
    ll = param[2]
    dz = 1
    dvdz = (mean_v(xx, zz + dz, param) - mean_v(xx, zz - dz, param)) / 2 / dz
    return  - ll * dvdz


def rhs(var, param, bvf=0.01):
    xx, zz, kk, mm = var
    right_hand_side = np.array([dxdt(kk, mm, xx, zz, param, bvf=bvf),
                                dzdt(kk, mm, xx, zz, param, bvf=bvf),
                                dkdt(kk, mm, xx, zz, param, bvf=bvf),
                                dmdt(kk, mm, xx, zz, param, bvf=bvf)])
    return right_hand_side


def RK3(dt, var, param, bvf=0.01):

    qq = dt * rhs(var, param, bvf=bvf)
    var = var + qq / 3
    qq = dt * rhs(var, param, bvf=bvf) - 5 / 9 * qq
    var = var + 15 / 16 * qq
    qq = dt * rhs(var, param, bvf=bvf) - 153 / 128 * qq
    var = var + 8 / 15 * qq

    return var
