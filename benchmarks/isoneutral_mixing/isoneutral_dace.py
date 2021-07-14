import numpy as np
import dace as dc


# Symbols
S1 = dc.symbol('S1', dtype=dc.int32)  # math.ceil(2 * size ** (1/3))
S2 = dc.symbol('S2', dtype=dc.int32)  # math.ceil(0.25 * size ** (1/3))


@dc.program
def get_drhodT(salt: dc.float64, temp:dc.float64, p: dc.float64):
    rho0 = 1024.0
    z0 = 0.0
    theta0 = 283.0 - 273.15
    grav = 9.81
    betaT = 1.67e-4
    betaTs = 1e-5
    gammas = 1.1e-8

    zz = -p - z0
    thetas = temp - theta0
    return -(betaTs * thetas + betaT * (1 - gammas * grav * zz * rho0)) * rho0


@dc.program
def get_drhodS(salt: dc.float64, temp: dc.float64, p: dc.float64):
    betaS = 0.78e-3
    rho0 = 1024.
    return betaS * rho0


@dc.program
def dm_taper(sx: dc.float64):
    """
    tapering function for isopycnal slopes
    """
    iso_slopec = 1e-3
    iso_dslope = 1e-3
    return 0.5 * (1. + np.tanh((-np.abs(sx) + iso_slopec) / iso_dslope))


@dc.program
def isoneutral_diffusion_pre(maskT: dc.float64[S1, S1, S2],
                             maskU: dc.float64[S1, S1, S2],
                             maskV: dc.float64[S1, S1, S2],
                             maskW: dc.float64[S1, S1, S2],
                             dxt: dc.float64[S1],
                             dxu: dc.float64[S1],
                             dyt: dc.float64[S1],
                             dyu: dc.float64[S1],
                             dzt: dc.float64[S2],
                             dzw: dc.float64[S2],
                             cost: dc.float64[S1],
                             cosu: dc.float64[S1],
                             salt: dc.float64[S1, S1, S2, 3],
                             temp: dc.float64[S1, S1, S2, 3],
                             zt: dc.float64[S2],
                             K_iso: dc.float64[S1, S1, S2],
                             K_11: dc.float64[S1, S1, S2],
                             K_22: dc.float64[S1, S1, S2],
                             K_33: dc.float64[S1, S1, S2],
                             Ai_ez: dc.float64[S1, S1, S2, 2, 2],
                             Ai_nz: dc.float64[S1, S1, S2, 2, 2],
                             Ai_bx: dc.float64[S1, S1, S2, 2, 2],
                             Ai_by: dc.float64[S1, S1, S2, 2, 2]):
    """
    Isopycnal diffusion for tracer
    following functional formulation by Griffies et al
    Code adopted from MOM2.1
    """
    nx, ny, nz = maskT.shape

    epsln = 1e-20
    # iso_slopec = 1e-3
    # iso_dslope = 1e-3
    K_iso_steep = 50.
    tau = 0

    drdT = np.empty_like(K_11)
    drdS = np.empty_like(K_11)
    dTdx = np.empty_like(K_11)
    dSdx = np.empty_like(K_11)
    dTdy = np.empty_like(K_11)
    dSdy = np.empty_like(K_11)
    dTdz = np.empty_like(K_11)
    dSdz = np.empty_like(K_11)

    """
    drho_dt and drho_ds at centers of T cells
    """
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                drdT[i, j, k] = maskT[i, j, k] * get_drhodT(
                    salt[i, j, k, tau], temp[i, j, k, tau], np.abs(zt[k])
                )
                drdS[i, j, k] = maskT[i, j, k] * get_drhodS(
                    salt[i, j, k, tau], temp[i, j, k, tau], np.abs(zt[k])
                )

    """
    gradients at top face of T cells
    """
    for i in range(nx):
        for j in range(ny):
            for k in range(nz-1):
                dTdz[i, j, k] = maskW[i, j, k] * \
                    (temp[i, j, k+1, tau] - temp[i, j, k, tau]) / \
                    dzw[k]
                dSdz[i, j, k] = maskW[i, j, k] * \
                    (salt[i, j, k+1, tau] - salt[i, j, k, tau]) / \
                    dzw[k]
            dTdz[i, j, -1] = 0.
            dSdz[i, j, -1] = 0.

    """
    gradients at eastern face of T cells
    """
    for i in range(nx-1):
        for j in range(ny):
            for k in range(nz):
                dTdx[i, j, k] = maskU[i, j, k] * (temp[i+1, j, k, tau] - temp[i, j, k, tau]) \
                    / (dxu[i] * cost[j])
                dSdx[i, j, k] = maskU[i, j, k] * (salt[i+1, j, k, tau] - salt[i, j, k, tau]) \
                    / (dxu[i] * cost[j])
    dTdx[-1, :, :] = 0.
    dSdx[-1, :, :] = 0.

    """
    gradients at northern face of T cells
    """
    for i in range(nx):
        for j in range(ny-1):
            for k in range(nz):
                dTdy[i, j, k] = maskV[i, j, k] * (temp[i, j+1, k, tau] - temp[i, j, k, tau]) \
                    / dyu[j]
                dSdy[i, j, k] = maskV[i, j, k] * (salt[i, j+1, k, tau] - salt[i, j, k, tau]) \
                    / dyu[j]
    dTdy[:, -1, :] = 0.
    dSdy[:, -1, :] = 0.

    """
    Compute Ai_ez and K11 on center of east face of T cell.
    """
    for i in range(1, nx-2):
        for j in range(2, ny-2):
            for k in range(0, nz):
                if k == 0:
                    diffloc = 0.5 * (K_iso[i, j, k] + K_iso[i+1, j, k])
                else:
                    diffloc = 0.25 * (K_iso[i, j, k] + K_iso[i, j, k-1] + K_iso[i+1, j, k] + K_iso[i+1, j, k-1])

                sumz = 0.

                # for kr in (0, 1):
                for kr in range(2):
                    if k == 0 and kr == 0:
                        continue

                    # for ip in (0, 1):
                    for ip in range(2):
                        drodxe = drdT[i + ip, j, k] * dTdx[i, j, k] \
                            + drdS[i + ip, j, k] * dSdx[i, j, k]
                        drodze = drdT[i + ip, j, k] * dTdz[i + ip, j, k + kr - 1] \
                            + drdS[i + ip, j, k] * dSdz[i + ip, j, k + kr - 1]
                        sxe = -drodxe / (min(0., drodze) - epsln)
                        taper = dm_taper(sxe)
                        sumz += dzw[k + kr - 1] * maskU[i, j, k] * max(K_iso_steep, diffloc * taper)
                        Ai_ez[i, j, k, ip, kr] = taper * sxe * maskU[i, j, k]

                K_11[i, j, k] = sumz / (4. * dzt[k])

    """
    Compute Ai_nz and K_22 on center of north face of T cell.
    """
    for i in range(2, nx-2):
        for j in range(1, ny-2):
            for k in range(0, nz):
                if k == 0:
                    diffloc = 0.5 * (K_iso[i, j, k] + K_iso[i, j+1, k])
                else:
                    diffloc = 0.25 * (K_iso[i, j, k] + K_iso[i, j, k-1] + K_iso[i, j+1, k] + K_iso[i, j+1, k-1])

                sumz = 0.

                # for kr in (0, 1):
                for kr in range(2):
                    if k == 0 and kr == 0:
                        continue

                    # for jp in (0, 1):
                    for jp in range(2):
                        drodyn = drdT[i, j + jp, k] * dTdy[i, j, k] \
                            + drdS[i, j + jp, k] * dSdy[i, j, k]
                        drodzn = drdT[i, j + jp, k] * dTdz[i, j + jp, k + kr - 1] \
                            + drdS[i, j + jp, k] * dSdz[i, j + jp, k + kr - 1]
                        syn = -drodyn / (min(0., drodzn) - epsln)
                        taper = dm_taper(syn)
                        sumz += dzw[k + kr - 1] * maskV[i, j, k] * max(K_iso_steep, diffloc * taper)
                        Ai_nz[i, j, k, jp, kr] = taper * syn * maskV[i, j, k]

                K_22[i, j, k] = sumz / (4. * dzt[k])

    """
    compute Ai_bx, Ai_by and K33 on top face of T cell.
    """
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            for k in range(nz-1):
                sumx = 0.
                sumy = 0.

                # for kr in (0, 1):
                for kr in range(2):
                    drodzb = drdT[i, j, k + kr] * dTdz[i, j, k] \
                        + drdS[i, j, k + kr] * dSdz[i, j, k]

                    # eastward slopes at the top of T cells
                    # for ip in (0, 1):
                    for ip in range(2):
                        drodxb = drdT[i, j, k + kr] * dTdx[i - 1 + ip, j, k + kr] \
                            + drdS[i, j, k + kr] * dSdx[i - 1 + ip, j, k + kr]
                        sxb = -drodxb / (min(0., drodzb) - epsln)
                        taper = dm_taper(sxb)
                        sumx += dxu[i - 1 + ip] * K_iso[i, j, k] * taper * sxb**2 * maskW[i, j, k]
                        Ai_bx[i, j, k, ip, kr] = taper * sxb * maskW[i, j, k]

                    # northward slopes at the top of T cells
                    # for jp in (0, 1):
                    for jp in range(2):
                        facty = cosu[j - 1 + jp] * dyu[j - 1 + jp]
                        drodyb = drdT[i, j, k + kr] * dTdy[i, j + jp - 1, k + kr] \
                            + drdS[i, j, k + kr] * dSdy[i, j + jp - 1, k + kr]
                        syb = -drodyb / (min(0., drodzb) - epsln)
                        taper = dm_taper(syb)
                        sumy += facty * K_iso[i, j, k] * taper * syb**2 * maskW[i, j, k]
                        Ai_by[i, j, k, jp, kr] = taper * syb * maskW[i, j, k]

                K_33[i, j, k] = sumx / (4 * dxt[i]) + sumy / (4 * dyt[j] * cost[j])

            K_33[i, j, -1] = 0.


# Avoiding specialization of symbols ...
cpu_sdfg = isoneutral_diffusion_pre.to_sdfg(strict=False)
import copy
gpu_sdfg = copy.deepcopy(cpu_sdfg)
from dace.transformation.auto import auto_optimize as autoopt
opt_cpu_sdfg = autoopt.auto_optimize(cpu_sdfg, dc.dtypes.DeviceType.CPU, symbols=None)
# opt_gpu_sdfg = autoopt.auto_optimize(gpu_sdfg, dc.dtypes.DeviceType.GPU, symbols=None)
# Make CPU sdfg single-threaded
for node, _ in opt_cpu_sdfg.all_nodes_recursive():
# for node, _ in cpu_sdfg.all_nodes_recursive():
    if isinstance(node, dc.nodes.MapEntry):
        node.map.schedule = dc.dtypes.ScheduleType.Sequential
# # Move arrays to the device (GPU only)
# for k, v in opt_gpu_sdfg.arrays.items():
#     if not v.transient and type(v) == dc.data.Array:
#         v.storage = dc.dtypes.StorageType.GPU_Global
cpu_exec = opt_cpu_sdfg.compile()
# cpu_exec = cpu_sdfg.compile()
# try:
#     gpu_exec = opt_gpu_sdfg.compile()
# except:
#     gpu_exec = None


def prepare_inputs(*inputs, device):
    if device == 'cpu':
        return inputs
    elif device == 'gpu':
        import cupy as cp
        out = [cp.asarray(k) for k in inputs]
        cp.cuda.stream.get_current_stream().synchronize()
        return out


input_names = ['maskT', 'maskU', 'maskV', 'maskW',
        'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'dzw',
        'cost', 'cosu',
        'salt', 'temp', 'zt',
        'K_iso', 'K_11', 'K_22', 'K_33', 'Ai_ez', 'Ai_nz', 'Ai_bx', 'Ai_by']


def run(*inputs, device='cpu'):
    # isoneutral_diffusion_pre(*inputs)
    input_dict = {k: v for k, v in zip(input_names, inputs)}
    if device == 'cpu':
        cpu_exec(**input_dict, S1=inputs[0].shape[0], S2=inputs[0].shape[-1])
    # elif device == 'gpu':
    #     gpu_exec(*inputs, S1=inputs[0].shape[0], S2=inputs[0].shape[-1])
    return inputs[-7:]
