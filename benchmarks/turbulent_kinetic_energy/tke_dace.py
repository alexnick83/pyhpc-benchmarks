import numpy as np
from scipy.linalg import lapack
import dace as dc


# Symbols
N1 = dc.symbol('N1', dtype=dc.int32)
N2 = dc.symbol('N2', dtype=dc.int32)
S1 = dc.symbol('S1', dtype=dc.int32)  # math.ceil(2 * size ** (1/3))
S2 = dc.symbol('S2', dtype=dc.int32)  # math.ceil(0.25 * size ** (1/3))


def where(mask, a, b):
    return np.where(mask, a, b)


@dc.program
def arange(start: dc.int64, X: dc.int64[N2]):
    for i in dc.map[0:N2]:
        X[i] = start + i


@dc.program
def solve_tridiag(a: dc.float64[N1, N1, N2],
                  b: dc.float64[N1, N1, N2],
                  c: dc.float64[N1, N1, N2],
                  d: dc.float64[N1, N1, N2]):
    """
    Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
    """
    assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape

    # n = a.shape[0]

    # for i in range(1, n):
    for i in range(1, a.shape[0]):
        w = a[i] / b[i - 1]
        b[i] += -w * c[i - 1]
        d[i] += -w * d[i - 1]

    out = np.empty_like(a)
    out[-1] = d[-1] / b[-1]

    # for i in range(n - 2, -1, -1):
    for i in range(a.shape[0] - 2, -1, -1):
        out[i] = (d[i] - c[i] * out[i + 1]) / b[i]

    return out


@dc.program
def solve_implicit(ks: dc.int64[N1, N1],
                   a: dc.float64[N1, N1, N2],
                   b: dc.float64[N1, N1, N2],
                   c: dc.float64[N1, N1, N2],
                   d: dc.float64[N1, N1, N2],
                   b_edge: dc.float64[N1, N1, N2]):
    land_mask = (ks >= 0)[:, :, np.newaxis]
    tmprng = np.ndarray(a.shape[2], dtype=np.int64)
    arange(0, tmprng)
    # edge_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
    #                          == ks[:, :, np.newaxis])
    # water_mask = land_mask & (np.arange(a.shape[2])[np.newaxis, np.newaxis, :]
    #                           >= ks[:, :, np.newaxis])
    edge_mask = land_mask & (tmprng[np.newaxis, np.newaxis, :]
                             == ks[:, :, np.newaxis])
    water_mask = land_mask & (tmprng[np.newaxis, np.newaxis, :]
                              >= ks[:, :, np.newaxis])

    a_tri = water_mask * a * np.logical_not(edge_mask)
    b_tri = np.where(water_mask, b, 1.)
    # if b_edge is not None:
    b_tri[:] = np.where(edge_mask, b_edge, b_tri)
    c_tri = water_mask * c
    d_tri = water_mask * d
    # if d_edge is not None:
    #     d_tri[:] = np.where(edge_mask, d_edge, d_tri)

    return solve_tridiag(a_tri, b_tri, c_tri, d_tri), water_mask


# def solve_tridiag(a, b, c, d):
#     """
#     Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.
#     """
#     assert a.shape == b.shape and a.shape == c.shape and a.shape == d.shape
#     a[..., 0] = c[..., -1] = 0  # remove couplings between slices
#     return lapack.dgtsv(a.flatten()[1:], b.flatten(), c.flatten()[:-1], d.flatten())[3].reshape(a.shape)


@dc.program
def _calc_cr(rjp: dc.float64,
             rj: dc.float64,
             rjm: dc.float64,
             vel: dc.float64):
    """
    Calculates cr value used in superbee advection scheme
    """
    eps = 1e-20  # prevent division by 0
    return np.where(vel > 0., rjm, rjp) / np.where(np.abs(rj) < eps, eps, rj)


def pad_z_edges(arr):
    arr_shape = list(arr.shape)
    arr_shape[2] += 2
    out = np.zeros(arr_shape, arr.dtype)
    out[:, :, 1:-1] = arr
    return out


@dc.program
def limiter(cr: dc.float64):
    return np.maximum(0., np.maximum(np.minimum(1., 2 * cr), np.minimum(2., cr)))


# @dc.program
# def _adv_superbee(vel: dc.float64[S1, S1, S2],
#                   var: dc.float64[S1, S1, S2],
#                   mask: dc.float64[S1, S1, S2],
#                   dx: dc.float64[S1],
#                   axis: dc.int32,
#                   cost: dc.float64[S1, S1, S2],
#                   cosu: dc.float64[S1, S1, S2],
#                   dt_tracer: dc.float64):
#     velfac = 1
#     if axis == 0:
#         sm1, s, sp1, sp2 = ((slice(1 + n, -2 + n or None), slice(2, -2), slice(None))
#                             for n in range(-1, 3))
#         dx = cost[np.newaxis, 2:-2, np.newaxis] * \
#             dx[1:-2, np.newaxis, np.newaxis]
#     elif axis == 1:
#         sm1, s, sp1, sp2 = ((slice(2, -2), slice(1 + n, -2 + n or None), slice(None))
#                             for n in range(-1, 3))
#         dx = (cost * dx)[np.newaxis, 1:-2, np.newaxis]
#         velfac = cosu[np.newaxis, 1:-2, np.newaxis]
#     elif axis == 2:
#         vel, var, mask = (pad_z_edges(a) for a in (vel, var, mask))
#         sm1, s, sp1, sp2 = ((slice(2, -2), slice(2, -2), slice(1 + n, -2 + n or None))
#                             for n in range(-1, 3))
#         dx = dx[np.newaxis, np.newaxis, :-1]
#     else:
#         raise ValueError('axis must be 0, 1, or 2')
#     uCFL = np.abs(velfac * vel[s] * dt_tracer / dx)
#     rjp = (var[sp2] - var[sp1]) * mask[sp1]
#     rj = (var[sp1] - var[s]) * mask[s]
#     rjm = (var[s] - var[sm1]) * mask[sm1]
#     cr = limiter(_calc_cr(rjp, rj, rjm, vel[s]))
#     return velfac * vel[s] * (var[sp1] + var[s]) * 0.5 - np.abs(velfac * vel[s]) * ((1. - cr) + uCFL * cr) * rj * 0.5


@dc.program
def adv_flux_superbee_wgrid(adv_fe: dc.float64[S1, S1, S2],
                            adv_fn: dc.float64[S1, S1, S2],
                            adv_ft: dc.float64[S1, S1, S2],
                            var: dc.float64[S1, S1, S2],
                            u_wgrid: dc.float64[S1, S1, S2],
                            v_wgrid: dc.float64[S1, S1, S2],
                            w_wgrid: dc.float64[S1, S1, S2],
                            maskW: dc.float64[S1, S1, S2],
                            dxt: dc.float64[S1],
                            dyt: dc.float64[S1],
                            dzw: dc.float64[S2],
                            cost: dc.float64[S1],
                            cosu: dc.float64[S1],
                            dt_tracer: dc.int64):
    """
    Calculates advection of a tracer defined on Wgrid
    """
    nx, ny, nz = var.shape

    maskUtr = np.zeros_like(maskW)
    maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]

    adv_fe[...] = 0.
    for i in range(1, nx-2):
        for j in range(2, ny-2):
            for k in range(nz):
                vel = u_wgrid[i, j, k]
                u_cfl = np.abs(vel * dt_tracer / (cost[j] * dxt[i]))
                r_jp = (var[i+2, j, k] - var[i+1, j, k]) * maskUtr[i+1, j, k]
                r_j = (var[i+1, j, k] - var[i, j, k]) * maskUtr[i, j ,k]
                r_jm = (var[i, j, k] - var[i-1, j, k]) * maskUtr[i-1, j, k]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, vel))
                adv_fe[i, j, k] = vel * (var[i+1, j, k] + var[i, j, k]) * 0.5 - np.abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5

    maskVtr = np.zeros_like(maskW)
    maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]

    adv_fn[...] = 0.
    for i in range(2, nx-2):
        for j in range(1, ny-2):
            for k in range(nz):
                vel = cosu[j] * v_wgrid[i, j, k]
                u_cfl = np.abs(vel * dt_tracer / (cost[j] * dyt[j]))
                r_jp = (var[i, j+2, k] - var[i, j+1, k]) * maskVtr[i, j+1, k]
                r_j = (var[i, j+1, k] - var[i, j, k]) * maskVtr[i, j, k]
                r_jm = (var[i, j, k] - var[i, j-1, k]) * maskVtr[i, j-1, k]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, v_wgrid[i, j, k]))
                adv_fn[i, j, k] = vel * (var[i, j+1, k] + var[i, j, k]) * 0.5 - np.abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5

    maskWtr = np.zeros_like(maskW)
    maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]

    adv_ft[...] = 0.
    for i in range(2, nx-2):
        for j in range(2, ny-2):
            for k in range(nz-1):
                kp1 = min(nz-2, k+1)
                kp2 = min(nz-1, k+2)
                km1 = max(0, k-1)

                vel = w_wgrid[i, j, k]
                u_cfl = np.abs(vel * dt_tracer / dzw[k])
                r_jp = (var[i, j, kp2] - var[i, j, k+1]) * maskWtr[i, j, kp1]
                r_j = (var[i, j, k+1] - var[i, j, k]) * maskWtr[i, j ,k]
                r_jm = (var[i, j, k] - var[i, j, km1]) * maskWtr[i, j, km1]
                cr = limiter(_calc_cr(r_jp, r_j, r_jm, vel))
                adv_ft[i, j, k] = vel * (var[i, j, k+1] + var[i, j, k]) * 0.5 - np.abs(vel) * ((1. - cr) + u_cfl * cr) * r_j * 0.5


# @dc.program
# def adv_flux_superbee_wgrid(adv_fe: dc.float64[S1, S1, S2],
#                             adv_fn: dc.float64[S1, S1, S2],
#                             adv_ft: dc.float64[S1, S1, S2],
#                             var: dc.float64[S1, S1, S2],
#                             u_wgrid: dc.float64[S1, S1, S2],
#                             v_wgrid: dc.float64[S1, S1, S2],
#                             w_wgrid: dc.float64[S1, S1, S2],
#                             maskW: dc.float64[S1, S1, S2],
#                             dxt: dc.float64[S1],
#                             dyt: dc.float64[S1],
#                             dzw: dc.float64[S2],
#                             cost: dc.float64[S1],
#                             cosu: dc.float64[S1],
#                             dt_tracer: dc.float64):
#     """
#     Calculates advection of a tracer defined on Wgrid
#     """
#     maskUtr = np.zeros_like(maskW)
#     maskUtr[:-1, :, :] = maskW[1:, :, :] * maskW[:-1, :, :]
#     adv_fe[...] = 0.
#     adv_fe[1:-2, 2:-2, :] = _adv_superbee(u_wgrid, var, maskUtr, dxt, 0, cost, cosu, dt_tracer)

#     maskVtr = np.zeros_like(maskW)
#     maskVtr[:, :-1, :] = maskW[:, 1:, :] * maskW[:, :-1, :]
#     adv_fn[...] = 0.
#     adv_fn[2:-2, 1:-2, :] = _adv_superbee(v_wgrid, var, maskVtr, dyt, 1, cost, cosu, dt_tracer)

#     maskWtr = np.zeros_like(maskW)
#     maskWtr[:, :, :-1] = maskW[:, :, 1:] * maskW[:, :, :-1]
#     adv_ft[...] = 0.
#     adv_ft[2:-2, 2:-2, :-1] = _adv_superbee(w_wgrid, var, maskWtr, dzw, 2, cost, cosu, dt_tracer)


@dc.program
def integrate_tke(u: dc.float64[S1, S1, S2, 3],
                  v: dc.float64[S1, S1, S2, 3],
                  w: dc.float64[S1, S1, S2, 3],
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
                  kbot: dc.int64[S1, S1],
                  kappaM: dc.float64[S1, S1, S2],
                  mxl: dc.float64[S1, S1, S2],
                  forc: dc.float64[S1, S1, S2],
                  forc_tke_surface: dc.float64[S1, S1],
                  tke: dc.float64[S1, S1, S2, 3],
                  dtke: dc.float64[S1, S1, S2, 3]):
    tau = 0
    taup1 = 1
    taum1 = 2

    dt_tracer = 1
    dt_mom = 1
    AB_eps = 0.1
    alpha_tke = 1.
    c_eps = 0.7
    K_h_tke = 2000.

    flux_east = np.zeros_like(maskU)
    flux_north = np.zeros_like(maskU)
    flux_top = np.zeros_like(maskU)

    sqrttke = np.sqrt(np.maximum(0., tke[:, :, :, tau]))

    """
    integrate Tke equation on W grid with surface flux boundary condition
    """
    dt_tke = dt_mom  # use momentum time step to prevent spurious oscillations

    """
    vertical mixing and dissipation of TKE
    """
    ks = kbot[2:-2, 2:-2] - 1

    # a_tri = np.zeros_like(maskU[2:-2, 2:-2])
    # b_tri = np.zeros_like(maskU[2:-2, 2:-2])
    # c_tri = np.zeros_like(maskU[2:-2, 2:-2])
    # d_tri = np.zeros_like(maskU[2:-2, 2:-2])
    # delta = np.zeros_like(maskU[2:-2, 2:-2])
    a_tri = np.zeros((S1 - 4,S1 - 4,S2))
    b_tri = np.zeros((S1 - 4,S1 - 4,S2))
    c_tri = np.zeros((S1 - 4,S1 - 4,S2))
    d_tri = np.zeros((S1 - 4,S1 - 4,S2))
    delta = np.zeros((S1 - 4,S1 - 4,S2))


    delta[:, :, :-1] = dt_tke / dzt[np.newaxis, np.newaxis, 1:] * alpha_tke * 0.5 \
        * (kappaM[2:-2, 2:-2, :-1] + kappaM[2:-2, 2:-2, 1:])

    a_tri[:, :, 1:-1] = -delta[:, :, :-2] / \
        dzw[np.newaxis, np.newaxis, 1:-1]
    a_tri[:, :, -1] = -delta[:, :, -2] / (0.5 * dzw[-1])

    b_tri[:, :, 1:-1] = 1 + (delta[:, :, 1:-1] + delta[:, :, :-2]) / dzw[np.newaxis, np.newaxis, 1:-1] \
        + dt_tke * c_eps \
        * sqrttke[2:-2, 2:-2, 1:-1] / mxl[2:-2, 2:-2, 1:-1]
    b_tri[:, :, -1] = 1 + delta[:, :, -2] / (0.5 * dzw[-1]) \
        + dt_tke * c_eps / mxl[2:-2, 2:-
                                     2, -1] * sqrttke[2:-2, 2:-2, -1]
    b_tri_edge = 1 + delta / dzw[np.newaxis, np.newaxis, :] \
        + dt_tke * c_eps / mxl[2:-2, 2:-2, :] * sqrttke[2:-2, 2:-2, :]

    c_tri[:, :, :-1] = -delta[:, :, :-1] / dzw[np.newaxis, np.newaxis, :-1]

    d_tri[...] = tke[2:-2, 2:-2, :, tau] + dt_tke * forc[2:-2, 2:-2, :]
    d_tri[:, :, -1] += dt_tke * forc_tke_surface[2:-2, 2:-2] / (0.5 * dzw[-1])

    sol, water_mask = solve_implicit(ks, a_tri, b_tri, c_tri, d_tri, b_tri_edge)
    tke[2:-2, 2:-2, :, taup1] = np.where(water_mask, sol, tke[2:-2, 2:-2, :, taup1])

    """
    Add TKE if surface density flux drains TKE in uppermost box
    """
    # tke_surf_corr = np.zeros(maskU.shape[:2])
    tke_surf_corr = np.zeros((S1, S1))
    mask = tke[2:-2, 2:-2, -1, taup1] < 0.0
    tke_surf_corr[2:-2, 2:-2] = np.where(
        mask,
        -tke[2:-2, 2:-2, -1, taup1] * 0.5 * dzw[-1] / dt_tke,
        0.
    )
    tke[2:-2, 2:-2, -1, taup1] = np.maximum(0., tke[2:-2, 2:-2, -1, taup1])

    """
    add tendency due to lateral diffusion
    """
    flux_east[:-1, :, :] = K_h_tke * (tke[1:, :, :, tau] - tke[:-1, :, :, tau]) \
        / (cost[np.newaxis, :, np.newaxis] * dxu[:-1, np.newaxis, np.newaxis]) * maskU[:-1, :, :]
    flux_east[-1, :, :] = 0.
    flux_north[:, :-1, :] = K_h_tke * (tke[:, 1:, :, tau] - tke[:, :-1, :, tau]) \
        / dyu[np.newaxis, :-1, np.newaxis] * maskV[:, :-1, :] * cosu[np.newaxis, :-1, np.newaxis]
    flux_north[:, -1, :] = 0.
    tke[2:-2, 2:-2, :, taup1] += dt_tke * maskW[2:-2, 2:-2, :] * \
        ((flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
            + (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
            / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))

    """
    add tendency due to advection
    """
    adv_flux_superbee_wgrid(
        flux_east, flux_north, flux_top, tke[:, :, :, tau],
        u[..., tau], v[..., tau], w[..., tau], maskW, dxt, dyt, dzw,
        cost, cosu, dt_tracer
    )

    dtke[2:-2, 2:-2, :, tau] = maskW[2:-2, 2:-2, :] * (-(flux_east[2:-2, 2:-2, :] - flux_east[1:-3, 2:-2, :])
                                                                / (cost[np.newaxis, 2:-2, np.newaxis] * dxt[2:-2, np.newaxis, np.newaxis])
                                                                - (flux_north[2:-2, 2:-2, :] - flux_north[2:-2, 1:-3, :])
                                                                / (cost[np.newaxis, 2:-2, np.newaxis] * dyt[np.newaxis, 2:-2, np.newaxis]))
    dtke[:, :, 0, tau] += -flux_top[:, :, 0] / dzw[0]
    dtke[:, :, 1:-1, tau] += - \
        (flux_top[:, :, 1:-1] - flux_top[:, :, :-2]) / dzw[1:-1]
    dtke[:, :, -1, tau] += - \
        (flux_top[:, :, -1] - flux_top[:, :, -2]) / \
        (0.5 * dzw[-1])

    """
    Adam Bashforth time stepping
    """
    tke[:, :, :, taup1] += dt_tracer * ((1.5 + AB_eps) * dtke[:, :, :, tau] - (0.5 + AB_eps) * dtke[:, :, :, taum1])

    return tke, dtke, tke_surf_corr


# Avoiding specialization of symbols ...
cpu_sdfg = integrate_tke.to_sdfg(strict=False)
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


input_names = ['u', 'v', 'w',
        'maskU', 'maskV', 'maskW',
        'dxt', 'dxu', 'dyt', 'dyu', 'dzt', 'dzw',
        'cost', 'cosu',
        'kbot',
        'kappaM', 'mxl', 'forc',
        'forc_tke_surface',
        'tke', 'dtke']


def run(*inputs, device='cpu'):
    # outputs = integrate_tke(*inputs)
    input_dict = {k: v for k, v in zip(input_names, inputs)}
    if device == 'cpu':
        outputs = cpu_exec(**input_dict, S1=inputs[0].shape[0], S2=inputs[0].shape[-1])
    # elif device == 'gpu':
    #     outputs = gpu_exec(*inputs, S1=inputs[0].shape[0], S2=inputs[0].shape[-1])
    return outputs


# def run(*inputs, device='cpu'):
#     outputs = integrate_tke(*inputs)
#     return outputs
