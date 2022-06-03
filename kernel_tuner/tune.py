from helpers import Grid, Fields, tune_and_store
import numpy as np
import sys
from pprint import pprint

def tune_advec(axis, grid):
    fields = Fields.from_grid(grid, ["u", "v", "w"])

    args = [
        fields.u.tend, fields.u.fld,
        fields.v.fld, fields.w.fld,
        fields.rhoref, fields.rhorefh,
        1.0 / fields.rhoref, 1.0 / fields.rhorefh,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend
    ]

    kernel_name = f"advec_{axis}_wrapper"
    kernel_source = "advec_2i5.cu"
    params = dict(REWRITE_FINITE_DIFFERENCE=[0], USE_RECIPROCAL=[0, 1])

    return tune_and_store(grid, args, kernel_name, kernel_source, extra_params=params)


def tune_advec_u(grid):
    return tune_advec('u', grid)


def tune_advec_v(grid):
    return tune_advec('v', grid)


def tune_advec_w(grid):
    return tune_advec('w', grid)


def tune_advec_uvw(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w"])

    args = [
        fields.u.tend, fields.v.tend, fields.w.tend,
        fields.u.fld, fields.v.fld, fields.w.fld,
        fields.rhoref, fields.rhorefh,
        1.0 / fields.rhoref, 1.0 / fields.rhorefh,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend
    ]

    kernel_name = f"advec_uvw_wrapper"
    kernel_source = "advec_2i5.cu"
    params = dict(REWRITE_FINITE_DIFFERENCE=[0], FUSE_BLOCKS=[0, 1], USE_RECIPROCAL=[0, 1])

    return tune_and_store(grid, args, kernel_name, kernel_source, extra_params=params)


def tune_advec_uvw_interior(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w"])
    border = np.int32(3)

    args = [
        fields.u.tend, fields.v.tend, fields.w.tend,
        fields.u.fld, fields.v.fld, fields.w.fld,
        fields.rhoref, fields.rhorefh,
        1.0 / fields.rhoref, 1.0 / fields.rhorefh,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart + border,
        grid.iend, grid.jend, grid.kend - border
    ]

    kernel_name = "advec_uvw_interior_wrapper"
    kernel_source = "advec_2i5.cu"
    params = dict(REWRITE_FINITE_DIFFERENCE=[0], FUSE_BLOCKS=[0, 1], USE_RECIPROCAL=[0, 1])
    nlayers = grid.ktot - 2 * border

    return tune_and_store(grid, args, kernel_name, kernel_source, extra_params=params, nlayers=nlayers)



def tune_advec_s(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w", "s"])

    args = [
        fields.s.tend, fields.s.fld,
        fields.u.fld, fields.v.fld, fields.w.fld,
        fields.rhoref, fields.rhorefh,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend
    ]

    kernel_name = "advec_s_wrapper"
    kernel_source = "advec_2i5.cu"
    params = dict(REWRITE_FINITE_DIFFERENCE=[0, 1])

    return tune_and_store(grid, args, kernel_name, kernel_source, extra_params=params)


def tune_advec_s_lim(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w", "s"])

    args = [
        fields.s.tend, fields.s.fld,
        fields.u.fld, fields.v.fld, fields.w.fld,
        fields.rhoref, fields.rhorefh,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend
    ]

    kernel_name = "advec_s_lim_wrapper"
    kernel_source = "advec_2i5.cu"
    params = dict(REWRITE_FINITE_DIFFERENCE=[0, 1])

    return tune_and_store(grid, args, kernel_name, kernel_source, extra_params=params)


def tune_advec_sn(n):
    def tune(grid):
        fields = Fields.from_grid(grid, ["u", "v", "w"] + [f"s{i}" for i in range(n)])

        args = \
            [getattr(fields, f"s{i}").tend for i in range(n)] + \
            [getattr(fields, f"s{i}").fld for i in range(n)] + \
            [fields.u.fld, fields.v.fld, fields.w.fld,
             fields.rhoref, fields.rhorefh,
             grid.dzi, grid.dxi, grid.dyi,
             grid.icells, grid.ijcells,
             grid.istart, grid.jstart, grid.kstart,
             grid.iend, grid.jend, grid.kend
        ]

        key = f"advec_s_{n}"
        kernel_name = "advec_s_multi_wrapper"
        kernel_source = "advec_2i5.cu"
        params = dict(REWRITE_FINITE_DIFFERENCE=[0, 1], ADVEC_S_MULTI_SIZE=[n], FUSE_BLOCKS=[0, 1])

        return tune_and_store(grid, args, kernel_name, kernel_source, key=key, extra_params=params)

    return tune


def tune_calc_cfl(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w", "s"])
    args = [
        fields.s.tend,
        fields.u.fld, fields.v.fld, fields.w.fld,
        grid.dzi, grid.dxi, grid.dyi,
        grid.icells, grid.ijcells,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend
    ]

    kernel_name = "calc_cfl_wrapper"
    kernel_source = "advec_2i5.cu"

    return tune_and_store(grid, args, kernel_name, kernel_source)


def tune_diff_uvw(grid):
    fields = Fields.from_grid(grid, ["u", "v", "w", "evisc"])
    visc = grid.TF(1.0)  # TODO: is this ok?

    args = [
        fields.u.tend, fields.v.tend, fields.w.tend,
        fields.evisc.fld,  # TODO: is this ok?
        fields.u.fld, fields.v.fld, fields.w.fld,
        fields.u.flux_bot, fields.u.flux_top,
        fields.v.flux_bot, fields.v.flux_top,
        grid.dzi, grid.dzhi, grid.dxi, grid.dyi,
        fields.rhoref, fields.rhorefh,
        visc,
        grid.istart, grid.jstart, grid.kstart,
        grid.iend, grid.jend, grid.kend,
        grid.icells, grid.ijcells,
    ]

    kernel_name = "diff_uvw_wrapper"
    kernel_source = "diff_smag2.cu"

    return tune_and_store(grid, args, kernel_name, kernel_source)

def main(args):
    tune_functions = dict(
        advec_u=tune_advec_u,
        advec_v=tune_advec_v,
        advec_w=tune_advec_w,
        advec_uvw=tune_advec_uvw,
        advec_uvw_int=tune_advec_uvw_interior,
        advec_s=tune_advec_s,
        advec_s1=tune_advec_sn(1),
        advec_s2=tune_advec_sn(2),
        advec_s4=tune_advec_sn(4),
        advec_s8=tune_advec_sn(8),
        advec_s_lim=tune_advec_s_lim,
        calc_cfl=tune_calc_cfl,
        diff_uvw=tune_diff_uvw,
    )

    if len(args) < 2:
        print(f'usage: python {args[0]} <kernel-name> <grid-size> <float-type>')
        return

    kernel_name = args[1]
    if kernel_name not in tune_functions:
        print(f'error: unknown kernel "{kernel_name}"')
        return

    gridsize = 512
    if len(args) >= 3:
        try:
            gridsize = int(args[2])
        except:
            print(f'error: invalid gridsize: {args[2]}')
            return

    TF = np.float64
    if len(args) >= 4:
        float_name = args[3]

        if float_name == 'float':
            TF = np.float32
        elif float_name == 'double':
            TF = np.float64
        else:
            print(f"invalid float type: \"{float_name}\"")

    grid = Grid(3200, 3200, 3200, gridsize, gridsize, gridsize, 2, 1, TF)
    pprint(tune_functions[kernel_name](grid))


if __name__ == "__main__":
    main(sys.argv)
