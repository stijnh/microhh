#include "../src/advec_2i5.cu"

using TF = float_type;
using TuneTiling = TilingStrategy<
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_Z,
        TILING_FACTOR_X,
        TILING_FACTOR_Y,
        TILING_FACTOR_Z,
        LOOP_UNROLL_FACTOR_X,
        LOOP_UNROLL_FACTOR_Y,
        LOOP_UNROLL_FACTOR_Z,
        BLOCKS_PER_MP
>;

extern "C"
TILING_KERNEL(TuneTiling)
void advec_u_wrapper(TF* __restrict__ ut, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_u_body<TF>(),
            ut, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}

extern "C"
TILING_KERNEL(TuneTiling)
void advec_v_wrapper(TF* __restrict__ vt, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_v_body<TF>(),
            vt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}

extern "C"
TILING_KERNEL(TuneTiling)
void advec_w_wrapper(TF* __restrict__ wt, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_w_body<TF>(),
            wt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}

extern "C"
TILING_KERNEL(TuneTiling)
void advec_s_wrapper(TF* __restrict__ st, const TF* __restrict__ s,
                     const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
                     const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
                     const TF* __restrict__ dzi, const TF dxi, const TF dyi,
                     const int jj, int kk,
                     const int istart, const int jstart, const int kstart,
                     const int iend,   const int jend,   const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_s_body<TF>(),
            st, s,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}

extern "C"
TILING_KERNEL(TuneTiling)
void advec_s_lim_wrapper(
        TF* __restrict__ st, const TF* __restrict__ s,
        const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
        const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
        const TF* __restrict__ dzi, const TF dxi, const TF dyi,
        const int jj, int kk,
        const int istart, const int jstart, const int kstart,
        const int iend, const int jend, const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_s_lim_body<TF>(),
            st, s,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}

extern "C"
TILING_KERNEL(TuneTiling)
void calc_cfl_wrapper(
                TF* const __restrict__ tmp1,
                const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
                const TF* __restrict__ dzi, const TF dxi, const TF dyi,
                const int jj, const int kk,
                const int istart, const int jstart, const int kstart,
                const int iend, const int jend, const int kend)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            calc_cfl_body<TF>(),
            tmp1,
            u, v, w,
            dzi, dxi, dyi,
            jj, kk,
            kstart, kend);
}