#include "../src/diff_smag2.cu"

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

static constexpr Surface_model surface_model = Surface_model::Enabled;

extern "C"
TILING_KERNEL(TuneTiling)
void calc_strain2_wrapper(
        TF* __restrict__ strain2,
        TF* __restrict__ u,
        TF* __restrict__ v,
        TF* __restrict__ w,
        TF* __restrict__ dudz,
        TF* __restrict__ dvdz,
        TF* __restrict__ dzi,
        TF* __restrict__ dzhi,
        const TF dxi, const TF dyi,
        const int istart, const int iend,
        const int jstart, const int jend,
        const int kstart, const int kend,
        const int jj,     const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            calc_strain2_body<TF, surface_model>(),
            strain2,
            u, v, w,
            dudz, dvdz, dzi, dzhi,
            dxi, dyi,
            kstart,
            jj, kk);
}


extern "C"
TILING_KERNEL(TuneTiling)
void evisc_wrapper(
        TF* __restrict__ evisc,
        TF* __restrict__ N2,
        TF* __restrict__ bgradbot,
        TF* __restrict__ mlen0,
        TF* __restrict__ z0m,
        TF* __restrict__ z,
        const TF tPri,
        const int istart, const int iend,
        const int jstart, const int jend,
        const int kstart, const int kend,
        const int jj,     const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            evisc_body<TF, surface_model>(),
            evisc, N2, bgradbot, mlen0, z0m, z,
            tPri,
            kstart,
            jj, kk);
}


extern "C"
TILING_KERNEL(TuneTiling)
void evisc_neutral_wrapper(
        TF* __restrict__ evisc,
        TF* __restrict__ z0m,
        TF* __restrict__ z,
        TF* __restrict__ mlen0,
        const int istart, const int jstart, const int kstart,
        const int iend, const int jend, const int kend,
        const int jj, const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            evisc_neutral_body<TF>(),
            evisc, z0m, z, mlen0,
            jj, kk);
}


extern "C"
TILING_KERNEL(TuneTiling)
void diff_uvw_wrapper(TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
                TF* __restrict__ evisc,
                TF* __restrict__ u, TF* __restrict__ v, TF* __restrict__ w,
                TF* __restrict__ fluxbotu, TF* __restrict__ fluxtopu,
                TF* __restrict__ fluxbotv, TF* __restrict__ fluxtopv,
                TF* __restrict__ dzi, TF* __restrict__ dzhi, const TF dxi, const TF dyi,
                TF* __restrict__ rhoref, TF* __restrict__ rhorefh,
                const TF visc,
                const int istart, const int jstart, const int kstart,
                const int iend,   const int jend,   const int kend,
                const int jj,     const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            diff_uvw_body<TF, surface_model>(),
            ut, vt, wt,
            evisc,
            u, v, w,
            fluxbotu, fluxtopu,
            fluxbotv, fluxtopv,
            dzi, dzhi, dxi, dyi,
            rhoref, rhorefh,
            visc,
            kstart, kend,
            jj, kk);
}


extern "C"
TILING_KERNEL(TuneTiling)
void diff_c_wrapper(TF* __restrict__ at, TF* __restrict__ a, TF* __restrict__ evisc,
              TF* __restrict__ fluxbot, TF* __restrict__ fluxtop,
              TF* __restrict__ dzi, TF* __restrict__ dzhi, const TF dxidxi, const TF dyidyi,
              TF* __restrict__ rhoref, TF* __restrict__ rhorefh,
              const TF tPri, const TF visc,
              const int istart, const int jstart, const int kstart,
              const int iend,   const int jend,   const int kend,
              const int jj,     const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            diff_c_body<TF, surface_model>(),
            at, a, evisc,
            fluxbot, fluxtop,
            dzi, dzhi, dxidxi, dyidyi,
            rhoref, rhorefh,
            tPri, visc,
            kstart, kend,
            jj, kk);
}


extern "C"
TILING_KERNEL(TuneTiling)
void calc_dnmul_wrapper(TF* __restrict__ dnmul, TF* __restrict__ evisc,
                  TF* __restrict__ dzi, TF tPrfac, const TF dxidxi, const TF dyidyi,
                  const int istart, const int jstart, const int kstart,
                  const int iend,   const int jend,   const int kend,
                  const int jj,     const int kk)
{
    TuneTiling::execute_block(
            istart, jstart, kstart,
            iend, jend, kend,
            calc_dnmul_body<TF>(),
            dnmul, evisc,
            dzi, tPrfac, dxidxi, dyidyi,
            jj, kk);
}