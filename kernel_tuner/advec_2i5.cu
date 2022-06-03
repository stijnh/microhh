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
ELEMENTWISE_KERNEL(TuneTiling)
void advec_u_wrapper(TF* __restrict__ ut, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_u_body<TF>(),
            ut, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_v_wrapper(TF* __restrict__ vt, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_v_body<TF>(),
            vt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_w_wrapper(TF* __restrict__ wt, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_w_body<TF>(),
            wt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
}

template<typename TF>
struct advec_uvw_body
{
    template <typename Level>
    __forceinline__ __device__ void operator()(
            const int i, const int j, const int k, const Level level,
            TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
        advec_u_body<TF>()(i, j, k, level, ut, u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
        advec_v_body<TF>()(i, j, k, level, vt, u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
        advec_w_body<TF>()(i, j, k, level, wt, u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
    }
};

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_uvw_wrapper(
               TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
               const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
#if defined(FUSE_BLOCKS) and FUSE_BLOCKS
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_uvw_body<TF>(),
            ut, vt, wt,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
#else
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_u_body<TF>(),
            ut, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);

    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_v_body<TF>(),
            vt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);

    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_w_body<TF>(),
            wt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
#endif
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_uvw_interior_wrapper(
               TF* __restrict__ ut, TF* __restrict__ vt, TF* __restrict__ wt,
               const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
#if defined(FUSE_BLOCKS) and FUSE_BLOCKS
    TuneTiling::process_cta_middle(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_uvw_body<TF>(),
            ut, vt, wt,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
#else
    TuneTiling::process_cta_middle(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_u_body<TF>(),
            ut, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);

    TuneTiling::process_cta_middle(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_v_body<TF>(),
            vt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);

    TuneTiling::process_cta_middle(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_w_body<TF>(),
            wt, u,
            v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
#endif
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_s_wrapper(TF* __restrict__ st, const TF* __restrict__ s,
                     const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
                     const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
                     const TF* __restrict__ dzi, const TF dxi, const TF dyi,
                     const int jj, int kk,
                     const int istart, const int jstart, const int kstart,
                     const int iend,   const int jend,   const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_s_body<TF>(),
            st, s,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
}


template<typename TF, size_t N>
struct advec_s_multi_body
{
    template <typename Level>
    __forceinline__ __device__ void operator()(
            const int i, const int j, const int k, const Level level,
            TF* __restrict__ * __restrict__ st, const TF* __restrict__ * __restrict__ s,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
#pragma unroll N
        for (size_t index = 0; index < N; ++index)
        {
            advec_s_body<TF>()(
                    i, j, k, level,
                    st[index], s[index],
                    u, v, w,
                    rhoref, rhorefh,
                    dzi, dxi, dyi,
                    jj, kk
            );
        }
    }
};

#ifndef ADVEC_S_MULTI_SIZE
#define ADVEC_S_MULTI_SIZE 1
#endif
template <typename T, size_t N>
struct array { T elements[N]; };

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_s_multi_wrapper(
                     array<TF* __restrict__, ADVEC_S_MULTI_SIZE> st,
                     array<const TF* __restrict__, ADVEC_S_MULTI_SIZE> s,
                     const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
                     const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
                     const TF* __restrict__ dzi, const TF dxi, const TF dyi,
                     const int jj, int kk,
                     const int istart, const int jstart, const int kstart,
                     const int iend,   const int jend,   const int kend)
{
#if defined(FUSE_BLOCKS) and FUSE_BLOCKS
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_s_multi_body<TF, ADVEC_S_MULTI_SIZE>(),
            st.elements, s.elements,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
#else
    static constexpr size_t N = ADVEC_S_MULTI_SIZE;
    #pragma unroll N
    for (size_t index = 0; index < N; ++index)
    {
        TuneTiling::process_cta(
                istart, jstart, kstart,
                iend, jend, kend,
                advec_s_body<TF>(),
                st.elements[index], s.elements[index],
                u, v, w,
                rhoref, rhorefh,
                dzi, dxi, dyi,
                jj, kk);
    }
#endif
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_s_lim_wrapper(
        TF* __restrict__ st, const TF* __restrict__ s,
        const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
        const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
        const TF* __restrict__ dzi, const TF dxi, const TF dyi,
        const int jj, int kk,
        const int istart, const int jstart, const int kstart,
        const int iend, const int jend, const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            advec_s_lim_body<TF>(),
            st, s,
            u, v, w,
            rhoref, rhorefh,
            dzi, dxi, dyi,
            jj, kk);
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void calc_cfl_wrapper(
                TF* const __restrict__ tmp1,
                const TF* __restrict__ u, const TF* __restrict__ v, const TF* __restrict__ w,
                const TF* __restrict__ dzi, const TF dxi, const TF dyi,
                const int jj, const int kk,
                const int istart, const int jstart, const int kstart,
                const int iend, const int jend, const int kend)
{
    TuneTiling::process_cta(
            istart, jstart, kstart,
            iend, jend, kend,
            calc_cfl_body<TF>(),
            tmp1,
            u, v, w,
            dzi, dxi, dyi,
            jj, kk);
}