#include "../src/advec_2i5.cu"

#ifndef USE_RECIPROCAL
#define USE_RECIPROCAL false
#endif

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

template <int axis, typename F, typename Strategy>
struct SmemTilingStrategy {
    static_assert(axis >= 0 && axis <= 2, "invalid axis, must be 0, 1, 2");
    static_assert(Strategy::tile_size_x > 1, "tile_size_x must be greater than one");
    static_assert(Strategy::tile_size_y > 1, "tile_size_y must be greater than one");
    static_assert(Strategy::tile_size_z > 1, "tile_size_z must be greater than one");

    static constexpr int smem_ii = 1;
    static constexpr int smem_jj = Strategy::tile_size_x;// - int(axis != 0);
    static constexpr int smem_kk = smem_jj * (Strategy::tile_size_y) - int(axis != 1);
    static constexpr int smem_size = smem_kk * (Strategy::tile_size_z) - int(axis != 2);
    static constexpr int smem_strides[3] = {smem_ii, smem_jj, smem_kk};

    CUDA_HOST_DEVICE
    SmemTilingStrategy(F functor = {}): functor_(functor) {}

    template <typename TF, typename ...Args>
    CUDA_DEVICE void initialize_smem(
            TF* smem,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            Args... args
    ) {
        const int thread_idx_x = Strategy::block_size_x == 1 ? 0 : threadIdx.x;
        const int thread_idx_y = Strategy::block_size_y == 1 ? 0 : threadIdx.y;
        const int thread_idx_z = Strategy::block_size_z == 1 ? 0 : threadIdx.z;

#pragma unroll(Strategy::unroll_factor_z)
        for (int dk = 0; dk < Strategy::tile_factor_z; dk++) {
            const int koffset = dk * Strategy::block_size_z + thread_idx_z;
            const int k = kstart + blockIdx.z * (Strategy::tile_size_z - 1) + koffset;


#pragma unroll(Strategy::unroll_factor_y)
            for (int dj = 0; dj < Strategy::tile_factor_y; dj++) {
                const int joffset = dj * Strategy::block_size_y + thread_idx_y;
                const int j = jstart + blockIdx.y * (Strategy::tile_size_y - 1) + joffset;


#pragma unroll(Strategy::unroll_factor_x)
                for (int di = 0; di < Strategy::tile_factor_x; di++) {
                    const int ioffset = di * Strategy::block_size_x + thread_idx_x;
                    const int i = istart + blockIdx.x * (Strategy::tile_size_x - 1) + ioffset;

                    bool valid = true;
                    if (axis == 2) {
                        if (k > kend) valid = false;
                    } else {
                        if (k >= kend || koffset >= Strategy::tile_size_z - 1) valid = false;
                    }

                    if (axis == 1) {
                        if (j > jend) valid = false;
                    } else {
                        if (j >= jend || joffset >= Strategy::tile_size_y - 1) valid = false;
                    }

                    if (axis == 0) {
                        if (i > iend) valid = false;
                    } else {
                        if (i >= iend || ioffset >= Strategy::tile_size_x - 1) valid = false;
                    }

                    const int index = ioffset * smem_ii + joffset * smem_jj + koffset * smem_kk;
                    if (valid) smem[index] = functor_.init(i, j, k, args...);
                }
            }
        }
    }

    template <bool UseSmem, typename TF, typename ...Args>
    CUDA_DEVICE void execute_internal(
            const TF* smem,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            Args... args
    ) {
        const int thread_idx_x = Strategy::block_size_x == 1 ? 0 : threadIdx.x;
        const int thread_idx_y = Strategy::block_size_y == 1 ? 0 : threadIdx.y;
        const int thread_idx_z = Strategy::block_size_z == 1 ? 0 : threadIdx.z;

#pragma unroll(Strategy::unroll_factor_z)
        for (int dk = 0; dk < Strategy::tile_factor_z; dk++)
        {
            const int koffset = dk * Strategy::block_size_z + thread_idx_z;
            const int k = kstart + blockIdx.z * (Strategy::tile_size_z - 1)  + koffset;

#pragma unroll(Strategy::unroll_factor_y)
            for (int dj = 0; dj < Strategy::tile_factor_y; dj++)
            {
                const int joffset = dj * Strategy::block_size_y + thread_idx_y;
                const int j = jstart + blockIdx.y * (Strategy::tile_size_y - 1) + joffset;

#pragma unroll(Strategy::unroll_factor_x)
                for (int di = 0; di < Strategy::tile_factor_x; di++)
                {
                    const int ioffset = di * Strategy::block_size_x + thread_idx_x;
                    const int i = istart + blockIdx.x * (Strategy::tile_size_x - 1) + ioffset;

                    if (i >= iend || ioffset >= Strategy::tile_size_x - 1) continue;
                    if (j >= jend || joffset >= Strategy::tile_size_y - 1) continue;
                    if (k >= kend || koffset >= Strategy::tile_size_z - 1) continue;

                    TF left, right;

                    if (UseSmem) {
                        const int lindex = ioffset * smem_ii + joffset * smem_jj + koffset * smem_kk;
                        const int rindex = lindex + smem_strides[axis];

                        left = smem[lindex];
                        right = smem[rindex];
                    } else {
                        left = functor_.init(i, j, k, args...);
                        right = functor_.init(i +  (int)(axis == 0), j + (int)(axis == 1), k + (int)(axis == 2), args...);
                    }

                    functor_.execute(
                            left, right,
                            i, j, k,
                            args...
                    );
                }
            }
        }
    }

    template <typename ...Args>
    CUDA_DEVICE void execute_smem(Args... args) {
        execute_internal<true>(args...);
    }

    template <typename ...Args>
    CUDA_DEVICE void execute_nosmem(Args... args) {
        execute_internal<false>((TF*) nullptr, args...);
    }

    template <typename TF, typename ...Args>
    CUDA_DEVICE void execute(
            TF* smem,
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            Args... args
    ) {
        initialize_smem(smem, istart, jstart, kstart, iend, jend, kend, args...);
        __syncthreads();
        execute_smem(smem, istart, jstart, kstart, iend, jend, kend, args...);
    }

private:
    F functor_;
};


template <typename TF>
struct advec_u_i_interp {
    __forceinline__ __device__
    TF init(
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
        const int ii1 = 1;
        const int ii2 = 2;
        const int ii3 = 3;
        const int ijk = i + j*jj + k*kk;

        return fabs(interp2(u[ijk-ii1], u[ijk])) * interp5_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk], u[ijk+ii1], u[ijk+ii2])
               - interp2(u[ijk-ii1], u[ijk]) * interp6_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk], u[ijk+ii1], u[ijk+ii2]);
    }

    __forceinline__ __device__
    void execute(
            TF left, TF right,
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
        const int ijk = i + j*jj + k*kk;
        ut[ijk] += (right - left) * dxi;
    }
};

template <typename TF>
struct advec_u_j_interp {
    __forceinline__ __device__
    TF init(
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
        const int ii1 = 1;
        const int jj1 = 1*jj;
        const int jj2 = 2*jj;
        const int jj3 = 3*jj;
        const int ijk = i + j*jj + k*kk;

        return fabs(interp2(v[ijk-ii1], v[ijk])) * interp5_ws(u[ijk-jj3], u[ijk-jj2], u[ijk-jj1], u[ijk], u[ijk+jj1], u[ijk+jj2])
               - interp2(v[ijk-ii1], v[ijk]) * interp6_ws(u[ijk-jj3], u[ijk-jj2], u[ijk-jj1], u[ijk], u[ijk+jj1], u[ijk+jj2]);
    }

    __forceinline__ __device__
    void execute(
            TF left, TF right,
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk
    )
    {
        const int ijk = i + j*jj + k*kk;
        ut[ijk] += (right - left) * dyi;
    }
};

template <typename TF>
struct advec_u_k_interp {
    __forceinline__ __device__
    TF init(
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk)
    {
        const int ii1 = 1;
        const int kk1 = 1*kk;
        const int kk2 = 2*kk;
        const int kk3 = 3*kk;
        const int ijk = i + j*jj + k*kk;

        return -rhorefh[k] * interp2(w[ijk-ii1], w[ijk]) * interp6_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk], u[ijk+kk1], u[ijk+kk2])
            + rhorefh[k] * fabs(interp2(w[ijk-ii1], w[ijk])) * interp5_ws(u[ijk-kk3], u[ijk-kk2], u[ijk-kk1], u[ijk], u[ijk+kk1], u[ijk+kk2]);
    }

    __forceinline__ __device__
    void execute(
            TF left, TF right,
            const int i, const int j, const int k,
            TF* __restrict__ ut, const TF* __restrict__ u,
            const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, int kk
    )
    {
        const int ijk = i + j*jj + k*kk;
        ut[ijk] += (right - left) * dzi[k] / rhoref[k];
    }
};

template <typename T>
__host__ __device__ constexpr T max3(T a, T b, T c) {
    if (a > b && a > c) {
        return a;
    } else if (b > c) {
        return b;
    } else {
        return c;
    }
}


extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_u_g(TF* __restrict__ ut, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    using Tiling = TuneTiling;
    using AdvecUI = SmemTilingStrategy<0, advec_u_i_interp<TF>, Tiling>;
    using AdvecUJ = SmemTilingStrategy<1, advec_u_j_interp<TF>, Tiling>;
    using AdvecUK = SmemTilingStrategy<2, advec_u_k_interp<TF>, Tiling>;

#if USE_SMEM_X || USE_SMEM_Y || USE_SMEM_Z
    static constexpr size_t smem_size = max3(
            AdvecUI::smem_size * int(USE_SMEM_X != 0),
            AdvecUJ::smem_size * int(USE_SMEM_Y != 0),
            AdvecUK::smem_size * int(USE_SMEM_Z != 0)
    );
    __shared__ TF smem[smem_size];
#endif

#if USE_SMEM_X
    AdvecUI{}.execute(smem,
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, dzi, dxi, dyi, jj, kk);
    __syncthreads();
#else
    AdvecUI{}.execute_nosmem(
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, dzi, dxi, dyi, jj, kk);
#endif

#if USE_SMEM_Y
    AdvecUJ{}.execute(smem,
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, v, w, dzi, dxi, dyi, jj, kk);
    __syncthreads();
#else
    AdvecUJ{}.execute_nosmem(
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, v, w, dzi, dxi, dyi, jj, kk);
#endif

#if USE_SMEM_Z
    AdvecUK{}.execute(smem,
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
#else
    AdvecUK{}.execute_nosmem(
            istart, jstart, kstart,
            iend, jend, kend,
            ut, u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
#endif
}