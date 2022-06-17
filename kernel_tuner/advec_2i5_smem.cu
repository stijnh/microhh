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

template <typename TF>
struct advec_u_i_interp {
    __forceinline__ __device__
    TF init(
            const int ijk,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk)
    {
        const int ii1 = 1;
        const int ii2 = 2;
        const int ii3 = 3;

        return fabs(interp2(u[ijk-ii1], u[ijk])) * interp5_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk], u[ijk+ii1], u[ijk+ii2])
               - interp2(u[ijk-ii1], u[ijk]) * interp6_ws(u[ijk-ii3], u[ijk-ii2], u[ijk-ii1], u[ijk], u[ijk+ii1], u[ijk+ii2]);
    }

    __forceinline__ __device__
    TF execute(
            TF left, TF right,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk)
    {
        return (right - left) * dxi;
    }
};

template <typename TF>
struct advec_u_j_interp {
    __forceinline__ __device__
    TF init(
            const int ijk_,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int ijknjj3 = i * ii + k * kk + (j - 3) * jj;
        const int ijknjj2 = i * ii + k * kk + (j - 2) * jj;
        const int ijknjj1 = i * ii + k * kk + (j - 1) * jj;
        const int ijk     = i * ii + k * kk + (j    ) * jj;
        const int ijkpjj1 = i * ii + k * kk + (j + 1) * jj;
        const int ijkpjj2 = i * ii + k * kk + (j + 2) * jj;

        return fabs(interp2(v[ijk-ii], v[ijk])) * interp5_ws(u[ijknjj3], u[ijknjj2], u[ijknjj1], u[ijk], u[ijkpjj1], u[ijkpjj2])
               - interp2(v[ijk-ii], v[ijk]) * interp6_ws(u[ijknjj3], u[ijknjj2], u[ijknjj1], u[ijk], u[ijkpjj1], u[ijkpjj2]);
    }

    __forceinline__ __device__
    TF execute(
            TF left, TF right,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk)
    {
        return (right - left) * dyi;
    }
};

template <typename TF>
struct advec_u_k_interp {
    __forceinline__ __device__
    TF init(
            const int ijk_,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk)
    {
        const int ii = 1;
        const int ijknkk3 = i * ii + j * jj + (k - 3) * kk;
        const int ijknkk2 = i * ii + j * jj + (k - 2) * kk;
        const int ijknkk1 = i * ii + j * jj + (k - 1) * kk;
        const int ijk     = i * ii + j * jj + (k    ) * kk;
        const int ijkpkk1 = i * ii + j * jj + (k + 1) * kk;
        const int ijkpkk2 = i * ii + j * jj + (k + 2) * kk;

        return -rhorefh[k] * interp2(w[ijk-ii], w[ijk]) * interp6_ws(u[ijknkk3], u[ijknkk2], u[ijknkk1], u[ijk], u[ijkpkk1], u[ijkpkk2])
               + rhorefh[k] * fabs(interp2(w[ijk-ii], w[ijk])) * interp5_ws(u[ijknkk3], u[ijknkk2], u[ijknkk1], u[ijk], u[ijkpkk1], u[ijkpkk2]);
    }

    __forceinline__ __device__
    TF execute(
            TF left, TF right,
            const int i, const int j, const int k,
            const TF* __restrict__ u, const TF* __restrict__ v,  const TF* __restrict__ w,
            const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
            const TF* __restrict__ dzi, const TF dxi, const TF dyi,
            const int jj, const int kk
    )
    {
        return (right - left) * dzi[k] / rhoref[k];
    }
};


template <typename Strategy>
__forceinline__ __device__
int calculate_i(int istart, int dx) {
    int thread_idx_x = threadIdx.x;

    if (Strategy::block_size_x == warpSize) {
        return istart + blockIdx.x * Strategy::tile_size_x + thread_idx_x + dx * warpSize;
    } else if (Strategy::block_size_x % warpSize == 0) {
        int laneid = thread_idx_x % warpSize;
        int warpid = thread_idx_x / warpSize;
        return istart + blockIdx.x * Strategy::tile_size_x + warpid * Strategy::tile_factor_x * warpSize + laneid +
               dx * warpSize;
    } else {
        int thread_idx_x = Strategy::block_size_x == 1 ? 0 : threadIdx.x;
        return istart + blockIdx.x * Strategy::tile_size_x + thread_idx_x * Strategy::tile_factor_x + dx;
    }

    // ((blockIdx.x * Strategy::block_size_x / warpSize + warpid) * Strategy::tile_factor_x + dx) * warpSize  + laneid
}

template <typename Strategy>
__forceinline__ __device__
int calculate_j(int jstart, int dy) {
    int thread_idx_y = Strategy::block_size_y == 1 ? 0 : threadIdx.y;
    return jstart + blockIdx.y * Strategy::tile_size_y + thread_idx_y * Strategy::tile_factor_y + dy;
}

template <typename Strategy>
__forceinline__ __device__
int calculate_k(int kstart, int dz) {
    int thread_idx_z = Strategy::block_size_z == 1 ? 0 : threadIdx.z;
    return kstart + blockIdx.z * Strategy::tile_size_z + thread_idx_z * Strategy::tile_factor_z + dz;
}

template <typename Strategy, typename FI, typename FJ, typename FK, typename ...Args>
__forceinline__ __device__
void diff_tiling(
        FI fi, FJ fj, FK fk,
        const int istart, const int jstart, const int kstart,
        const int iend,   const int jend,   const int kend,
        TF *output, const int ii, const int jj, const int kk,
        Args... args
) {
    double tile_i;
    double tile_j[Strategy::tile_factor_x];
    double tile_k[Strategy::tile_factor_y][Strategy::tile_factor_x];

    const int axis_start[3] = {istart, jstart, kstart};
    const int axis_end[3] = {iend, jend, kend};
    int indices[3];

#pragma unroll
    for (int dz = 0; dz < Strategy::tile_factor_z; dz++) {
        indices[2] = calculate_k<Strategy>(axis_start[2], dz);
        if (indices[2] >= axis_end[2] && Strategy::tile_size_xyz[2] > 1) break;

#pragma unroll
        for (int dy = 0; dy < Strategy::tile_factor_y; dy++) {
            indices[1] = calculate_j<Strategy>(axis_start[1], dy);
            if (indices[1] >= axis_end[1] && Strategy::tile_size_xyz[1] > 1) break;

#pragma unroll
            for (int dx = 0; dx < Strategy::tile_factor_x; dx++) {
                indices[0] = calculate_i<Strategy>(axis_start[0], dx);

                int i = indices[0];
                int j = indices[1];
                int k = indices[2];
                int ijk = i * ii + j * jj + k * kk;

                TF result = 0;
                TF left_i, right_i;
                right_i = i <= iend ? fi.init(ijk + ii, i + 1, j, k, args...) : 0.0;

                if (dx == 0 || Strategy::block_size_x % warpSize != 0) {
                    left_i = i < iend ? fi.init(ijk, i, j, k, args...) : 0.0;
                } else {
                    int laneid = threadIdx.x % warpSize;
                    left_i = __shfl_sync(
                        0xffffffff,
                        laneid == warpSize - 1 ? tile_i: right_i,
                        (laneid + warpSize - 1) % warpSize
                    );
                }

                if (i >= iend) break;
                tile_i = right_i;
                result += fi.execute(left_i, right_i, i, j, k, args...);

                TF left_j = dy == 0 ? fj.init(ijk, i, j, k, args...) : tile_j[dx];
                TF right_j = fj.init(ijk + jj, i, j + 1, k, args...);
                tile_j[dx] = right_j;
                result += fj.execute(left_j, right_j, i, j, k, args...);

                TF left_k = dz == 0 ? fk.init(ijk, i, j, k, args...) : tile_k[dy][dx];
                TF right_k = fk.init(ijk + kk, i, j, k + 1, args...);
                tile_k[dy][dx] = right_k;
                result += fk.execute(left_k, right_k, i, j, k, args...);

                output[ijk] += result;
            }
        }
    }
}

extern "C"
ELEMENTWISE_KERNEL(TuneTiling)
void advec_u_shared(TF* __restrict__ ut, const TF* __restrict__ u,
               const TF* __restrict__ v,  const TF* __restrict__ w,
               const TF* __restrict__ rhoref, const TF* __restrict__ rhorefh,
               const TF* __restrict__ dzi, const TF dxi, const TF dyi,
               const int jj, const int kk,
               const int istart, const int jstart, const int kstart,
               const int iend,   const int jend,   const int kend)
{
    diff_tiling<TuneTiling>(
            advec_u_i_interp<TF> {}, advec_u_j_interp<TF> {}, advec_u_k_interp<TF> {},
            istart, jstart, kstart,
            iend, jend, kend,
            ut, 1, jj, kk,
            u, v, w, rhoref, rhorefh, dzi, dxi, dyi, jj, kk);
}