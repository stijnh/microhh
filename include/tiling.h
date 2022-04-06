#ifndef TILING
#define TILING

#include "grid.h"

template <
        unsigned int block_size_x_,
        unsigned int block_size_y_,
        unsigned int block_size_z_,
        unsigned int tile_factor_x_,
        unsigned int tile_factor_y_,
        unsigned int tile_factor_z_,
        unsigned int unroll_factor_x_,
        unsigned int unroll_factor_y_,
        unsigned int unroll_factor_z_
>
struct TilingStrategy {
    static constexpr unsigned int block_size_x = block_size_x_;
    static constexpr unsigned int block_size_y = block_size_y_;
    static constexpr unsigned int block_size_z = block_size_z_;
    static constexpr unsigned int tile_factor_x = tile_factor_x_;
    static constexpr unsigned int tile_factor_y = tile_factor_y_;
    static constexpr unsigned int tile_factor_z = tile_factor_z_;
    static constexpr unsigned int unroll_factor_x = unroll_factor_x_;
    static constexpr unsigned int unroll_factor_y = unroll_factor_y_;
    static constexpr unsigned int unroll_factor_z = unroll_factor_z_;

    static constexpr unsigned int tile_size_x = tile_factor_x * block_size_x;
    static constexpr unsigned int tile_size_y = tile_factor_y * block_size_y;
    static constexpr unsigned int tile_size_z = tile_factor_z * block_size_x;

    static constexpr unsigned int block_size_xyz = block_size_z * block_size_y * block_size_z;
    static constexpr unsigned int tile_size_xyz = tile_size_x * tile_size_y * tile_size_z;

    template <typename F, typename... Args>
    __device__ __forceinline__
    static void execute(
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            F fun, Args... args
    ) {
        static_assert(block_size_xyz > 0, "invalid block size");
        static_assert(tile_size_xyz > 0, "invalid tile size");

#pragma unroll(unroll_factor_z)
        for (int dk = 0; dk < tile_size_z; dk += block_size_z) {
            const int thread_idx_z = block_size_z > 1 ? threadIdx.z : 0;
            const int k = kstart + (blockIdx.z * tile_factor_z) * block_size_z + thread_idx_z + dk;
            if (k >= kend) break;

#pragma unroll(unroll_factor_y)
            for (int dj = 0; dj < tile_size_y; dj += block_size_y) {
                const int thread_idx_y = block_size_y > 1 ? threadIdx.y : 0;
                const int j = jstart + (blockIdx.y * tile_factor_y) * block_size_y + thread_idx_y + dj;
                if (j >= jend) break;

#pragma unroll(unroll_factor_x)
                for (int di = 0; di < tile_size_x; di += block_size_x) {
                    const int thread_idx_x = block_size_x > 1 ? threadIdx.x : 0;
                    const int i = istart + (blockIdx.x * tile_factor_x) * block_size_x + thread_idx_x + di;
                    if (i >= iend) break;

                    fun(i, j, k, args...);
                }
            }
        }
    }

    static dim3 grid_size(const int itot, const int jtot, const int ktot) {
        return dim3(
                (itot / tile_size_x) + (itot % tile_size_x > 0),
                (jtot / tile_size_y) + (jtot % tile_size_y > 0),
                (ktot / tile_size_z) + (ktot % tile_size_z > 0)
        );
    }

    static dim3 grid_size(
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend
    ) {
        return grid_size(iend - istart, jend - jstart, kend - kstart);
    }

    template <typename TF>
    static dim3 grid_size(const Grid_data<TF>& gd) {
        return grid_size(gd.imax, gd.jmax, gd.kmax);
    }

    static dim3 block_size() {
        return dim3(block_size_x, block_size_y, block_size_z);
    }
};

using DefaultTiling = TilingStrategy<256, 1, 1, 1, 1, 1, 1, 1, 1>;

#define TILING_KERNEL(Strategy)  __global__ __launch_bounds__(Strategy::block_size_xyz)

#endif
