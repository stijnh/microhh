#ifndef TILING
#define TILING

#include "grid.h"
#include "tools.h"

#if defined(__CUDACC__) && __CUDACC_VER_MAJOR__ >= 11 && __CUDACC_VER_MINOR__ >= 2 // since 11.2
#define CUDA_ASSUME(expr) __builtin_assume(expr)
#else
#define CUDA_ASSUME(expr) do{} while(0)
#endif

#define CUDA_DEVICE __device__ __forceinline__
#define CUDA_HOST_DEVICE __host__ __device__ __forceinline__
#define ELEMENTWISE_KERNEL(Strategy)  __global__ __launch_bounds__(Strategy::block_size_xyz, Strategy::blocks_per_sm)


template <
        unsigned int block_size_x_,
        unsigned int block_size_y_,
        unsigned int block_size_z_,
        unsigned int tile_factor_x_,
        unsigned int tile_factor_y_,
        unsigned int tile_factor_z_,
        unsigned int unroll_factor_x_,
        unsigned int unroll_factor_y_,
        unsigned int unroll_factor_z_,
        unsigned int blocks_per_sm_=1
>
struct TilingStrategy
{
    static constexpr unsigned int block_size_x = block_size_x_;
    static constexpr unsigned int block_size_y = block_size_y_;
    static constexpr unsigned int block_size_z = block_size_z_;
    static constexpr unsigned int tile_factor_x = tile_factor_x_;
    static constexpr unsigned int tile_factor_y = tile_factor_y_;
    static constexpr unsigned int tile_factor_z = tile_factor_z_;
    static constexpr unsigned int unroll_factor_x = unroll_factor_x_;
    static constexpr unsigned int unroll_factor_y = unroll_factor_y_;
    static constexpr unsigned int unroll_factor_z = unroll_factor_z_;
    static constexpr unsigned int blocks_per_sm = blocks_per_sm_;

    static constexpr unsigned int tile_size_x = tile_factor_x * block_size_x;
    static constexpr unsigned int tile_size_y = tile_factor_y * block_size_y;
    static constexpr unsigned int tile_size_z = tile_factor_z * block_size_z;

    static constexpr unsigned int block_size_xyz = block_size_x * block_size_y * block_size_z;
    static constexpr unsigned int tile_size_xyz = tile_size_x * tile_size_y * tile_size_z;

    using single_layer_strategy = TilingStrategy<
            block_size_x, block_size_y, 1,
            tile_factor_x, tile_factor_y, 1,
            unroll_factor_x, unroll_factor_y, 1,
            blocks_per_sm>;

    static_assert(block_size_xyz > 0, "invalid block size");
    static_assert(tile_size_xyz > 0, "invalid tile size");

    template <typename F, typename... Args>
    CUDA_DEVICE
    static void process_cta_layer(
            const int istart, const int jstart,
            const int iend, const int jend,
            F fun, Args... args)
    {
#pragma unroll(unroll_factor_y)
        for (int dj = 0; dj < tile_factor_y; dj++)
        {
            const int thread_idx_y = block_size_y > 1 ? threadIdx.y : 0;
            const int j = jstart + blockIdx.y * tile_size_y + dj * block_size_y + thread_idx_y;
            if (block_size_y > 1 && j >= jend) break;

#pragma unroll(unroll_factor_x)
            for (int di = 0; di < tile_factor_x; di++)
            {
                const int thread_idx_x = block_size_x > 1 ? threadIdx.x : 0;
                const int i = istart + blockIdx.x * tile_size_x + di * block_size_x + thread_idx_x;
                if (block_size_x > 1 && i >= iend) break;

                fun(i, j, args...);
            }
        }
    }

    template <typename F, typename... Args>
    CUDA_DEVICE
    static void process_cta(
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            F fun, Args... args)
    {
        struct Level {
            CUDA_HOST_DEVICE
            Level(int dist_start, int dist_end): dist_start_(dist_start), dist_end_(dist_end) {}

            CUDA_HOST_DEVICE
            int distance_to_start() const {
                return dist_start_;
            }

            CUDA_HOST_DEVICE
            int distance_to_end() const {
                return dist_end_;
            }

          private:
            int dist_start_;
            int dist_end_;
        };

#pragma unroll(unroll_factor_z)
        for (int dk = 0; dk < tile_factor_z; dk++)
        {
            const int thread_idx_z = block_size_z > 1 ? threadIdx.z : 0;
            const int k = kstart + blockIdx.z * tile_size_z + dk * block_size_z + thread_idx_z;
            if (block_size_z > 1 && k >= kend) break;

            Level level(k - kstart, kend - k - 1);
            process_cta_layer(istart, jstart, iend, jend, fun, k, level, args...);
        }
    }

    static dim3 grid_size(const int itot, const int jtot, const int ktot)
    {
        return dim3(
                (itot / tile_size_x) + (itot % tile_size_x > 0),
                (jtot / tile_size_y) + (jtot % tile_size_y > 0),
                (ktot / tile_size_z) + (ktot % tile_size_z > 0)
        );
    }

    static dim3 grid_size(
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend)
    {
        return grid_size(iend - istart, jend - jstart, kend - kstart);
    }

    template <typename TF>
    static dim3 grid_size(const Grid_data<TF>& gd)
    {
        return grid_size(gd.istart, gd.jstart, gd.kstart, gd.iend, gd.jend, gd.kend);
    }

    static dim3 block_size()
    {
        return dim3(block_size_x, block_size_y, block_size_z);
    }
};

using DefaultTiling = TilingStrategy<256, 1, 1, 1, 1, 1, 1, 1, 1>;


template <typename Strategy, typename F, typename... Args>
ELEMENTWISE_KERNEL(Strategy)
static void elementwise_kernel(
            const int istart, const int jstart, const int kstart,
            const int iend, const int jend, const int kend,
            F fun, Args... args)
{
    Strategy::process_cta(istart, jstart, kstart, iend, jend, kend, fun, args...);
}

template <typename Strategy = DefaultTiling, typename F, typename ...Args>
static void launch_elementwise_kernel(
        const int istart, const int jstart, const int kstart,
        const int iend, const int jend, const int kend,
        F fun, Args... args)
{
    dim3 block_size = Strategy::block_size();
    dim3 grid_size = Strategy::grid_size(istart, jstart, kstart, iend, jend, kend);

    elementwise_kernel<Strategy><<<grid_size, block_size>>>(
        istart, jstart, kstart,
        iend, jend, kend,
        fun, args...);
    cuda_check_error();
}

template <typename Strategy = DefaultTiling, typename TF, typename F, typename ...Args>
static void launch_elementwise_kernel(const Grid_data<TF>& gd, F fun, Args... args)
{
    launch_elementwise_kernel<Strategy, F, Args...>(
            gd.istart, gd.jstart, gd.kstart,
            gd.iend, gd.jend, gd.kend,
            fun, args...);
}
#endif
