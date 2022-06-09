import numpy as np
import pycuda.driver as drv
import re
import os
import math
import kernel_tuner
from collections import OrderedDict
import json
import platform
from datetime import datetime

class Grid:
    def __init__(self, xsize, ysize, zsize, itot, jtot, ktot, ijgc=1, kgc=1, TF=np.float64):
        """
        Simple (equidistant) grid
        """

        np.random.seed(666)
        self.TF = TF

        self.xsize = TF(xsize)
        self.ysize = TF(ysize)
        self.zsize = TF(zsize)

        self.itot = np.int32(itot)
        self.jtot = np.int32(jtot)
        self.ktot = np.int32(ktot)

        self.igc = np.int32(ijgc)
        self.jgc = np.int32(ijgc)
        self.kgc = np.int32(kgc)

        self.icells = np.int32(itot+2*ijgc)
        self.jcells = np.int32(jtot+2*ijgc)
        self.kcells = np.int32(ktot+2*kgc)

        self.ijcells = np.int32(self.icells*self.jcells)
        self.ncells  = np.int32(self.icells*self.jcells*self.kcells)

        self.istart = np.int32(self.igc)
        self.jstart = np.int32(self.jgc)
        self.kstart = np.int32(self.kgc)

        self.iend = np.int32(itot+self.igc)
        self.jend = np.int32(jtot+self.jgc)
        self.kend = np.int32(ktot+self.kgc)

        self.dx = TF(self.xsize / self.itot)
        self.dy = TF(self.ysize / self.jtot)
        self.dz = np.random.random(self.kcells).astype(TF)

        self.dxi = TF(1/self.dx)
        self.dyi = TF(1/self.dy)

        self.dzi   = np.random.random(self.kcells).astype(TF)
        self.dzhi  = np.random.random(self.kcells).astype(TF)
        self.dzi4  = np.random.random(self.kcells).astype(TF)
        self.dzhi4 = np.random.random(self.kcells).astype(TF)



class Field3d:
    def __init__(self, ncells, ijcells, TF=np.float64):
        """
        Simple 3D field incl. some surface fields
        """

        self.fld  = np.random.random(ncells).astype(TF)
        self.tend = np.zeros(ncells, dtype=TF)

        self.fld_bot = np.random.random(ijcells).astype(TF)
        self.fld_top = np.random.random(ijcells).astype(TF)

        self.flux_bot = np.random.random(ijcells).astype(TF)
        self.flux_top = np.random.random(ijcells).astype(TF)


class Fields:
    def __init__(self, fields, ncells, ijcells, kcells, TF=np.float64):

        np.random.seed(666)

        for field in fields:
            setattr(self, field, Field3d(ncells, ijcells, TF))

        self.rhoref  = np.random.random(kcells).astype(TF)
        self.rhorefh = np.random.random(kcells).astype(TF)

    @staticmethod
    def from_grid(grid, fields):
        return Fields(fields, grid.ncells, grid.ijcells, grid.kcells, TF=grid.TF)


def device_name(device=0):
    drv.init()
    name = drv.Device(device).name().lower()
    return re.sub('[^a-zA-Z0-9_-]', '-', name)


def device_attribute(name, device=0):
    drv.init()
    device = drv.Device(device)
    return device.get_attribute(drv.device_attribute.names[name])


def float_name(TF):
    if TF == np.float64:
        return "double"
    elif TF == np.float32:
        return "float"
    else:
        raise "unknown type: {}".format(TF)

def tune_kernel(grid, args, kernel_name, kernel_source, cache_file, extra_tuning, extra_params=dict(), nlayers=None, cta_padding=0):
    # Tune parameters
    tune_params = OrderedDict()
    tune_params["BLOCK_SIZE_X"] = [1, 2, 4, 8, 16, 32, 128, 256]
    tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16, 32]
    tune_params["BLOCK_SIZE_Z"] = [1, 2]
    tune_params["STATIC_STRIDES"] = [0]
    tune_params["TILING_FACTOR_X"] = [1]
    tune_params["TILING_FACTOR_Y"] = [1]
    tune_params["TILING_FACTOR_Z"] = [1]
    tune_params["TILING_STRATEGY"] = [0]
    tune_params["REWRITE_INTERP"] = [0]
    tune_params["BLOCKS_PER_MP"] = [0]
    tune_params["LOOP_UNROLL_FACTOR_X"] = [1]
    tune_params["LOOP_UNROLL_FACTOR_Y"] = [1]
    tune_params["LOOP_UNROLL_FACTOR_Z"] = [1]
    strategy = "brute_force"
    strategy_options = dict()

    if extra_tuning:
        tune_params["BLOCK_SIZE_X"] = [1, 2, 4, 8, 16, 32, 128, 256, 512, 1024]
        tune_params["BLOCK_SIZE_Y"] = [1, 2, 4, 8, 16, 32]
        tune_params["BLOCK_SIZE_Z"] = [1, 2, 4]
        tune_params["TILING_FACTOR_X"] = [1, 2, 4, 8]
        tune_params["TILING_FACTOR_Y"] = [1, 2, 4]
        tune_params["TILING_FACTOR_Z"] = [1, 2, 4]
        tune_params["LOOP_UNROLL_FACTOR_X"] = [0, 1] #tune_params["TILING_FACTOR_X"]
        tune_params["LOOP_UNROLL_FACTOR_Y"] = [0, 1] #tune_params["TILING_FACTOR_Y"]
        tune_params["LOOP_UNROLL_FACTOR_Z"] = [0, 1] #tune_params["TILING_FACTOR_Z"]
        tune_params["BLOCKS_PER_MP"] = [0, 1, 2, 3, 4]
        strategy = "bayes_opt"


    for key, values in extra_params.items():
        # Only add if extra_tuning is enabled, otherwise just add a dummy parameter with a single value.
        tune_params[key] = values if extra_tuning else [values[0]]

    max_threads_per_sm = device_attribute("MAX_THREADS_PER_MULTIPROCESSOR")
    max_threads_per_block = device_attribute("MAX_THREADS_PER_BLOCK")

    restrictions = [
        f"BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z * BLOCKS_PER_MP <= {max_threads_per_sm}",
        f"32 <= BLOCK_SIZE_X * BLOCK_SIZE_Y * BLOCK_SIZE_Z <= {max_threads_per_block}",
        "LOOP_UNROLL_FACTOR_X == 0 or TILING_FACTOR_X % LOOP_UNROLL_FACTOR_X == 0",
        "LOOP_UNROLL_FACTOR_Y == 0 or TILING_FACTOR_Y % LOOP_UNROLL_FACTOR_Y == 0",
        "LOOP_UNROLL_FACTOR_Z == 0 or TILING_FACTOR_Z % LOOP_UNROLL_FACTOR_Z == 0",
        f"BLOCK_SIZE_X * TILING_FACTOR_X > {cta_padding}",
        f"BLOCK_SIZE_Y * TILING_FACTOR_Y > {cta_padding}",
        f"BLOCK_SIZE_Z * TILING_FACTOR_Z > {cta_padding}",
    ]

    # general options
    lang = 'CUDA'
    block_size_names = ['BLOCK_SIZE_' + c for c in 'XYZ']
    grid_div_x = [f'max(BLOCK_SIZE_X * TILING_FACTOR_X - {cta_padding}, 1)']
    grid_div_y = [f'max(BLOCK_SIZE_Y * TILING_FACTOR_Y - {cta_padding}, 1)']
    grid_div_z = [f'max(BLOCK_SIZE_Z * TILING_FACTOR_Z - {cta_padding}, 1)']

    problem_size = (grid.itot, grid.jtot, grid.ktot if nlayers is None else nlayers)
    print(f'tuning {kernel_name} on {problem_size} for {float_name(grid.TF)}')

    # Calculate the number of iterations
    total_points = problem_size[0] * problem_size[1] * problem_size[2]
    iterations = int(math.ceil(1e7 / total_points))
    iterations = min(max(5, iterations), 100)  # Clamp between 5 and 100

    # Compiler flags
    current_dir = os.path.dirname(os.path.abspath(__file__))
    flags = [
        "-I", current_dir,
        "-I", current_dir + "/../include",
        "--define-macro", "float_type={}".format(float_name(grid.TF)),
        "--define-macro", "RESTRICTKEYWORD=__restrict__",
        "--define-macro", "USECUDA=1",
        "--extended-lambda",
        "-std=c++17",
        '-Xptxas="-v"',
    ]

    # True answer from the kernel without optimizations
    params = OrderedDict((key, values[0]) for key, values in tune_params.items())
    params['BLOCK_SIZE_X'] = params['BLOCK_SIZE_Y'] = params['BLOCK_SIZE_Z'] = 1 + cta_padding

    unopt_flags = ['-O0', '-Xcicc', '-O0', '-Xptxas', '-O0']
    outputs = kernel_tuner.run_kernel(
        kernel_name,
        kernel_source,
        problem_size,
        args,
        params,
        block_size_names=block_size_names,
        compiler_options=flags + unopt_flags,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        grid_div_z=grid_div_z,
        lang=lang,
    )

    # The answers are buffers that differ from the inputs (i.e, that have been written to)
    answers = [None] * len(outputs)
    for index, output in enumerate(outputs):
        count = np.sum(~np.isfinite(output))

        if count > 0:
            raise RuntimeError(f'error: argument {index} of {kernel_name} contains {count} non-finite values!')

        if np.any(output != args[index]):
            average = np.average(output)
            median = np.median(output[output != 0])
            nzeros = np.sum(output == 0)
            print(f'argument {index} of {kernel_name}: average: {average}, median: {median}, zeros={nzeros}')
            answers[index] = output

    def verify(answers, results, atol=None):
        is_valid = True

        for index, (answer, result) in enumerate(zip(answers, results)):
            if answer is None or np.allclose(answer, result, atol=atol):
                continue

            is_valid = False
            invalid = ~np.isclose(answer, result, atol=atol)
            count = np.sum(invalid)
            fraction = np.average(invalid) * 100
            print(f'argument {index} of {kernel_name} is invalid: {count} errors ({fraction}%)')

            for i in np.argwhere(invalid)[:10]:
                print(f' * index {i}: {answer[i]} != {result[i]}')


        return is_valid

    # Tune it!
    return kernel_tuner.tune_kernel(
        kernel_name,
        kernel_source,
        problem_size,
        args,
        tune_params,
        compiler_options=flags,
        restrictions=restrictions,
        iterations=iterations,
        cache=cache_file,
        block_size_names=block_size_names,
        grid_div_x=grid_div_x,
        grid_div_y=grid_div_y,
        grid_div_z=grid_div_z,
        #quiet=True,
        strategy=strategy,
        strategy_options=strategy_options,
        answer=answers,
        verify=verify,
        atol=1e-12,
        #lang='cupy',  # TODO: cupy fails, why?
        lang=lang,
    )

def store_results(filename, configs, env):
    best_config = min(configs, key=lambda r: r['time'])

    if os.path.exists(filename):
        with open(filename, 'r') as f:
            file_config = json.load(f)

            if file_config['time'] < best_config['time']:
                return file_config

    time = best_config.pop('time')
    times = best_config.pop('times')

    data = dict(
        date=datetime.now().isoformat(),
        config=best_config,
        time=time,
        times=times,
        hostname=platform.node(),
        env=env,
    )

    new_filename = filename + '.tmp'
    with open(new_filename, 'w') as f:
        json.dump(data, f, sort_keys=True, indent=4)

    os.replace(new_filename, filename)  # TODO: is this an atomic operation?
    return data


def tune_and_store(grid, args, kernel_name, kernel_source, key=None, **kwargs):
    if key is None: key = kernel_name
    experiment_key = f"{key}_{grid.itot}x{grid.jtot}x{grid.ktot}_{float_name(grid.TF)}_{device_name()}"
    cache_file = f'cache/{experiment_key}.json'
    results_file = f'results/{experiment_key}.json'

    results_a, env = tune_kernel(grid, args, kernel_name, kernel_source, cache_file, False, **kwargs)
    store_results(results_file, results_a, env)

    results_b, env = tune_kernel(grid, args, kernel_name, kernel_source, cache_file, True, **kwargs)
    return store_results(results_file, results_b, env)


if __name__ == '__main__':
    TF = np.float32
    grid   = Grid(3200, 3200, 3200, 32, 32, 32, 2, 1, TF)
    fields = Fields(['u', 'v', 'w', 's'], grid.ncells, grid.ijcells, grid.kcells, TF)
