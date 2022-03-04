# Snellius @ SURFSara.
#
# NOTE: for Intel, you need to compile NetCDF yourself with EasyBuild.
# See notes at: https://github.com/microhh/microhh/issues/73
#
# GCC:
# module purge
# module load 2021
# module load CMake/3.20.1-GCCcore-10.3.0
# module load foss/2021a
# module load netCDF/4.8.0-gompi-2021a
# module load CUDA/11.3.1
#
# Intel:
# module purge
# module load 2021
# module load CMake/3.20.1-GCCcore-10.3.0
# module load intel/2021a
# module load netCDF/4.8.0-iimpi-2021a
# module load FFTW/3.3.9-intel-2021a
#

# Switch between Intel and GCC:
set(USEINTEL FALSE)

# GPU builds are always with GCC:
if(USECUDA)
    set(USEINTEL FALSE)
endif()

# Select correct compilers for Intel/GCC + parallel/serial:
if(USEMPI)
    if(USEINTEL)
        set(ENV{CC} mpiicc )
        set(ENV{CXX} mpiicpc)
	set(ENV{FC} mpiifort)
    else()
        set(ENV{CC} mpicc )
        set(ENV{CXX} mpicxx)
        set(ENV{FC} mpif90)
    endif()
else()
    if(USEINTEL)
        set(ENV{CC} icc )
        set(ENV{CXX} icpc)
        set(ENV{FC} ifort)
    else()
        set(ENV{CC} gcc )
        set(ENV{CXX} g++)
        set(ENV{FC} gfortran)
    endif()
endif()

# Set compiler flags / options:
if(USECUDA)
    set(USER_CXX_FLAGS "-std=c++14 -fopenmp")
    set(USER_CXX_FLAGS_RELEASE "-Ofast -march=native")
    add_definitions(-DRESTRICTKEYWORD=__restrict__)
else()
    if(USEINTEL)
        set(USER_CXX_FLAGS "-std=c++14 -restrict")
        set(USER_CXX_FLAGS_RELEASE "-Ofast -march=native")
        add_definitions(-DRESTRICTKEYWORD=restrict)
    else()
        set(USER_CXX_FLAGS "-std=c++14")
        set(USER_CXX_FLAGS_RELEASE "-Ofast -march=native -mfma -mavx2 -m3dnow -fomit-frame-pointer")

        set(USER_FC_FLAGS "-fdefault-real-8 -fdefault-double-8 -fPIC -ffixed-line-length-none -fno-range-check")
        set(USER_FC_FLAGS_RELEASE "-DNDEBUG -Ofast -march=native -mfma -mavx2 -m3dnow -fomit-frame-pointer")

        add_definitions(-DRESTRICTKEYWORD=__restrict__)
    endif()
endif()

set(NETCDF_INCLUDE_DIR $ENV{NETCDF_INC})
set(FFTW_INCLUDE_DIR   $ENV{FFTW_INC})
set(BOOST_INCLUDE_DIR  $ENV{BOOST_INC})
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR} ${BOOST_INCLUDE_DIR})


set(NETCDF_LIB_C $ENV{NETCDF_LIB}/libnetcdf.so)
set(FFTW_LIB     $ENV{FFTW_LIB}/libfftw3.so)
set(FFTWF_LIB    $ENV{FFTW_LIB}/libfftw3f.so)
set(HDF5_LIB     $ENV{HDF5_LIB}/libhdf5.so)
set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z)

# Disable MPI-IO for cross-sections on GPFS file systems.
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
add_definitions(-DDISABLE_2D_MPIIO=1)

if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(LIBS ${LIBS} -rdynamic cufft)
    set(USER_CUDA_NVCC_FLAGS "-arch=sm_80 --use_fast_math")
    list(APPEND CUDA_NVCC_FLAGS "-std=c++14 --expt-relaxed-constexpr")
endif()
