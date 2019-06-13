project(kima)
# Just for Cmake to stop complaining
cmake_minimum_required(VERSION 3.5)
# Both kima and Dnest4 need c++11 so might as well set it globally for now
set(CMAKE_CXX_STANDARD 11)

add_executable(kima)
target_compile_definitions(kima PUBLIC EIGEN_MPL2_ONLY)
# target_compile_options(kima PRIVATE -pthread)
set(source_dir ${CMAKE_CURRENT_SOURCE_DIR}/src)
file(GLOB sources
    ${source_dir}/*.cpp
    ${source_dir}/distributions/*.cpp)
target_sources(kima PUBLIC ${sources})
# Include .h from kima
target_include_directories(kima PUBLIC ${source_dir})
# Include Eigen
target_include_directories(kima PUBLIC eigen)


# ----------------------- DNEST4 --------------------------------
# Compile Dnest4 lib - Building everything here to not mess with DNest4 files
add_library(dnest4 STATIC)

# Define DNest4 sources and add include dirs
set(dnest4_dir ${CMAKE_CURRENT_SOURCE_DIR}/DNest4/code)
file(GLOB dnest4_sources
    ${dnest4_dir}/*.cpp
    ${dnest4_dir}/Distributions/*.cpp
    ${dnest4_dir}/RJObject/ConditionalPriors/*.cpp)
target_sources(dnest4 PRIVATE ${dnest4_sources})
target_include_directories(dnest4 PUBLIC ${dnest4_dir})

# Link DNest4 lib. Pthread comes from the PUBLIC interface
target_link_libraries(kima PUBLIC dnest4)



# OS-specific definitions
# ---------------------------------------------------------------
# If we are on windows, handle the lack of unistd and getopt files from DNest4
if (WIN32)
    set(libs_dir ${CMAKE_CURRENT_SOURCE_DIR}/libs/unistd)
    file(GLOB libs_sources
        ${libs_dir}/*.c)
    target_sources(dnest4 PRIVATE ${libs_sources})
    target_include_directories(dnest4 PRIVATE ${libs_dir})
endif()

# Debug compile options and definitions (can't target release and debug specifically yet). 
# UNIX: Add pthread for threading
# WINDOWS: Define _USE_MATH_DEFINES to have M_PI and M_PI_2 definitions
# Extra info: iostream includes cmath in MSVC so we need definition at compiler level
if (WIN32)
    # target_compile_options(dnest4 PUBLIC -Wall)
    target_compile_definitions(kima PUBLIC _USE_MATH_DEFINES)
else()
    # target_compile_options(dnest4 PUBLIC -Wall -Wextra -Wpedantic -Wconversion)
    target_link_libraries(dnest4 PUBLIC pthread)
endif()
