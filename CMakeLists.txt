cmake_minimum_required(VERSION 3.20)
project(clear_day_analysis LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# pybind11 provided by your Python build env (pip install pybind11)
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

# Build the external submodule library target: sunpos
add_subdirectory(external/Updated-PSA-sun-position-algorithm)

# Build the Python extension module
pybind11_add_module(_sunpos cpp/bindings/sunpos_pybind.cpp)

# Link against the sunpos static library from the submodule
target_link_libraries(_sunpos PRIVATE sunpos)

# Install extension into the package directory inside the wheel
install(TARGETS _sunpos
  LIBRARY DESTINATION clear_day_analysis
  RUNTIME DESTINATION clear_day_analysis
)