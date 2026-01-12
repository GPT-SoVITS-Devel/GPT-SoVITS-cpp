cmake_minimum_required(VERSION 3.15)
project(gsv_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

unset(CMAKE_MSVC_RUNTIME_LIBRARY)

# 依赖引入
include(cmake/base.cmake)
include(cmake/JSON.cmake)
include(cmake/fmt.cmake)
include(cmake/cpp-pinyin.cmake)
include(cmake/boost-cmake.cmake)
include(cmake/tokenizers-cpp.cmake)
include(cmake/libsndfile.cmake)
include(cmake/SRELL.cmake)
include(cmake/xtensor.cmake)
include(cmake/utfcpp.cmake)
include(cmake/cld2-cmake.cmake)
include(cmake/cppjieba.cmake)

#add_executable(gsv_onnx main.cpp)



file(GLOB_RECURSE GPT_SOVITS_CPP_SOURCE
    "src/*.cpp"
    "src/bert/*.cpp"
    "src/g2p/*.cpp"
    "src/text/*.cpp"
    "src/export/*.cpp"
)

add_library(gsv_lib ${GPT_SOVITS_CPP_SOURCE})
target_include_directories(gsv_lib PUBLIC include)

target_link_libraries(gsv_lib PUBLIC
    cppmodule::base
    cppmodule::json
    cppmodule::fmt
    cppmodule::pinyin
    cppmodule::boost
    cppmodule::srell
    cppmodule::tokenizers
    cppmodule::sndfile
    cppmodule::xtensor
    cppmodule::utfcpp
    cppmodule::cld2_cmake
    cppmodule::cppjieba
)

if (NOT BUILD_TEST)
  include(example/example_all.cmake)
endif ()