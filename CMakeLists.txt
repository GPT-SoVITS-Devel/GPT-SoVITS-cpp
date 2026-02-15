cmake_minimum_required(VERSION 3.17)

if (POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif ()

project(gsv_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==============================================================================
# 编译选项配置
# ==============================================================================
option(USE_ONNX "Build with ONNX Runtime backend" ON)
option(USE_SYSTEM_ONNXRUNTIME "Prefer system ONNX Runtime over manual path" OFF)
option(ONNXRUNTIME_LIB_PATH "ONNX Runtime Lib PATH" "")
option(ONNXRUNTIME_INCLUDE_PATH "ONNX Runtime Include PATH" "")
option(USE_ONNX "Build with ONNX Runtime backend" ON)
option(USE_TENSORRT "Build with TensorRT backend" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(USE_U8PATH "Enable u8path support" ON)
# TODO: 预留其他后端选项,如 USE_OPENVINO

# TensorRT 必须依赖 CUDA
if (USE_TENSORRT AND NOT ENABLE_CUDA)
  message(STATUS "TensorRT requires CUDA. Auto enabling ENABLE_CUDA.")
  set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA" FORCE)
endif ()

# 至少需要开启一个推理后端
if (NOT USE_ONNX AND NOT USE_TENSORRT)
  message(FATAL_ERROR "At least one inference backend (USE_ONNX or USE_TENSORRT) must be enabled.")
endif ()

if (USE_U8PATH)
  add_compile_definitions(_USE_U8PATH_=1)
endif ()

# ==============================================================================
# 依赖引入
# ==============================================================================
include(cmake/base.cmake)
include(cmake/JSON.cmake)
include(cmake/fmt.cmake)
include(cmake/cpp-pinyin.cmake)
include(cmake/boost-cmake.cmake)
include(cmake/tokenizers-cpp.cmake)
include(cmake/libsndfile.cmake)
include(cmake/libsamplerate.cmake)
include(cmake/SRELL.cmake)
include(cmake/xsimd.cmake)
include(cmake/xtensor.cmake)
include(cmake/xtensor-blas.cmake)
include(cmake/utfcpp.cmake)
include(cmake/cld2-cmake.cmake)
include(cmake/cppjieba.cmake)

# open_jtalk+hts_engine

if(NOT TARGET hts_engine)
  add_definitions(-DCHARSET_UTF_8 -DMECAB_CHARSET=utf-8 -DMECAB_UTF8_USE_ONLY)
  add_definitions(-DHAVE_CONFIG_H
      -DDIC_VERSION=102
      -DMECAB_DEFAULT_RC="dummy"
      -DMECAB_WITHOUT_SHARE_DIC
      -DPACKAGE="open_jtalk"
      -DVERSION="${OPEN_JTALK_VERSION}")
  cppmodule_add_subdirectory(hts_engine "${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/hts_engine")
  set(HTS_ENGINE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/hts_engine/include)
  set(HTS_ENGINE_LIB hts_engine_API)
endif()
if(NOT TARGET openjtalk)
  cppmodule_add_subdirectory(openjtalk "${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/open_jtalk")
  include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/g2p/open_jtalk)
endif ()

# ==============================================================================
# 源码管理
# ==============================================================================
file(GLOB_RECURSE GPT_SOVITS_CPP_SOURCE
    "src/*.cpp"
    "src/bert/*.cpp"
    "src/g2p/*.cpp"
    "src/text/*.cpp"
    "src/export/*.cpp"
)

add_library(gsv_lib ${GPT_SOVITS_CPP_SOURCE})
target_include_directories(gsv_lib PUBLIC include)


# ==============================================================================
# 后端配置与链接
# ==============================================================================
# 收集需要链接的后端库
set(BACKEND_LIBS "")

# -----------------
# CUDA 通用配置
# -----------------
if (ENABLE_CUDA)
  set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "Path to specific CUDA Toolkit version")

  if (CUDA_TOOLKIT_ROOT_DIR)
    set(CUDAToolkit_ROOT "${CUDA_TOOLKIT_ROOT_DIR}")
  endif ()

  target_compile_definitions(gsv_lib PRIVATE _ENABLE_CUDA_=1)
  find_package(CUDAToolkit REQUIRED)

  target_compile_definitions(gsv_lib PUBLIC WITH_CUDA)
  list(APPEND BACKEND_LIBS CUDA::cudart)

endif ()

# -----------------
# ONNX Runtime 配置
# -----------------
if (USE_ONNX)
  target_compile_definitions(gsv_lib PUBLIC WITH_ONNX)
  set(ORT_FOUND FALSE)

  if (USE_SYSTEM_ONNXRUNTIME)
    find_package(onnxruntime REQUIRED)
    if (onnxruntime_FOUND)
      message(STATUS "Found system ONNX Runtime via Config.")

      target_link_libraries(gsv_lib PUBLIC onnxruntime)
      set(ORT_FOUND TRUE)
    else()
      find_path(ORT_INC_SYSTEM
              NAMES onnxruntime_c_api.h
              PATHS ${HOMEBREW_PREFIX}/include/onnxruntime /usr/local/include/onnxruntime
              PATH_SUFFIXES onnxruntime)

      find_library(ORT_LIB_SYSTEM
              NAMES onnxruntime
              PATHS ${HOMEBREW_PREFIX}/lib /usr/local/lib)

      if (ORT_LIB_SYSTEM AND ORT_INC_SYSTEM)
        message(STATUS "Found system ONNX Runtime: ${ORT_INC_SYSTEM}")
        target_include_directories(gsv_lib PUBLIC ${ORT_INC_SYSTEM})
        target_link_libraries(gsv_lib PUBLIC ${ORT_LIB_SYSTEM})
        set(ORT_FOUND TRUE)
      endif()
    endif()
  endif()

  # 手动路径模式
  if (NOT ORT_FOUND)
#    if (NOT ONNXRUNTIME_PATH AND (NOT ONNXRUNTIME_INCLUDE_PATH OR NOT ONNXRUNTIME_LIB_PATH))
#      message(FATAL_ERROR "USE_ONNX is ON but ONNXRUNTIME_PATH/ONNXRUNTIME_INCLUDE_PATH/ONNXRUNTIME_LIB_PATH is not defined and system ORT not found.")
#    endif ()

    message(STATUS "Using manual ONNX Runtime path: ${ONNXRUNTIME_PATH}")
    target_include_directories(gsv_lib PUBLIC ${ONNXRUNTIME_PATH}/include)
    target_link_directories(gsv_lib PUBLIC ${ONNXRUNTIME_PATH}/lib)

    # 基础库
    list(APPEND BACKEND_LIBS onnxruntime)
    if()
      list(APPEND BACKEND_LIBS onnxruntime_providers_shared)
    endif ()
    if (ENABLE_CUDA)
      list(APPEND BACKEND_LIBS onnxruntime_providers_cuda)
    endif ()
  endif()
  if (ONNXRUNTIME_INCLUDE_PATH)
    target_include_directories(gsv_lib PUBLIC ${ONNXRUNTIME_INCLUDE_PATH})
  endif ()
  if (ONNXRUNTIME_LIB_PATH)
    target_link_libraries(gsv_lib PUBLIC ${ONNXRUNTIME_LIB_PATH})
  endif ()
endif ()

# -----------------
# TensorRT 配置
# -----------------
if (USE_TENSORRT)
  if (NOT TENSORRT_PATH)
    message(FATAL_ERROR "USE_TENSORRT is ON but TENSORRT_PATH is not defined.")
  endif ()

  # CUDNN
  if (ENABLE_CUDA AND CUDNN_PATH)
    target_include_directories(gsv_lib PUBLIC ${CUDNN_PATH}/include)
    target_link_directories(gsv_lib PUBLIC ${CUDNN_PATH}/lib)
    list(APPEND BACKEND_LIBS cudnn)
  endif()

  target_compile_definitions(gsv_lib PUBLIC WITH_TENSORRT)
  target_include_directories(gsv_lib PUBLIC ${TENSORRT_PATH}/include)

  macro(find_trt_lib _VAR_NAME _LIB_NAME)
    find_library(${_VAR_NAME}
        NAMES ${_LIB_NAME} ${_LIB_NAME}_10 ${_LIB_NAME}_100 ${_LIB_NAME}_8 ${_LIB_NAME}_9
        PATHS ${TENSORRT_PATH}/lib
        NO_DEFAULT_PATH
        REQUIRED
    )
  endmacro()

  find_trt_lib(TRT_NVINFER        nvinfer)
  find_trt_lib(TRT_NVINFER_PLUGIN nvinfer_plugin)
  find_trt_lib(TRT_NVONNXPARSER   nvonnxparser)

  list(APPEND BACKEND_LIBS
      ${TRT_NVINFER}
      ${TRT_NVINFER_PLUGIN}
      ${TRT_NVONNXPARSER}
  )
endif ()

target_link_libraries(gsv_lib PUBLIC
    cppmodule::base
    cppmodule::json
    cppmodule::fmt
    cppmodule::pinyin
    cppmodule::boost
    cppmodule::srell
    cppmodule::tokenizers
    cppmodule::sndfile
    cppmodule::xsimd
    cppmodule::xtensor
    cppmodule::xtensor-blas
    cppmodule::utfcpp
    cppmodule::cld2_cmake
    cppmodule::cppjieba
    cppmodule::samplerate
    openjtalk
    ${BACKEND_LIBS}
)

function(target_copy_shared_libs _target _base_path _sub_dirs _patterns)
  if(NOT EXISTS "${_base_path}")
    message(WARNING "Path not found: ${_base_path}, skip copying dlls for patterns: ${_patterns}")
    return()
  endif()

  set(_found_files "")

  # 遍历可能的子目录(bin, lib)查找DLL
  foreach(_dir ${_sub_dirs})
    foreach(_pattern ${_patterns})
      file(GLOB _files "${_base_path}/${_dir}/${_pattern}")
      list(APPEND _found_files ${_files})
    endforeach()
  endforeach()

  # 去重,防止bin和lib下有同名文件导致冲突
  list(REMOVE_DUPLICATES _found_files)

  if(_found_files)
    foreach(_file ${_found_files})
      # 获取文件名用于显示日志
      get_filename_component(_file_name ${_file} NAME)

      add_custom_command(TARGET ${_target} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_file}"
          "$<TARGET_FILE_DIR:${_target}>"
          COMMENT "[Post-Build] Copying dependency: ${_file_name}"
      )
    endforeach()
  else()
    message(WARNING "No DLLs found in ${_base_path} matching ${_patterns}")
  endif()
endfunction()

# CUDA DLLs
if(ENABLE_CUDA AND DEFINED CUDA_TOOLKIT_ROOT_DIR)
  target_copy_shared_libs(gsv_lib
      "${CUDA_TOOLKIT_ROOT_DIR}"
      "bin"
      "cudart64_*.dll;cublas64_*.dll;cublasLt64_*.dll;cufft64_*.dll;curand64_*.dll"
  )
endif()

# TensorRT
if(DEFINED TENSORRT_PATH)
  target_copy_shared_libs(gsv_lib
      "${TENSORRT_PATH}"
      "lib;bin"
      "nvinfer*.dll;nvinfer_plugin*.dll;nvonnxparser*.dll"
  )
endif()

# cuDNN
if(DEFINED CUDNN_PATH)
  target_copy_shared_libs(gsv_lib
      "${CUDNN_PATH}"
      "bin"
      "cudnn*.dll"
  )
endif()

# 处理 ONNXRuntime
if(DEFINED ONNXRUNTIME_PATH)
  target_copy_shared_libs(gsv_lib
      "${ONNXRUNTIME_PATH}"
      "lib;bin"
      "onnxruntime*.dll;onnxruntime_providers_cuda.dll;onnxruntime_providers_shared.dll"
  )
endif()


if (NOT NO_TEST)
  include(example/example_all.cmake)
endif ()