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
option(USE_TENSORRT "Build with TensorRT backend" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
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
  if (NOT ONNXRUNTIME_PATH)
    message(FATAL_ERROR "USE_ONNX is ON but ONNXRUNTIME_PATH is not defined.")
  endif ()

  target_compile_definitions(gsv_lib PUBLIC WITH_ONNX)
  target_include_directories(gsv_lib PUBLIC ${ONNXRUNTIME_PATH}/include)
  target_link_directories(gsv_lib PUBLIC ${ONNXRUNTIME_PATH}/lib)

  # 基础库
  list(APPEND BACKEND_LIBS onnxruntime onnxruntime_providers_shared)

  if (ENABLE_CUDA)
    list(APPEND BACKEND_LIBS onnxruntime_providers_cuda)
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

#if (WIN32)
#  function(copy_dlls_to_runtime_dir search_path pattern)
#    file(TO_CMAKE_PATH "${search_path}" search_path_normalized)
#
#    if (NOT EXISTS "${search_path_normalized}")
#      return()
#    endif()
#
#    file(GLOB FOUND_DLLS "${search_path_normalized}/${pattern}")
#
#    if (NOT FOUND_DLLS)
#      return()
#    endif()
#
#    foreach(DLL_FILE ${FOUND_DLLS})
#      add_custom_command(TARGET gsv_lib POST_BUILD
#          COMMAND ${CMAKE_COMMAND} -E copy_if_different
#          "${DLL_FILE}"
#          "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
#          COMMENT "Setting up runtime env: Copying ${DLL_FILE}"
#      )
#    endforeach()
#  endfunction()
#
#  function(setup_backend_runtime)
#    # ONNX Runtime
#    if (USE_ONNX AND ONNXRUNTIME_PATH)
#      copy_dlls_to_runtime_dir("${ONNXRUNTIME_PATH}/lib" "*.dll")
#      copy_dlls_to_runtime_dir("${ONNXRUNTIME_PATH}/bin" "*.dll")
#    endif()
#
#    # CUDA / TensorRT (逻辑同上，简化展示)
#    if (ENABLE_CUDA)
#      # ... 你的 CUDA 查找逻辑 ...
#      # 记得把里面的 copy_dlls_to_target 替换为 copy_dlls_to_runtime_dir
#      # 传入路径即可，不需要传 target_name
#      # 示例:
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "cudart64_*.dll")
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "cublas64_*.dll")
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "cublasLt64_*.dll")
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "cufft64_*.dll")
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "curand64_*.dll")
#       copy_dlls_to_runtime_dir("${NORMALIZED_PATH}" "cublasLt64_*.dll")
#    endif()
#
#    if (USE_TENSORRT AND TENSORRT_PATH)
#      copy_dlls_to_runtime_dir("${TENSORRT_PATH}/lib" "*.dll")
#    endif()
#  endfunction()
#  setup_backend_runtime()
#
##  function(copy_dlls_to_target target_name search_path pattern)
##    # 正常情况下,Windows路径可能包含反斜杠,导致GLOB失败,需转为CMake内部格式
##    file(TO_CMAKE_PATH "${search_path}" search_path_normalized)
##
##    if (NOT EXISTS "${search_path_normalized}")
##      message(WARNING "[DLL Copy] Path not found: ${search_path_normalized}")
##      return()
##    endif()
##
##    file(GLOB FOUND_DLLS "${search_path_normalized}/${pattern}")
##
##    if (NOT FOUND_DLLS)
##      message(WARNING "[DLL Copy] No DLLs found in ${search_path_normalized} with pattern ${pattern}")
##      return()
##    endif()
##
##    # 正常情况下,应打印复制信息以便调试
##    message(STATUS "[DLL Copy] Found ${pattern} in ${search_path_normalized}, scheduling copy to target: ${target_name}")
##
##    foreach(DLL_FILE ${FOUND_DLLS})
##      # TODO: gsv_lib为静态库时,TargetDir为lib目录; 若需运行exe,需将DLL复制到bin目录或对exe目标调用此函数
##      add_custom_command(TARGET ${target_name} POST_BUILD
##          COMMAND ${CMAKE_COMMAND} -E copy_if_different
##          "${DLL_FILE}"
##          "$<TARGET_FILE_DIR:${target_name}>"
##          COMMENT "Copying dependency: ${DLL_FILE}"
##      )
##    endforeach()
##  endfunction()
##
##  function(auto_copy_backend_dlls target_name)
##    message(STATUS "Configuring auto-copy DLLs for target: ${target_name}...")
##
##    # Copy ONNX Runtime DLLs
##    if (USE_ONNX AND ONNXRUNTIME_PATH)
##      copy_dlls_to_target(${target_name} "${ONNXRUNTIME_PATH}/lib" "*.dll")
##      copy_dlls_to_target(${target_name} "${ONNXRUNTIME_PATH}/bin" "*.dll")
##    endif()
##
##    if (ENABLE_CUDA)
##      set(CUDA_SEARCH_PATHS "")
##
##      if (CUDAToolkit_LIBRARY_ROOT)
##        list(APPEND CUDA_SEARCH_PATHS "${CUDAToolkit_LIBRARY_ROOT}/bin")
##        list(APPEND CUDA_SEARCH_PATHS "${CUDAToolkit_LIBRARY_ROOT}/bin/x64")
##      endif()
##
##      if (DEFINED ENV{CUDA_PATH})
##        list(APPEND CUDA_SEARCH_PATHS "$ENV{CUDA_PATH}/bin")
##        list(APPEND CUDA_SEARCH_PATHS "$ENV{CUDA_PATH}/bin/x64")
##      endif()
##
##      list(REMOVE_DUPLICATES CUDA_SEARCH_PATHS)
##
##      foreach(SEARCH_PATH ${CUDA_SEARCH_PATHS})
##        # 规范化路径，处理斜杠差异
##        get_filename_component(NORMALIZED_PATH "${SEARCH_PATH}" ABSOLUTE)
##
##        if (EXISTS "${NORMALIZED_PATH}")
##          # message(STATUS "Searching CUDA DLLs in: ${NORMALIZED_PATH}")
##
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cudart64_*.dll")
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cublas64_*.dll")
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cublasLt64_*.dll")
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cufft64_*.dll")
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "curand64_*.dll")
##          copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cublasLt64_*.dll")
##
##          # cuDNN
##          if (NOT CUDNN_PATH)
##            copy_dlls_to_target(${target_name} "${NORMALIZED_PATH}" "cudnn*.dll")
##          endif()
##        endif()
##      endforeach()
##
##      # 如果显式指定了 CUDNN_PATH
##      if (CUDNN_PATH)
##        copy_dlls_to_target(${target_name} "${CUDNN_PATH}/bin" "cudnn*.dll")
##        copy_dlls_to_target(${target_name} "${CUDNN_PATH}/lib" "cudnn*.dll")
##      endif()
##    endif()
##
##    # TensorRT DLLs
##    if (USE_TENSORRT AND TENSORRT_PATH)
##      copy_dlls_to_target(${target_name} "${TENSORRT_PATH}/lib" "*.dll")
##      copy_dlls_to_target(${target_name} "${TENSORRT_PATH}/bin" "*.dll")
##    endif()
##  endfunction()
##
##  # 正常情况下,只有当 gsv_lib 为 SHARED 库时,复制到其目录才有运行时意义
##  auto_copy_backend_dlls(gsv_lib)
#endif()

if (NOT NO_TEST)
  include(example/example_all.cmake)
endif ()