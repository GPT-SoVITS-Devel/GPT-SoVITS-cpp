
include_directories(${CMAKE_HOME_DIRECTORY}/include)

AUX_SOURCE_DIRECTORY(example/clean_text GPT_SOVITS_CPP_TEST_CLEAN_TEXT_SOURCE)
add_executable(gpt_sovits_cpp_test_clean_text ${GPT_SOVITS_CPP_TEST_CLEAN_TEXT_SOURCE})
target_link_libraries(gpt_sovits_cpp_test_clean_text PUBLIC gsv_lib)


add_executable(gpt_sovits_cpp_test_bert example/model/bert.cpp)
target_link_libraries(gpt_sovits_cpp_test_bert PUBLIC gsv_lib)


if (WIN32 AND COMMAND auto_copy_backend_dlls)
  auto_copy_backend_dlls(gpt_sovits_cpp_test_clean_text)
  auto_copy_backend_dlls(gpt_sovits_cpp_test_bert)
endif()
