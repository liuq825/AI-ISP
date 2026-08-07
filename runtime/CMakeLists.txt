cmake_minimum_required(VERSION 3.20)
project(ai_isp_dark_preview_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(ai_isp_runtime STATIC
    src/profile_selector.cpp
    src/dark_trigger.cpp
    src/raw_formatter.cpp
    src/dmabuf_pool.cpp
    src/npu_executor.cpp
    src/failsafe.cpp
)
target_include_directories(ai_isp_runtime PUBLIC include)
if(MSVC)
    # 源码与中文注释统一为 UTF-8，避免受系统代码页影响。
    target_compile_options(ai_isp_runtime PRIVATE /utf-8)
endif()

add_executable(ai_isp_runtime_tests tests/runtime_tests.cpp)
target_link_libraries(ai_isp_runtime_tests PRIVATE ai_isp_runtime)
if(MSVC)
    target_compile_options(ai_isp_runtime_tests PRIVATE /utf-8)
endif()

enable_testing()
add_test(NAME ai_isp_runtime_tests COMMAND ai_isp_runtime_tests)
