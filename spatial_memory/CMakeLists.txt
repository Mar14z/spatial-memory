cmake_minimum_required(VERSION 3.16)
project(spatial_memory LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(SQLite3 REQUIRED)
find_package(Threads REQUIRED)

add_library(sm STATIC src/engine.cpp src/sqlite_store.cpp src/clip_tokenizer.cpp)
target_include_directories(sm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sm PUBLIC SQLite::SQLite3 Threads::Threads)
target_compile_options(sm PRIVATE -O2 -Wall -Wextra)

enable_testing()
find_package(GTest)
if(GTest_FOUND)
  add_executable(test_sm tests/test_sm.cpp)
  target_link_libraries(test_sm PRIVATE sm GTest::gtest_main pthread)
  add_test(NAME test_sm COMMAND test_sm)
else()
  message(WARNING "GTest not found, tests disabled")
endif()

add_executable(bench_sm bench/bench_engine.cpp)
target_link_libraries(bench_sm PRIVATE sm pthread)

add_executable(crash_test tools/crash_test.cpp)
target_link_libraries(crash_test PRIVATE sm pthread)

# ACL-dependent targets (only on the board with CANN sourced; skipped elsewhere)
find_library(ASCENDCL_LIB ascendcl PATHS $ENV{ASCEND_HOME_PATH}/lib64 NO_DEFAULT_PATH)
if(ASCENDCL_LIB)
  add_library(sm_acl STATIC src/clip_extractor.cpp src/clip_text_extractor.cpp)
  target_include_directories(sm_acl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(sm_acl PRIVATE $ENV{ASCEND_HOME_PATH}/include)
  target_link_libraries(sm_acl PUBLIC ${ASCENDCL_LIB})
  target_compile_options(sm_acl PRIVATE -O2)

  add_executable(clip_feature_test tools/clip_feature_test.cpp)
  target_link_libraries(clip_feature_test PRIVATE sm_acl pthread)

  add_executable(clip_text_test tools/clip_text_test.cpp)
  target_link_libraries(clip_text_test PRIVATE sm sm_acl pthread)
endif()
