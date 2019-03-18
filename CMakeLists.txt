cmake_minimum_required(VERSION 2.8)
project(fast_sparse_interpolation CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -fmessage-length=0 -std=c++11 -D_GNU_SOURCE=1 -D_REENTRANT -Dlinux -D__linux__ -Dx86_64 -D__x86_64__")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g3 -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2") #-mtune=native -march=native
set(CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -L/usr/local/lib")

link_directories(/usr/local/lib)

file(GLOB SRC "src/*.cpp")

file(GLOB INC "src/*.hpp")
file(GLOB INC_TEST "test/*.hpp")

option(BUILD_TEST ON)

## Build application
add_library(fast_sparse_interpolation ${SRC})
include_directories("${PROJECT_SOURCE_DIR}/src")

if(BUILD_TEST)
    if(BUILD_TEST STREQUAL "ON")
        file(GLOB SRC_TEST "test/*.cpp")
        add_executable(fsi-test ${SRC_TEST})
        include_directories("${PROJECT_SOURCE_DIR}/src")
        target_link_libraries(fast_sparse_interpolation)
    endif()
endif()



install(TARGETS fast_sparse_interpolation DESTINATION lib)
install(FILES ${INC} DESTINATION include/sfcpp)






