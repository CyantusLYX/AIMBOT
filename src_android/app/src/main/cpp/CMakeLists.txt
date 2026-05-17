cmake_minimum_required(VERSION 3.22.1)

project(bytetrack_jni)

add_library(
    bytetrack_jni
    SHARED
    bytetrack_jni.cpp
)

target_compile_features(bytetrack_jni PRIVATE cxx_std_17)

find_library(log-lib log)

target_link_libraries(
    bytetrack_jni
    ${log-lib}
)
