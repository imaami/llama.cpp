set(TARGET llama-reasoning-swap)

add_executable(${TARGET} reasoning-swap.cpp)
target_include_directories(${TARGET} PRIVATE ../server)
target_link_libraries(${TARGET} PRIVATE llama cpp-httplib vendor::nlohmann Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
