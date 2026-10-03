# The test binaries link the core library and are built as separate
# executables rather than through a framework. Each one exits non-zero on the
# first failure and prints a line per observation, which is what a developer
# running it by hand wants from a project that has no test dependency.

function(occ_add_test name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE occ_core)
    if(OCC_CXX_LIBS)
        target_link_libraries(${name} PRIVATE ${OCC_CXX_LIBS})
    endif()
    add_test(NAME ${name} COMMAND ${name})
endfunction()

occ_add_test(occ_test_seccomp test_seccomp.cpp)
occ_add_test(occ_test_container test_container.cpp)
