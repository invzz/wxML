set (LIB_NAME mainframe)
# Include the header files from the child directory
include_directories(
    ${wxWidgets_INCLUDE_DIRS}
)
find_package (Eigen3 3.3 REQUIRED NO_MODULE) 
# Source files
set(SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/main.frame.cc   ${CMAKE_CURRENT_SOURCE_DIR}/src/datamodel.control.cc )
# Header files
set(INC  )
# Executable 
add_library(${LIB_NAME} ${SRC} ${INC}) 

# Link wxWidgets
target_link_libraries(${LIB_NAME} PRIVATE  ${wxWidgets_LIBRARIES} eig_neuron)
target_link_libraries(${LIB_NAME} PRIVATE Eigen3::Eigen )

set(INCLUDEDIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/../inc/
    ${CMAKE_CURRENT_SOURCE_DIR}/../../nn_eigen/inc/
)

message(STATUS "Includedir: ${INCLUDEDIR}")

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  ${INCLUDEDIR} )
add_dependencies(${LIB_NAME} eig_neuron)




