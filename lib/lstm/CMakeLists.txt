set (LIB_NAME lstm)
# Include the header files from the child directory
include_directories(
    ${wxWidgets_INCLUDE_DIRS}
)
find_package (Eigen3 3.3 REQUIRED NO_MODULE) 
# Source files
set(SRC   src/lstm.cc  )
# Header files
set(INC   inc/lstm.hh inc/activation.hh)
# Executable 
add_library(${LIB_NAME} ${SRC} ${INC}) 

# Link wxWidgets
#target_link_libraries(${LIB_NAME} PRIVATE  ${wxWidgets_LIBRARIES} )
target_link_libraries(${LIB_NAME} PRIVATE Eigen3::Eigen )

set(INCLUDEDIR 
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

message(STATUS "Includedir: ${INCLUDEDIR}")

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}  ${INCLUDEDIR})





