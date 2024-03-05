set (LIB_NAME mainframe)
# Include the header files from the child directory
include_directories(
    ${wxWidgets_INCLUDE_DIRS}
)
# Source files
set(SRC ${LIB_NAME}.cc  appleListControl.cc )
# Header files
set(INC ${LIB_NAME}.hh appleData.hh csvListControl.hh )
# Executable 
add_library(${LIB_NAME} ${SRC} ${INC}) 

# Link wxWidgets
target_link_libraries(${LIB_NAME} PRIVATE  ${wxWidgets_LIBRARIES} eig_neuron)

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ../inc/ ../../nn_eigen/inc/ )





