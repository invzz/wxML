set (LIB_NAME neuron)
# Include the header files from the child directory
include_directories(
    ${wxWidgets_INCLUDE_DIRS}
)
# Source files
set(SRC ${LIB_NAME}.cc  )
# Header files
set(INC ${LIB_NAME}.hh  )
# Executable 
add_library(${LIB_NAME} ${SRC} ${INC}) 

# Link wxWidgets
target_link_libraries(${LIB_NAME} PRIVATE  ${wxWidgets_LIBRARIES})

target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} )

set_property(GLOBAL PROPERTY NN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})



