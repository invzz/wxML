#! /bin/bash
if [[ -z $PRJ_ROOT ]]; then 
    echo "PRJ_ROOT is not set. setting it to $PWD"
    export PRJ_ROOT=$PWD
fi
dir=$( echo "$1" | tr "[:lower:]" "[:upper:]" )
echo dir = $dir
model_name_lower=$(echo "$dir" | tr "A-Z" "a-z" )
echo low = $model_name_lower
mkdir -p $PRJ_ROOT/lib/$dir/inc $PRJ_ROOT/lib/$dir/src
cmake_file=$PRJ_ROOT/lib/$dir/CMakeLists.txt
touch $PRJ_ROOT/lib/$dir/CMakeLists.txt
touch $PRJ_ROOT/lib/$dir/inc/"$model_name_lower.hh"
touch $PRJ_ROOT/lib/$dir/src/"$model_name_lower.cc"
echo -e "set(LIB_NAME $model_name_lower)\nproject(\${LIB_NAME})\n" >> $cmake_file
echo -e "include_directories(BEFORE \${PROJECT_SOURCE_DIR}/src\n\${PROJECT_SOURCE_DIR}/inc\n)">> $cmake_file
echo -e "set(SRC \${PROJECT_SOURCE_DIR}/src/\${LIB_NAME}.cc)">> $cmake_file
echo -e "set(INC \${PROJECT_SOURCE_DIR}/inc/\${LIB_NAME}.hh)">> $cmake_file
echo -e "add_library(\${LIB_NAME} \${SRC} \${INC})" >> $cmake_file
# Add CMakeLists.txt
echo "add_subdirectory(\${LIB_DIR}/$dir)" >> $PRJ_ROOT/cmake/targets.cmake
# Add guard to header file
echo "#ifndef _H_$model_name_lower" >> $PRJ_ROOT/lib/$dir/inc/$model_name_lower.hh
echo "#define _H_$model_name_lower" >> $PRJ_ROOT/lib/$dir/inc/$model_name_lower.hh
echo "#endif" >> $PRJ_ROOT/lib/$dir/inc/$model_name_lower.hh