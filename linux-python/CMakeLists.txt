cmake_minimum_required(VERSION 3.16)

project(toy-acai-linux-python LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../linux/FindSiv3D.cmake")

if(NOT Python_EXECUTABLE AND EXISTS "/usr/bin/python3.11")
	set(Python_EXECUTABLE "/usr/bin/python3.11" CACHE FILEPATH "Python interpreter for nanobind")
endif()

find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
	COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
	OUTPUT_STRIP_TRAILING_WHITESPACE
	OUTPUT_VARIABLE nanobind_ROOT
	ERROR_VARIABLE nanobind_error
	RESULT_VARIABLE nanobind_result
)
if(NOT nanobind_result EQUAL 0)
	message(FATAL_ERROR "nanobind was not found for ${Python_EXECUTABLE}. Install it with: ${Python_EXECUTABLE} -m pip install --user nanobind\n${nanobind_error}")
endif()

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(toy_acai_core
	../src/BattlefieldContext.cpp
	../src/BattlefieldRenderer.cpp
	../src/PythonBindings.cpp
)
target_compile_features(toy_acai_core PRIVATE cxx_std_20)
target_compile_options(toy_acai_core PRIVATE -Wall -Wextra -Wpedantic -msse4.1)
target_link_libraries(toy_acai_core PRIVATE Siv3D::Siv3D)

add_custom_target(toy-acai-python-copy-resources ALL
	COMMAND "${CMAKE_COMMAND}" -E copy_directory
		"${SIV3D_RESOURCE_DIR}"
		"$<TARGET_FILE_DIR:toy_acai_core>/resources"
	COMMENT "Copying Siv3D engine resources"
)
add_dependencies(toy-acai-python-copy-resources toy_acai_core)
