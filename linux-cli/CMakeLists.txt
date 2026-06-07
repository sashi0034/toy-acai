cmake_minimum_required(VERSION 3.16)

project(toy-acai-linux-cli LANGUAGES CXX)

include("${CMAKE_CURRENT_LIST_DIR}/../linux/FindSiv3D.cmake")

add_executable(toy-acai
	../src/BattlefieldContext.cpp
	../src/BattlefieldRenderer.cpp
	../src/Main.cpp
)

target_compile_features(toy-acai PRIVATE cxx_std_20)
target_compile_options(toy-acai PRIVATE -Wall -Wextra -Wpedantic -msse4.1)
target_link_libraries(toy-acai PRIVATE Siv3D::Siv3D)

add_custom_target(toy-acai-copy-resources ALL
	COMMAND "${CMAKE_COMMAND}" -E copy_directory
		"${SIV3D_RESOURCE_DIR}"
		"$<TARGET_FILE_DIR:toy-acai>/resources"
	COMMENT "Copying Siv3D engine resources"
)
add_dependencies(toy-acai-copy-resources toy-acai)

enable_testing()
add_test(NAME toy-acai COMMAND toy-acai)
set_tests_properties(toy-acai PROPERTIES
	WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
