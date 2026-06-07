set(SIV3D_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../siv3d/OpenSiv3D" CACHE PATH "Path to the OpenSiv3D source tree")

get_filename_component(SIV3D_ROOT "${SIV3D_ROOT}" ABSOLUTE)
set(SIV3D_INCLUDE_DIR "${SIV3D_ROOT}/Siv3D/include")
set(SIV3D_LIBRARY "${SIV3D_ROOT}/Linux/build/libSiv3D.a")
set(SIV3D_RESOURCE_DIR "${SIV3D_ROOT}/Linux/App/resources")

if(NOT EXISTS "${SIV3D_INCLUDE_DIR}/Siv3D.hpp")
	message(FATAL_ERROR "Siv3D.hpp was not found. Set -DSIV3D_ROOT=/path/to/OpenSiv3D")
endif()

if(NOT EXISTS "${SIV3D_LIBRARY}")
	message(FATAL_ERROR "libSiv3D.a was not found. Expected: ${SIV3D_LIBRARY}")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(SIV3D_THIRD_PARTY REQUIRED
	alsa
	libavcodec
	libavformat
	libavutil
	libcurl
	freetype2
	gl
	glib-2.0
	gtk+-3.0
	harfbuzz
	libmpg123
	ogg
	opencv4
	opus
	opusfile
	libpng
	soundtouch
	libswresample
	libtiff-4
	libturbojpeg
	uuid
	vorbis
	vorbisenc
	vorbisfile
	libwebp
	x11
	glu
	xft
	zlib
)

find_package(Boost 1.71.0 REQUIRED)
find_package(Threads REQUIRED)
find_package(GIF REQUIRED)

add_library(Siv3D::Siv3D STATIC IMPORTED)
set_target_properties(Siv3D::Siv3D PROPERTIES
	IMPORTED_LOCATION "${SIV3D_LIBRARY}"
	INTERFACE_INCLUDE_DIRECTORIES "${SIV3D_INCLUDE_DIR};${SIV3D_INCLUDE_DIR}/ThirdParty;${SIV3D_THIRD_PARTY_INCLUDE_DIRS};${Boost_INCLUDE_DIRS}"
	INTERFACE_COMPILE_OPTIONS "${SIV3D_THIRD_PARTY_CFLAGS_OTHER}"
	INTERFACE_COMPILE_DEFINITIONS "__LINUX_ALSA__;AS_USE_NAMESPACE;MUPARSER_STATIC;_UNICODE;WITH_MINIAUDIO;WITH_NOSOUND"
	INTERFACE_LINK_DIRECTORIES "${SIV3D_THIRD_PARTY_LIBRARY_DIRS}"
	INTERFACE_LINK_OPTIONS "${SIV3D_THIRD_PARTY_LDFLAGS_OTHER}"
	INTERFACE_LINK_LIBRARIES "Threads::Threads;${GIF_LIBRARIES};${SIV3D_THIRD_PARTY_LIBRARIES};${CMAKE_DL_LIBS}"
)
