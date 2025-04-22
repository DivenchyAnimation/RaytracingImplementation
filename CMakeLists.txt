# CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
CMAKE_MINIMUM_REQUIRED(VERSION 3.18) # 3.18 is required for CMake's CUDA support

# Default to hardware acceleration
OPTION(USE_CUDA "Enable hardware acceleration for NVIDIA GPU" ON)

# Name of the project
if(USE_CUDA)
    PROJECT(A6 LANGUAGES CXX CUDA)
ELSE()
    PROJECT(A6 LANGUAGES CXX)
ENDIF()

# Is this the solution?
# Override with `cmake -DSOL=ON ..`
OPTION(SOL "Solution" OFF)

# Use glob to get the list of all source files.
# We don't really need to include header and resource files to build, but it's
# nice to have them also show up in IDEs.
#IF(${SOL})
#  FILE(GLOB_RECURSE SOURCES "src0/*.cpp")
#  FILE(GLOB_RECURSE SOURCES "src0/*.cu")
#  FILE(GLOB_RECURSE SOURCES "src0/*.cuh")
#  FILE(GLOB_RECURSE HEADERS "src0/*.h")
#ELSE()
#  FILE(GLOB_RECURSE SOURCES "src/*.cpp")
#  FILE(GLOB_RECURSE SOURCES "src/*.cu")
#  FILE(GLOB_RECURSE SOURCES "src/*.cuh")
#  FILE(GLOB_RECURSE HEADERS "src/*.h")
#ENDIF()

IF(${SOL})
# find all .cpp
FILE(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src0/*.cpp")
# (optional) gather headers if you want them visible too
FILE(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/src0/*.h")
ELSE()
FILE(GLOB_RECURSE CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")
# (optional) gather headers if you want them visible too
FILE(GLOB_RECURSE HEADER_FILES "${CMAKE_SOURCE_DIR}/src/*.h")
ENDIF()


# Determine which main to use
IF(USE_CUDA)
	# find all .cu and .cuh
	FILE(GLOB_RECURSE CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/*.cu")
    # Remove duplicate main.cu from sources
    LIST(REMOVE_ITEM CUDA_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cu")
	SET(SRC_MAIN "${CMAKE_SOURCE_DIR}/src/main.cu")
	SET_SOURCE_FILES_PROPERTIES(${SRC_MAIN} PROPERTIES LANGUAGE CUDA)
    # Remove main.cpp from sources
    LIST(REMOVE_ITEM CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
ELSE()
    # find main.cpp
    SET(CUDA_SOURCES )
    # Remove duplicate main.cpp from sources
    LIST(REMOVE_ITEM CPP_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
    FILE(SRC_MAIN "${CMAKE_SOURCE_DIR}/src/main.cpp")
ENDIF()

# now combine into one list
set(ALL_SOURCES
    ${SRC_MAIN}
    ${CPP_SOURCES}
    ${CUDA_SOURCES}
    ${HEADER_FILES}   # optional, so they show up under "Header Files"
)

SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set the executable.
ADD_EXECUTABLE(${CMAKE_PROJECT_NAME} ${ALL_SOURCES})

IF(USE_CUDA)
    # Set CUDA properties/flags
    SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES CUDA_STANDARD 11 CUDA_SEPARABLE_COMPILATION ON CUDA_ARCHITECTURES 75)
ENDIF()

# Use c++17
SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
SET_TARGET_PROPERTIES(${CMAKE_PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Get the GLM environment variable. Since GLM is a header-only library, we
# just need to add it to the include directory.
SET(GLM_INCLUDE_DIR "$ENV{GLM_INCLUDE_DIR}")
IF(NOT GLM_INCLUDE_DIR)
  # The environment variable was not set
  SET(ERR_MSG "Please point the environment variable GLM_INCLUDE_DIR to the root directory of your GLM installation.")
  IF(WIN32)
    # On Windows, try the default location
    MESSAGE(STATUS "Looking for GLM in ${DEF_DIR_GLM}")
    IF(IS_DIRECTORY ${DEF_DIR_GLM})
      MESSAGE(STATUS "Found!")
      SET(GLM_INCLUDE_DIR ${DEF_DIR_GLM})
    ELSE()
      MESSAGE(FATAL_ERROR ${ERR_MSG})
    ENDIF()
  ELSE()
    MESSAGE(FATAL_ERROR ${ERR_MSG})
  ENDIF()
ENDIF()
INCLUDE_DIRECTORIES(${GLM_INCLUDE_DIR})


# OS specific options and libraries
IF(WIN32)
  # -Wall produces way too many warnings.
  # -pedantic is not supported.
  # Disable warning 4996.
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996")
  SET_PROPERTY(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${CMAKE_PROJECT_NAME})
ELSE()
  # Enable all pedantic warnings.
  # Added -g
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -pedantic")
ENDIF()

# Add precompiled headers
target_precompile_headers(${CMAKE_PROJECT_NAME} PRIVATE "src/pch.h")
