# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.20

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake

# The command to remove a file.
RM = /Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/face.dir/depend.make
# Include the progress variables for this target.
include CMakeFiles/face.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/face.dir/flags.make

CMakeFiles/face.dir/face.cpp.o: CMakeFiles/face.dir/flags.make
CMakeFiles/face.dir/face.cpp.o: ../face.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/face.dir/face.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/face.dir/face.cpp.o -c "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/face.cpp"

CMakeFiles/face.dir/face.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/face.dir/face.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/face.cpp" > CMakeFiles/face.dir/face.cpp.i

CMakeFiles/face.dir/face.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/face.dir/face.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/face.cpp" -o CMakeFiles/face.dir/face.cpp.s

# Object files for target face
face_OBJECTS = \
"CMakeFiles/face.dir/face.cpp.o"

# External object files for target face
face_EXTERNAL_OBJECTS =

face: CMakeFiles/face.dir/face.cpp.o
face: CMakeFiles/face.dir/build.make
face: /usr/local/opt/opencv@2/lib/libopencv_videostab.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_ts.a
face: /usr/local/opt/opencv@2/lib/libopencv_superres.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_stitching.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_contrib.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_nonfree.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_ocl.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_gpu.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_photo.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_objdetect.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_legacy.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_video.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_ml.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_calib3d.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_features2d.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_highgui.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_imgproc.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_flann.2.4.13.dylib
face: /usr/local/opt/opencv@2/lib/libopencv_core.2.4.13.dylib
face: CMakeFiles/face.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable face"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/face.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/face.dir/build: face
.PHONY : CMakeFiles/face.dir/build

CMakeFiles/face.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/face.dir/cmake_clean.cmake
.PHONY : CMakeFiles/face.dir/clean

CMakeFiles/face.dir/depend:
	cd "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials" "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials" "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug" "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug" "/Users/lucian.rusu/Documents/UoB_Programs/ImageProcessingCW/Coursework materials/cmake-build-debug/CMakeFiles/face.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/face.dir/depend

