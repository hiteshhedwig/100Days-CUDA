#!/bin/bash

# Script to compile and run a specific day's CUDA code

# Display usage if no argument provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <day_number>"
    echo "Example: $0 1"
    exit 1
fi

# Get day number and format it with leading zeros
day_num=$1
day_formatted=$(printf "%03d" $day_num)
day_dir="day$day_formatted"

# Check if the day directory exists
if [ ! -d "days/$day_dir" ]; then
    echo "Error: Directory days/$day_dir does not exist!"
    exit 1
fi

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    mkdir build
    echo "Created build directory"
fi

# Ensure CMakeLists.txt includes the target day
# This is a simple check - it doesn't modify the file if the day isn't included
if ! grep -q "add_subdirectory(days/$day_dir)" CMakeLists.txt; then
    echo "Warning: days/$day_dir not found in CMakeLists.txt"
    echo "You may need to uncomment or add: add_subdirectory(days/$day_dir)"
fi

# Navigate to build directory, run cmake and make
echo "Building day $day_formatted..."
cd build
cmake ..

# Find the target name from the generated Makefile or build system
echo "Finding and building targets for day $day_formatted..."
# Look through the Makefile to find targets that match our day
targets=$(make help | grep -o "day${day_formatted}_[^ ]*" || true)

if [ -z "$targets" ]; then
    echo "No specific targets found for day $day_formatted, building all targets"
    make
else
    # Build each target that matches the day
    for target in $targets; do
        echo "Building target: $target"
        make $target
    done
fi

# Find the executable
executable=$(find ./days/$day_dir -type f -executable -name "day${day_formatted}_*")

if [ -z "$executable" ]; then
    echo "Error: Executable for day $day_formatted not found!"
    echo "Build may have failed or naming convention is different."
    exit 1
fi

# Run the executable
echo "Running $executable..."
$executable

echo "Completed execution of day $day_formatted"