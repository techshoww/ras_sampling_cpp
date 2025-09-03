#!/bin/bash

echo "Starting Python vs C++ comparison tests..."

# Step 1: Generate Python test data and results
echo "Step 1: Running Python tests..."
python generate_test_data.py

if [ $? -ne 0 ]; then
    echo "❌ Python test failed"
    exit 1
fi

# Step 2: Compile and run C++ tests
echo "Step 2: Compiling and running C++ tests..."
g++ -std=c++11 -O2 sampling.cpp test_cpp_output.cpp -o test_cpp_output

if [ $? -ne 0 ]; then
    echo "❌ C++ compilation failed"
    exit 1
fi

./test_cpp_output

if [ $? -ne 0 ]; then
    echo "❌ C++ test execution failed"
    exit 1
fi

# Step 3: Compare results
echo "Step 3: Comparing results..."
python compare_results.py

echo "Test completed!"