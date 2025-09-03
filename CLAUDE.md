# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Run
- **Build standalone tests**: `g++ -std=c++11 sampling.hpp test.cpp -o test`
- **Run standalone tests**: `./test` (or use the provided script: `bash run.sh`)
- **Quick build and test**: `bash run.sh` - builds and runs tests in one command

### Cross-Language Validation
- **Full test suite**: `bash run_tests.sh` - runs Python vs C++ comparison tests
- **Generate test data**: `python generate_test_data.py` - creates test cases and Python baseline results
- **Build comparison tests**: `g++ -std=c++11 -O2 test_cpp_output.cpp -o test_cpp_output`
- **Compare results**: `python compare_results.py` - validates C++ implementation against Python baseline

### Platform Notes
- Uses C++11 standard with `-O2` optimization for comparison tests
- Build scripts use bash (run.sh, run_tests.sh)
- Executable outputs: `test` (standalone), `test_cpp_output` (comparison tests)

## Code Architecture

### Core Implementation (sampling.hpp)
This is a header-only C++ implementation of Repetition-Aware Sampling (RAS) for language model token sampling. The implementation parallels a PyTorch version in sampling.py and provides a C++ port with identical functionality.

**Key Functions:**
- `softmax_stable()` - Numerically stable softmax implementation
- `nucleus_sampling()` - Top-p/Top-k sampling implementation  
- `random_sampling()` - Multinomial sampling from softmax distribution
- `ras_sampling()` - Main RAS algorithm that combines nucleus sampling with repetition detection
- `sampling_ids()` - High-level sampling interface with EOS token handling

**Algorithm Flow:**
1. RAS first performs nucleus sampling (top-p + top-k filtering)
2. Checks if the selected token appears too frequently in recent window (win_size tokens)
3. If repetition threshold (tau_r) is exceeded, falls back to random sampling
4. Handles EOS token filtering when ignore_eos=true

### Test Implementations
**Standalone Tests (test.cpp)**: Comprehensive unit tests that validate individual function behavior including softmax distributions, nucleus sampling, random sampling, RAS repetition detection, and EOS token handling.

**Cross-Language Validation Framework**: 
- **generate_test_data.py**: Creates randomized test cases and generates Python baseline results using PyTorch implementation
- **test_cpp_output.cpp**: Runs identical test cases through C++ implementation and outputs results in structured format
- **compare_results.py**: Statistical comparison of Python vs C++ outputs using statistical tests to validate implementation correctness

### Key Parameters
- `top_p`: Nucleus sampling cumulative probability threshold (default: 0.8)
- `top_k`: Maximum number of tokens to consider (default: 25)
- `win_size`: Sliding window size for repetition detection (default: 10)
- `tau_r`: Repetition threshold as fraction of window (default: 0.1)
- `speech_token_size`: EOS token ID for sequence termination

### Design Notes
- Uses static random number generators for efficiency
- Includes robust error handling for edge cases (empty inputs, max trials)
- Implements numerically stable computations (softmax with max subtraction)
- Supports both header-only and source file organization patterns