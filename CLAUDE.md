# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Run
- **Build**: `g++ -std=c++11 sampling.cpp test.cpp -o test`
- **Run tests**: `./test` (or use the provided script: `bash run.sh`)
- **Quick build and test**: `bash run.sh` - builds and runs tests in one command

### Platform Notes
- Uses C++11 standard
- Build script uses bash (run.sh)
- Executable output: `test` (Linux/macOS) or `test.exe` (Windows)

## Code Architecture

### Core Implementation (sampling.cpp)
This is a C++ implementation of Repetition-Aware Sampling (RAS) for language model token sampling. The implementation parallels a PyTorch version in sampling.py.

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

### Test Implementation (test.cpp)
Comprehensive test suite that validates:
- Softmax probability distributions
- Nucleus sampling behavior and distributions
- Random sampling distributions  
- RAS repetition detection and fallback logic
- EOS token handling in sampling_ids()

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