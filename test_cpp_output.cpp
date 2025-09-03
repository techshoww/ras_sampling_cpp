#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <random>
#include <sstream>
#include <string>
#define _USE_MATH_DEFINES
#include <cmath>
#include "sampling.hpp"
#include "test_data.hpp"

void run_cpp_tests() {
    std::vector<TestCase> test_cases = get_test_cases();
    std::ofstream outfile("cpp_results.txt");
    
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open cpp_results.txt for writing" << std::endl;
        return;
    }
    
    outfile << "C++ Test Results\n";
    outfile << "================\n\n";
    
    for (const auto& test_case : test_cases) {
        std::cout << "Running C++ test: " << test_case.name << std::endl;
        outfile << "Test Case: " << test_case.name << "\n";
        outfile << "Parameters:\n";
        outfile << "  weighted_scores: [";
        for (size_t i = 0; i < test_case.weighted_scores.size(); ++i) {
            outfile << test_case.weighted_scores[i];
            if (i < test_case.weighted_scores.size() - 1) outfile << ", ";
        }
        outfile << "]\n";
        outfile << "  decoded_tokens: [";
        for (size_t i = 0; i < test_case.decoded_tokens.size(); ++i) {
            outfile << test_case.decoded_tokens[i];
            if (i < test_case.decoded_tokens.size() - 1) outfile << ", ";
        }
        outfile << "]\n";
        outfile << "  speech_token_size: " << test_case.speech_token_size << "\n";
        outfile << "  top_p: " << test_case.top_p << "\n";
        outfile << "  top_k: " << test_case.top_k << "\n";
        outfile << "  win_size: " << test_case.win_size << "\n";
        outfile << "  tau_r: " << test_case.tau_r << "\n";
        outfile << "  ignore_eos: " << (test_case.ignore_eos ? "true" : "false") << "\n";
        
        // Run 1000 samples
        std::vector<int> samples;
        std::vector<int> distribution(test_case.weighted_scores.size(), 0);
        
        for (int i = 0; i < 1000; ++i) {
            try {
                int sample_id = sampling_ids(
                    test_case.weighted_scores,
                    test_case.decoded_tokens,
                    test_case.speech_token_size,
                    test_case.ignore_eos
                );
                samples.push_back(sample_id);
                if (sample_id >= 0 && sample_id < static_cast<int>(distribution.size())) {
                    distribution[sample_id]++;
                }
            } catch (const std::exception& e) {
                outfile << "Error in sample " << i << ": " << e.what() << "\n";
                break;
            }
        }
        
        // Output first 100 samples for verification
        outfile << "First 100 samples: [";
        for (size_t i = 0; i < std::min(static_cast<size_t>(100), samples.size()); ++i) {
            outfile << samples[i];
            if (i < std::min(static_cast<size_t>(100), samples.size()) - 1) outfile << ", ";
        }
        outfile << "]\n";
        
        // Output distribution
        outfile << "Distribution: [";
        for (size_t i = 0; i < distribution.size(); ++i) {
            outfile << distribution[i];
            if (i < distribution.size() - 1) outfile << ", ";
        }
        outfile << "]\n";
        outfile << "Total samples: " << samples.size() << "\n\n";
    }
    
    outfile.close();
    std::cout << "C++ test results saved to cpp_results.txt" << std::endl;
}

int main() {
    // Set seed for C++ random number generator to match Python
    std::srand(42);
    
    run_cpp_tests();
    return 0;
}