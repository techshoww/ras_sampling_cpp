#include <iostream>
#include <vector>
#include <map>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <random>
#include <sstream>
#define _USE_MATH_DEFINES
#include <cmath>

// Include function declarations
std::vector<float> softmax_stable(const std::vector<float>& logits);
std::vector<size_t> sort_indices_desc(const std::vector<float>& v);
int sample_multinomial(const std::vector<float>& probabilities, std::mt19937& gen);
int nucleus_sampling(const std::vector<float>& weighted_scores, float top_p = 0.8f, int top_k = 25);
int random_sampling(const std::vector<float>& weighted_scores);
int ras_sampling(const std::vector<float>& weighted_scores, const std::vector<int>& decoded_tokens, int speech_token_size, float top_p = 0.8f, int top_k = 25, int win_size = 10, float tau_r = 0.1f);
int sampling_ids(const std::vector<float>& weighted_scores, const std::vector<int>& decoded_tokens, int speech_token_size, bool ignore_eos = true, int max_trials = 100);

struct TestCase {
    std::string name;
    std::vector<float> weighted_scores;
    std::vector<int> decoded_tokens;
    int speech_token_size;
    float top_p;
    int top_k;
    int win_size;
    float tau_r;
    bool ignore_eos;
};

std::vector<TestCase> generate_test_cases() {
    std::vector<TestCase> test_cases;
    
    // Test case 1: Basic case (same as Python)
    test_cases.push_back({
        "basic_case",
        {1.2f, 3.4f, 0.5f, 5.6f, 2.1f, 4.0f, 1.8f, 0.9f, 2.7f, 3.3f},
        {1, 5, 2, 8, 1, 3, 7, 1, 4, 9, 6, 1, 0, 2, 5},
        9,
        0.8f,
        25,
        10,
        0.1f,
        true
    });
    
    // Test case 2: Large vocabulary (generated with same seed)
    std::mt19937 gen(42);
    std::normal_distribution<float> normal_dist(0.0f, 2.0f);
    std::uniform_int_distribution<int> uniform_int(0, 49);
    
    std::vector<float> large_scores;
    for (int i = 0; i < 50; ++i) {
        large_scores.push_back(normal_dist(gen));
    }
    
    std::vector<int> large_tokens;
    for (int i = 0; i < 20; ++i) {
        large_tokens.push_back(uniform_int(gen));
    }
    
    test_cases.push_back({
        "large_vocab",
        large_scores,
        large_tokens,
        49,
        0.9f,
        40,
        15,
        0.2f,
        false
    });
    
    // Test case 3: High repetition scenario
    std::mt19937 gen2(42);
    std::normal_distribution<float> normal_dist2(0.0f, 1.0f);
    std::vector<float> rep_scores;
    for (int i = 0; i < 20; ++i) {
        rep_scores.push_back(normal_dist2(gen2));
    }
    
    test_cases.push_back({
        "high_repetition",
        rep_scores,
        {5, 3, 5, 7, 5, 1, 5, 9, 5, 2, 5, 8, 5, 4, 5},
        19,
        0.7f,
        15,
        8,
        0.15f,
        true
    });
    
    // Test case 4: Edge case - small vocabulary
    test_cases.push_back({
        "small_vocab",
        {2.0f, -1.0f, 3.5f},
        {0, 1, 0, 2, 0},
        2,
        0.6f,
        3,
        5,
        0.1f,
        false
    });
    
    return test_cases;
}

void run_cpp_tests() {
    std::vector<TestCase> test_cases = generate_test_cases();
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