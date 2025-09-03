#include <iostream>
#include <vector>
#include <map>
#include <iomanip>
#include <cassert>
#define _USE_MATH_DEFINES
#include <cmath>

// Include the implementation code or header file if separated
// For this example, paste the code from sampling.cpp here, or include a header

int main() {
    // --- Test Data ---
    // Example logits, might represent unnormalized log probabilities
    std::vector<float> weighted_scores = {
        1.2f, 3.4f, 0.5f, 5.6f, 2.1f,
        4.0f, 1.8f, 0.9f, 2.7f, 3.3f
    };

    std::vector<int> decoded_tokens = {
        1, 5, 2, 8, 1, // Repetition of '1'
        3, 7, 1, 4, 9, // Another '1' in window
        6, 1, 0, 2, 5  // Final '1' makes it 4 in the last 10 tokens
    };
    int speech_token_size = 9; // Assume token 9 is the EOS token
    int win_size = 10;
    float tau_r = 0.1f; // 10% of window size is 1. 4 reps should trigger random fallback.

    std::cout << "Testing C++ Sampling Implementation\n";
    std::cout << "====================================\n";

    // --- Test Softmax ---
    std::cout << "\n1. Testing Softmax:\n";
    auto softmax_probs = softmax_stable(weighted_scores);
    std::cout << "Input logits: ";
    for (const auto& val : weighted_scores) std::cout << val << " ";
    std::cout << "\nSoftmax probs: ";
    for (const auto& val : softmax_probs) std::cout << std::fixed << std::setprecision(4) << val << " ";
    std::cout << "\nSum of probs: " << std::accumulate(softmax_probs.begin(), softmax_probs.end(), 0.0f) << "\n";

    // --- Test Nucleus Sampling ---
    std::cout << "\n2. Testing Nucleus Sampling (Top-p=0.8, Top-k=25):\n";
    std::map<int, int> nucleus_counts;
    int n_samples = 10000;
    for (int i = 0; i < n_samples; ++i) {
         int idx = nucleus_sampling(weighted_scores, 0.8f, 25);
         nucleus_counts[idx]++;
    }
    std::cout << "Sampled indices distribution over " << n_samples << " trials:\n";
    for (const auto& pair : nucleus_counts) {
        std::cout << "  Index " << pair.first << ": " << pair.second
                  << " (" << (100.0 * pair.second / n_samples) << "%)\n";
    }

    // --- Test Random Sampling ---
    std::cout << "\n3. Testing Random Sampling:\n";
    std::map<int, int> random_counts;
    for (int i = 0; i < n_samples; ++i) {
         int idx = random_sampling(weighted_scores);
         random_counts[idx]++;
    }
    std::cout << "Sampled indices distribution over " << n_samples << " trials:\n";
    for (const auto& pair : random_counts) {
        std::cout << "  Index " << pair.first << ": " << pair.second
                  << " (" << (100.0 * pair.second / n_samples) << "%)\n";
    }

    // --- Test RAS Sampling (Repetition Triggered) ---
    std::cout << "\n4. Testing RAS Sampling (Repetition Expected to Trigger Random):\n";
    std::cout << "Decoded tokens (last " << win_size << "): ";
    int window_start = std::max(0, static_cast<int>(decoded_tokens.size()) - win_size);
    for (size_t i = window_start; i < decoded_tokens.size(); ++i) {
        std::cout << decoded_tokens[i] << " ";
    }
    std::cout << "\n";
    std::cout << "Repetition count of token 1 in window: 4\n";
    std::cout << "Threshold (win_size * tau_r): " << win_size * tau_r << "\n";
    std::cout << "Repetition count >= threshold, should fallback to random.\n";

    std::map<int, int> ras_counts;
    for (int i = 0; i < n_samples; ++i) {
         int idx = ras_sampling(weighted_scores, decoded_tokens, speech_token_size, 0.8f, 25, win_size, tau_r);
         ras_counts[idx]++;
    }
    std::cout << "RAS sampled indices distribution over " << n_samples << " trials:\n";
    for (const auto& pair : ras_counts) {
        std::cout << "  Index " << pair.first << ": " << pair.second
                  << " (" << (100.0 * pair.second / n_samples) << "%)\n";
    }
    // Note: The distribution should resemble the random sampling one due to fallback.

    // --- Test sampling_ids with ignore_eos ---
    std::cout << "\n5. Testing sampling_ids (ignore_eos=true):\n";
    try {
        // Modify scores to make EOS (index 9) very likely to test ignore_eos logic
        std::vector<float> eos_scores(weighted_scores.size(), -10.0f);
        if (speech_token_size >= 0 && speech_token_size < static_cast<int>(eos_scores.size())) {
             eos_scores[speech_token_size] = 100.0f;
        }
        int sampled_id = sampling_ids(eos_scores, decoded_tokens, speech_token_size, true, 100);
        std::cout << "Sampled ID (with high EOS prob, ignore_eos=true): " << sampled_id << "\n";
        // This should eventually sample something other than 9 (if other probs allow)
        // or hit max_trials if *only* 9 has significant probability.
        // For this test case, it will likely hit max_trials as 9 is dominant.
    } catch (const std::runtime_error& e) {
        std::cout << "Caught expected exception (max trials): " << e.what() << "\n";
    }

    std::cout << "\n6. Testing sampling_ids (ignore_eos=false):\n";
    try {
        // Use original scores where EOS is not dominant
        int sampled_id = sampling_ids(weighted_scores, decoded_tokens, speech_token_size, false, 100);
        std::cout << "Sampled ID (ignore_eos=false): " << sampled_id << "\n";
        // This can be any valid index, including 9.
    } catch (const std::runtime_error& e) {
        std::cout << "Caught exception: " << e.what() << "\n";
    }


    std::cout << "\nAll tests completed.\n";
    return 0;
}