import numpy as np
import torch
import json
import random

# Import the sampling functions
from sampling import nucleus_sampling, random_sampling, ras_sampling, sampling_ids

def generate_test_cases():
    """Generate random test data for comparing Python and C++ implementations"""
    
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    test_cases = []
    
    # Test case 1: Basic case
    weighted_scores = torch.tensor([1.2, 3.4, 0.5, 5.6, 2.1, 4.0, 1.8, 0.9, 2.7, 3.3])
    decoded_tokens = [1, 5, 2, 8, 1, 3, 7, 1, 4, 9, 6, 1, 0, 2, 5]
    speech_token_size = 9
    
    test_cases.append({
        "name": "basic_case",
        "weighted_scores": weighted_scores.tolist(),
        "decoded_tokens": decoded_tokens,
        "speech_token_size": speech_token_size,
        "top_p": 0.8,
        "top_k": 25,
        "win_size": 10,
        "tau_r": 0.1,
        "ignore_eos": True
    })
    
    # Test case 2: Large vocabulary
    weighted_scores = torch.randn(50) * 2.0
    decoded_tokens = [random.randint(0, 49) for _ in range(20)]
    speech_token_size = 49
    
    test_cases.append({
        "name": "large_vocab",
        "weighted_scores": weighted_scores.tolist(),
        "decoded_tokens": decoded_tokens,
        "speech_token_size": speech_token_size,
        "top_p": 0.9,
        "top_k": 40,
        "win_size": 15,
        "tau_r": 0.2,
        "ignore_eos": False
    })
    
    # Test case 3: High repetition scenario
    weighted_scores = torch.randn(20)
    # Create high repetition of token 5
    decoded_tokens = [5, 3, 5, 7, 5, 1, 5, 9, 5, 2, 5, 8, 5, 4, 5]
    speech_token_size = 19
    
    test_cases.append({
        "name": "high_repetition",
        "weighted_scores": weighted_scores.tolist(),
        "decoded_tokens": decoded_tokens,
        "speech_token_size": speech_token_size,
        "top_p": 0.7,
        "top_k": 15,
        "win_size": 8,
        "tau_r": 0.15,
        "ignore_eos": True
    })
    
    # Test case 4: Edge case - small vocabulary
    weighted_scores = torch.tensor([2.0, -1.0, 3.5])
    decoded_tokens = [0, 1, 0, 2, 0]
    speech_token_size = 2
    
    test_cases.append({
        "name": "small_vocab",
        "weighted_scores": weighted_scores.tolist(),
        "decoded_tokens": decoded_tokens,
        "speech_token_size": speech_token_size,
        "top_p": 0.6,
        "top_k": 3,
        "win_size": 5,
        "tau_r": 0.1,
        "ignore_eos": False
    })
    
    return test_cases

def run_python_tests():
    """Run tests with Python implementation and save results"""
    
    test_cases = generate_test_cases()
    results = []
    
    for case in test_cases:
        print(f"Running Python test: {case['name']}")
        
        weighted_scores = torch.tensor(case['weighted_scores'])
        decoded_tokens = case['decoded_tokens']
        speech_token_size = case['speech_token_size']
        
        # Run multiple samples to get distribution
        samples = []
        for i in range(1000):
            try:
                sample_id = sampling_ids(
                    weighted_scores=weighted_scores,
                    decoded_tokens=decoded_tokens,
                    speech_token_size=speech_token_size,
                    sampling=0,  # Not used in current implementation
                    ignore_eos=case['ignore_eos']
                )
                # Handle both tensor and scalar returns
                if hasattr(sample_id, 'item'):
                    samples.append(int(sample_id.item()))
                elif hasattr(sample_id, '__iter__'):
                    samples.append(int(sample_id[0]))
                else:
                    samples.append(int(sample_id))
            except Exception as e:
                print(f"Error in sample {i}: {e}")
                break
        
        # Calculate distribution
        vocab_size = len(case['weighted_scores'])
        distribution = [0] * vocab_size
        for sample in samples:
            if 0 <= sample < vocab_size:
                distribution[sample] += 1
        
        result = {
            "test_case": case['name'],
            "parameters": case,
            "samples": samples[:100],  # Save first 100 samples for verification
            "distribution": distribution,
            "total_samples": len(samples)
        }
        results.append(result)
    
    # Save results to file
    with open('python_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Also generate a C++ header file with test data for consistency
    generate_cpp_test_data(test_cases)
    
    print("Python test results saved to python_results.json")
    return results

def generate_cpp_test_data(test_cases):
    """Generate a C++ header file with the exact same test data as Python"""
    with open('test_data.hpp', 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <vector>\n")
        f.write("#include <string>\n\n")
        f.write("struct TestCase {\n")
        f.write("    std::string name;\n")
        f.write("    std::vector<float> weighted_scores;\n")
        f.write("    std::vector<int> decoded_tokens;\n")
        f.write("    int speech_token_size;\n")
        f.write("    float top_p;\n")
        f.write("    int top_k;\n")
        f.write("    int win_size;\n")
        f.write("    float tau_r;\n")
        f.write("    bool ignore_eos;\n")
        f.write("};\n\n")
        f.write("std::vector<TestCase> get_test_cases() {\n")
        f.write("    return {\n")
        
        for i, case in enumerate(test_cases):
            f.write("        {\n")
            f.write(f'            "{case["name"]}",\n')
            f.write("            {")
            f.write(", ".join([f"{x}f" for x in case["weighted_scores"]]))
            f.write("},\n")
            f.write("            {")
            f.write(", ".join([str(x) for x in case["decoded_tokens"]]))
            f.write("},\n")
            f.write(f'            {case["speech_token_size"]},\n')
            f.write(f'            {case["top_p"]}f,\n')
            f.write(f'            {case["top_k"]},\n')
            f.write(f'            {case["win_size"]},\n')
            f.write(f'            {case["tau_r"]}f,\n')
            f.write(f'            {str(case["ignore_eos"]).lower()}\n')
            f.write("        }")
            if i < len(test_cases) - 1:
                f.write(",")
            f.write("\n")
        
        f.write("    };\n")
        f.write("}\n")
    
    print("C++ test data header saved to test_data.hpp")

if __name__ == "__main__":
    run_python_tests()