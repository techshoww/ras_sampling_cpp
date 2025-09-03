import json
import numpy as np
from scipy import stats

def parse_cpp_results(filename):
    """Parse C++ results from text file"""
    results = []
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by test cases
    test_sections = content.split('Test Case: ')[1:]  # Skip header
    
    for section in test_sections:
        lines = section.strip().split('\n')
        test_name = lines[0]
        
        # Parse parameters and results
        result = {"test_case": test_name}
        
        for line in lines:
            if line.startswith('First 100 samples:'):
                # Extract samples array
                start = line.find('[') + 1
                end = line.find(']')
                samples_str = line[start:end]
                if samples_str.strip():
                    result['samples'] = [int(x.strip()) for x in samples_str.split(',')]
                else:
                    result['samples'] = []
            
            elif line.startswith('Distribution:'):
                # Extract distribution array
                start = line.find('[') + 1
                end = line.find(']')
                dist_str = line[start:end]
                if dist_str.strip():
                    result['distribution'] = [int(x.strip()) for x in dist_str.split(',')]
                else:
                    result['distribution'] = []
            
            elif line.startswith('Total samples:'):
                result['total_samples'] = int(line.split(':')[1].strip())
        
        results.append(result)
    
    return results

def compare_distributions(py_dist, cpp_dist, test_name):
    """Compare two distributions using statistical tests"""
    
    if len(py_dist) != len(cpp_dist):
        print(f"❌ {test_name}: Distribution lengths don't match ({len(py_dist)} vs {len(cpp_dist)})")
        return False
    
    # Convert to numpy arrays
    py_dist = np.array(py_dist)
    cpp_dist = np.array(cpp_dist)
    
    # Chi-square test for distribution similarity
    # Only test tokens that were sampled in either implementation
    mask = (py_dist > 0) | (cpp_dist > 0)
    if not np.any(mask):
        print(f"⚠️  {test_name}: No samples in either distribution")
        return True
    
    py_masked = py_dist[mask]
    cpp_masked = cpp_dist[mask]
    
    # Add small constant to avoid zero counts
    py_masked = py_masked + 1
    cpp_masked = cpp_masked + 1
    
    try:
        # Chi-square test
        statistic, p_value = stats.chisquare(cpp_masked, py_masked)
        
        # Compute relative differences for major tokens
        total_py = np.sum(py_dist)
        total_cpp = np.sum(cpp_dist)
        
        if total_py > 0 and total_cpp > 0:
            py_probs = py_dist / total_py
            cpp_probs = cpp_dist / total_cpp
            
            # Check if major tokens (>5% probability) are reasonably close
            major_tokens = (py_probs > 0.05) | (cpp_probs > 0.05)
            if np.any(major_tokens):
                max_diff = np.max(np.abs(py_probs[major_tokens] - cpp_probs[major_tokens]))
                
                if max_diff < 0.1:  # 10% tolerance for major tokens
                    print(f"✅ {test_name}: Distributions are similar (max diff: {max_diff:.3f}, p-value: {p_value:.3f})")
                    return True
                else:
                    print(f"⚠️  {test_name}: Large difference in major tokens (max diff: {max_diff:.3f})")
                    return False
            else:
                print(f"✅ {test_name}: No major tokens to compare")
                return True
        else:
            print(f"❌ {test_name}: Empty distributions")
            return False
    
    except Exception as e:
        print(f"❌ {test_name}: Statistical test failed: {e}")
        return False

def compare_samples(py_samples, cpp_samples, test_name):
    """Compare first few samples for exact matching (to check determinism)"""
    
    min_len = min(len(py_samples), len(cpp_samples), 10)  # Compare first 10 samples
    
    if min_len == 0:
        print(f"⚠️  {test_name}: No samples to compare")
        return True
    
    matches = sum(1 for i in range(min_len) if py_samples[i] == cpp_samples[i])
    match_rate = matches / min_len
    
    # For random sampling, we don't expect exact matches, but distributions should be similar
    # Just report the info
    print(f"ℹ️  {test_name}: Sample match rate in first {min_len}: {matches}/{min_len} ({match_rate:.1%})")
    
    return True  # Always pass sample comparison as we focus on distribution matching

def main():
    """Main comparison function"""
    
    print("Comparing Python and C++ Results")
    print("=================================\n")
    
    # Load Python results
    try:
        with open('python_results.json', 'r') as f:
            py_results = json.load(f)
        print("✅ Loaded Python results")
    except FileNotFoundError:
        print("❌ python_results.json not found. Run generate_test_data.py first.")
        return
    except Exception as e:
        print(f"❌ Error loading Python results: {e}")
        return
    
    # Load C++ results
    try:
        cpp_results = parse_cpp_results('cpp_results.txt')
        print("✅ Loaded C++ results")
    except FileNotFoundError:
        print("❌ cpp_results.txt not found. Compile and run the C++ test first.")
        return
    except Exception as e:
        print(f"❌ Error loading C++ results: {e}")
        return
    
    print(f"\nComparing {len(py_results)} test cases...\n")
    
    # Compare each test case
    all_passed = True
    
    for py_result in py_results:
        test_name = py_result['test_case']
        
        # Find corresponding C++ result
        cpp_result = next((r for r in cpp_results if r['test_case'] == test_name), None)
        
        if cpp_result is None:
            print(f"❌ {test_name}: No corresponding C++ result found")
            all_passed = False
            continue
        
        print(f"\n--- {test_name} ---")
        
        # Compare samples (informational)
        sample_ok = compare_samples(
            py_result.get('samples', []),
            cpp_result.get('samples', []),
            test_name
        )
        
        # Compare distributions (main test)
        dist_ok = compare_distributions(
            py_result.get('distribution', []),
            cpp_result.get('distribution', []),
            test_name
        )
        
        if not dist_ok:
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests passed! C++ implementation matches Python behavior.")
    else:
        print("❌ Some tests failed. Check the differences above.")
    
    return all_passed

if __name__ == "__main__":
    main()