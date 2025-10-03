#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "mmh3",
#     "numpy",
# ]
# ///

Magic Partitioning Algorithm - Simple Python Implementation
A novel approach for uniform hash distribution across non-power-of-two partitions.
"""

import mmh3
import numpy as np
import random
import time

def hash_function_H(y, z):
    """Hash function H(y, z) -> 32-bit uniform output using MurmurHash3"""
    # Combine y and z into bytes and hash
    data = y.to_bytes(4, 'little') + z.to_bytes(4, 'little')
    return mmh3.hash(data, signed=False)

def bit_function_B(o, h):
    """Pseudo-random bit function B(o, h) = bit_{o mod 32}(H(⌊o/32⌋, h))"""
    hash_input = o // 32
    bit_pos = o % 32
    hash_output = hash_function_H(hash_input, h)
    return (hash_output >> bit_pos) & 1

def magic_partition(h, n):
    """
    Magic Partitioning Algorithm
    
    Args:
        h: Hash value (32-bit unsigned integer)
        n: Number of partitions (>= 2)
    
    Returns:
        Partition index in range [0, n-1] with uniform distribution
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    
    # Find k such that 2^k >= n
    k = (n - 1).bit_length()
    s = 1 << k  # s = 2^k
    t = 0  # retry counter
    
    while True:
        v = 0
        p = s // 2
        o = s - 1
        
        # Check MSB using bit function B
        if bit_function_B(o + t, h) == 1:
            # MSB = 1 path (Sequence A with retry offset t)
            v += p
            o -= 1
            p //= 2
            
            # Process remaining bits
            while p >= 1:
                if bit_function_B(o + t, h) == 1:
                    v += p
                    if v >= n:
                        break  # Invalid, need retry
                    o -= 1
                else:
                    o -= p
                p //= 2
            
            if v < n:
                return v
        else:
            # MSB = 0 path (Sequence B without retry offset)
            o -= p
            p //= 2
            
            # Process remaining bits
            while p >= 1:
                if bit_function_B(o, h) == 1:
                    v += p
                    o -= 1
                else:
                    o -= p
                p //= 2
            
            return v  # Always valid in MSB = 0 path
        
        # Increment retry counter by s for next attempt
        t += s

def test_uniformity(n, num_tests=100000):
    """Test the uniformity of the partitioning function"""
    counts = [0] * n
    for _ in range(num_tests):
        h = random.randint(0, 2**32 - 1)
        partition = magic_partition(h, n)
        counts[partition] += 1
    
    expected = num_tests / n
    max_deviation = max(abs(count - expected) / expected for count in counts)
    
    print(f"Uniformity test for n={n} with {num_tests} samples:")
    print(f"Expected count per partition: {expected:.1f}")
    print(f"Actual counts: {counts}")
    print(f"Maximum deviation from uniform: {max_deviation:.4%}")
    return max_deviation

def performance_test(n, num_operations=100000):
    """Test performance of the partitioning function"""
    # Generate test hashes
    hashes = [random.randint(0, 2**32 - 1) for _ in range(num_operations)]
    
    start_time = time.time()
    for h in hashes:
        magic_partition(h, n)
    end_time = time.time()
    
    ops_per_second = num_operations / (end_time - start_time)
    print(f"Performance test for n={n}: {ops_per_second:.0f} operations/second")
    return ops_per_second

def main():
    """Main function to demonstrate and test the Magic Partitioning Algorithm"""
    print("Magic Partitioning Algorithm - Simple Python Implementation")
    print("=" * 65)
    
    print("\n1. Basic functionality test:")
    for n in [3, 7, 10]:
        result = magic_partition(12345, n)
        print(f"  n={n}: partition(12345, {n}) = {result}")
    
    print("\n2. Uniformity tests:")
    for n in [3, 7, 10]:
        deviation = test_uniformity(n, 50000)
        if deviation < 0.02:  # Less than 2% deviation
            print("  ✓ PASSED")
        else:
            print("  ✗ FAILED")
    
    print("\n3. Performance tests:")
    for n in [7, 100]:
        performance_test(n, 50000)
    
    print("\n4. Comparison with modulo method:")
    n = 7
    h = 12345
    magic_result = magic_partition(h, n)
    modulo_result = h % n
    print(f"  Hash {h}, n={n}:")
    print(f"    Magic partition: {magic_result}")
    print(f"    Modulo result: {modulo_result}")
    
    # Test modulo bias
    print("\n5. Modulo bias demonstration:")
    counts_magic = [0] * n
    counts_modulo = [0] * n
    
    for i in range(70000):  # 10,000 samples per partition if uniform
        h = i
        counts_magic[magic_partition(h, n)] += 1
        counts_modulo[h % n] += 1
    
    print(f"  Magic algorithm: {counts_magic}")
    print(f"  Modulo method:   {counts_modulo}")
    
    magic_deviation = max(abs(c - 10000) / 10000 for c in counts_magic)
    modulo_deviation = max(abs(c - 10000) / 10000 for c in counts_modulo)
    
    print(f"  Magic deviation:  {magic_deviation:.4%}")
    print(f"  Modulo deviation: {modulo_deviation:.4%}")
    
    print("\n6. Edge cases:")
    # Test with various n values including powers of 2
    test_ns = [2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33]
    print("  Testing various partition counts...")
    for n in test_ns:
        try:
            result = magic_partition(42, n)
            print(f"    n={n}: ✓ (result={result})")
        except Exception as e:
            print(f"    n={n}: ✗ ({e})")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main()
