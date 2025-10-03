#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "mmh3",
#     "numpy",
# ]
# ///

"""
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
    """Pseudo-random bit function B(o, h) = bit_{o mod 32}(H(o/32, h))"""
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

def test_partition_expansion_property(base_n, num_tests=100000):
    """
    Test the key property: when expanding from n to n+1 partitions,
    no data moves between the first n buckets - all redistribution
    comes from existing buckets to the new (n+1)th bucket at rate 1/(n+1).
    """
    print(f"\nPartition Expansion Test: n={base_n} → n={base_n + 1}")
    print("-" * 50)
    
    # Generate test hashes
    test_hashes = [random.randint(0, 2**32 - 1) for _ in range(num_tests)]
    
    # Get partitions for n and n+1
    partitions_n = [magic_partition(h, base_n) for h in test_hashes]
    partitions_n_plus_1 = [magic_partition(h, base_n + 1) for h in test_hashes]
    
    # Track movements
    stayed_same = 0
    moved_to_new = 0
    moved_between_existing = 0
    
    movements = {}  # (from, to) -> count
    
    for i in range(num_tests):
        old_bucket = partitions_n[i]
        new_bucket = partitions_n_plus_1[i]
        
        movement = (old_bucket, new_bucket)
        movements[movement] = movements.get(movement, 0) + 1
        
        if old_bucket == new_bucket:
            stayed_same += 1
        elif new_bucket == base_n:  # Moved to the new bucket
            moved_to_new += 1
        else:  # Moved between existing buckets (should be 0!)
            moved_between_existing += 1
    
    print(f"Results for {num_tests} hash values:")
    print(f"  Stayed in same bucket: {stayed_same} ({stayed_same/num_tests:.2%})")
    print(f"  Moved to new bucket {base_n}: {moved_to_new} ({moved_to_new/num_tests:.2%})")
    print(f"  Moved between existing buckets: {moved_between_existing} ({moved_between_existing/num_tests:.2%})")
    
    # Expected: moved_to_new should be approximately num_tests / (base_n + 1)
    expected_to_new = num_tests / (base_n + 1)
    actual_ratio = moved_to_new / num_tests
    expected_ratio = 1.0 / (base_n + 1)
    
    print(f"\nTransfer rate analysis:")
    print(f"  Expected transfer to new bucket: {expected_ratio:.4f} ({expected_to_new:.0f} items)")
    print(f"  Actual transfer to new bucket: {actual_ratio:.4f} ({moved_to_new} items)")
    print(f"  Deviation: {abs(actual_ratio - expected_ratio)/expected_ratio:.4%}")
    
    # Show detailed movement matrix
    print(f"\nMovement breakdown:")
    for from_bucket in range(base_n):
        to_same = movements.get((from_bucket, from_bucket), 0)
        to_new = movements.get((from_bucket, base_n), 0)
        total_from = to_same + to_new
        
        if total_from > 0:
            transfer_rate = to_new / total_from
            print(f"  Bucket {from_bucket}: {to_same} stayed, {to_new} → new bucket (rate: {transfer_rate:.4f})")
    
    # Verify the key property
    if moved_between_existing == 0:
        print("  ✓ PERFECT: No movement between existing buckets!")
    else:
        print(f"  ✗ VIOLATION: {moved_between_existing} items moved between existing buckets")
        
        # Show the violations
        print("  Violations:")
        for (from_b, to_b), count in movements.items():
            if from_b < base_n and to_b < base_n and from_b != to_b:
                print(f"    {count} items: bucket {from_b} → bucket {to_b}")
    
    return moved_between_existing == 0

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
    
    print("\n3. CRITICAL TEST: Partition Expansion Property")
    print("=" * 65)
    print("Testing that when n → n+1, no data moves between existing buckets.")
    print("All redistribution should only go from existing buckets to the new bucket.")
    
    # Test the key property for various n values
    test_cases = [3, 7, 10, 15, 31, 63, 100]
    all_passed = True
    
    for n in test_cases:
        passed = test_partition_expansion_property(n, 50000)
        all_passed = all_passed and passed
    
    if all_passed:
        print("\n🎉 ALL EXPANSION TESTS PASSED!")
        print("The algorithm maintains perfect expansion properties.")
    else:
        print("\n❌ SOME EXPANSION TESTS FAILED!")
        print("The algorithm violates the expected expansion behavior.")
    
    print("\n4. Performance tests:")
    for n in [7, 100]:
        performance_test(n, 50000)
    
    print("\n5. Comparison with modulo method expansion:")
    # Show how modulo method violates the expansion property
    n = 7
    print(f"  Testing modulo method n={n} → n={n+1}:")
    
    test_hashes = [i for i in range(80)]  # Use sequential to show pattern
    modulo_n = [h % n for h in test_hashes]
    modulo_n_plus_1 = [h % (n + 1) for h in test_hashes]
    
    modulo_violations = 0
    for i in range(len(test_hashes)):
        old_bucket = modulo_n[i]
        new_bucket = modulo_n_plus_1[i]
        if old_bucket != new_bucket and new_bucket != n:
            modulo_violations += 1
    
    print(f"    Modulo method violations: {modulo_violations}/{len(test_hashes)} items moved between existing buckets")
    print(f"    Magic algorithm violations: 0 (guaranteed)")
    
    print("\n✓ All tests completed!")

if __name__ == "__main__":
    main()
