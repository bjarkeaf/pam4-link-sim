#%% Utility functions for LFSR and PRBS generation

def lfsr_sequence(register_size, taps, num_bits, seed = None):
    """Generates a sequence of bits using a Linear Feedback Shift Register (LFSR).

    Args:
        register_size (int): The size of the LFSR register.
        taps (list of int): The positions of the taps in the LFSR (0-indexed from the most significant bit).
        seed (int): The initial state of the LFSR.
        num_bits (int): The number of bits to generate.

    Returns:
        list of int: A list containing the generated bits.
    """
    state = seed if seed is not None else 1  # Default seed value if not provided

    bits = []
    for _ in range(num_bits):
        # Calculate the feedback/output bit
        feedback_bit = 0
        for tap in taps:
            lsb_tap = register_size - 1 - tap  # Convert tap position to match the bit indexing
            feedback_bit ^= (state >> lsb_tap) & 1  # XOR the tapped bits

        # Shift the register to the right and insert the feedback bit at the leftmost position (MSB)
        state = (state >> 1) | (feedback_bit << (register_size - 1))
        bits.append(feedback_bit) # Append the output bit (feedback bit) to the sequence

    return bits

def prbs_sequence(register_size, seed, num_bits=None, taps=None):
    """Generates a Pseudo-Random Binary Sequence (PRBS) using an LFSR.

    Args:
        register_size (int): The size of the LFSR register.
        seed (int): The initial state of the LFSR.
        num_bits (int): The number of bits to generate.
        taps (list of int, optional): Custom tap positions. If not provided, uses standard PRBS taps
                                      for supported register sizes (7, 9, 11, 13, 15, 20, 23, 31).
                                      If register size not in supported list and taps not provided,
                                      defaults to empty list (no feedback).
    Returns:
        list of int: A list containing the generated PRBS bits.
    """
    # Tap positions are 0-indexed from the MSB, so tap 0 corresponds to tapping between the two leftmost bits.
    # Tap 0 corresponds to the x term in the polynomial, and the highest tap corresponds to the x^n term where n is the register size.
    standard_taps = {
        7:  [5, 6],            # x^7 + x^6 + 1
        9:  [4, 8],            # x^9 + x^5 + 1
        11: [8, 10],           # x^11 + x^9 + 1
        13: [0, 1, 11, 12],    # x^13 + x^12 + x^2 + x + 1
        15: [13, 14],          # x^15 + x^14 + 1
        20: [2, 19],           # x^20 + x^3 + 1
        23: [17, 22],          # x^23 + x^18 + 1
        31: [27, 30],          # x^31 + x^28 + 1
    }

    # Use provided num_bits or default to a full cycle length for the given register size (2^n - 1)
    if num_bits is None:
        num_bits = (1 << register_size) - 1  # 2^n - 1

    # Use provided taps or default to standard PRBS if available
    if taps is None:
        taps = standard_taps.get(register_size, [])
    
    return lfsr_sequence(register_size=register_size, taps=taps, seed=seed, num_bits=num_bits)

def gray_code_pairs(bit_sequence):
    """Pairs consecutive bits and Gray-codes them into integers (0-3).
    Handles odd-length input naturally by concatenating two repetitions,
    per IEEE 802.3 120.5.11.2.1 (the pairing phase shift across repetitions
    falls out automatically from pairing the doubled sequence)."""
    sequence_length = len(bit_sequence)
    if sequence_length % 2 != 0:
        bit_sequence = bit_sequence + bit_sequence[:1] # Duplicate first bit to make length even
    gray_to_int = [0, 1, 3, 2]  # {0,0}->0 {0,1}->1 {1,0}->3 {1,1}->2
    return [gray_to_int[(bit_sequence[i] << 1) | bit_sequence[i+1]] for i in range(0, sequence_length, 2)]

def precode(data_sequence, block_size=46):
    # Not needed for test patterns, only for training pattern, but included for completeness.
    """Precodes a sequence of integer symbols (0-3) per IEEE 802.3 94.2.2.6, by subtracting 
    the previous precoded symbol from the current data symbol modulo 4. First symbol of each
    block of size block_size is passed through unchanged."""
    if not data_sequence:
        return []
    precoded_sequence = []
    for i in range(len(data_sequence)):
        if i % block_size == 0:
            precoded_bit = data_sequence[i]  # termination symbol, passes through
        else:
            precoded_bit = (data_sequence[i] - precoded_sequence[i - 1]) % 4
        precoded_sequence.append(precoded_bit)
    return precoded_sequence

def int_to_pam4(integer_sequence):
    """Converts a sequence of integers (0, 1, 2, 3) to their corresponding PAM4 symbols (-1, -1/3, 1/3, 1)."""
    int_to_pam4 = [-1, -1/3, 1/3, 1]
    return [int_to_pam4[i] for i in integer_sequence]


#%% Function testing

# Generating test patterns for lanes 0-3
seeds = [0b0000010101011, 0b0011101000001, 0b1001000101100, 0b0100010000010]  # Different seeds for each lane
test_patterns = []
for seed in seeds:
    prbs_bits = prbs_sequence(register_size=13, seed=seed)  # Generate full PRBS-13 sequence (8191 bits)
    prbs_bits += prbs_bits  # Duplicate the sequence to ensure even length for pairing
    gray_pairs = gray_code_pairs(prbs_bits)  # Pair and Gray-code the bits
    pam4_symbols = int_to_pam4(gray_pairs)  # Convert to PAM4 symbols
    test_patterns.append(pam4_symbols)

print("Generated PAM4 test patterns for lanes 0-3 (first 20 symbols of each lane):")
for lane, pattern in enumerate(test_patterns):
    print(f"Lane {lane}: {pattern[:20]}")


# %%
