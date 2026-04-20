"""
Verhoeff Algorithm Implementation for Aadhaar Validation
Reference: https://en.wikipedia.org/wiki/Verhoeff_algorithm
"""

# Verhoeff multiplication table (d)
D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
    [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
    [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
    [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
    [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
]

# Verhoeff permutation table (p)
P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [1, 7, 2, 5, 3, 8, 9, 0, 4, 6],
    [2, 4, 1, 9, 7, 6, 3, 8, 5, 0],
    [3, 8, 5, 0, 1, 2, 4, 9, 6, 7],
    [4, 3, 9, 2, 6, 0, 7, 1, 5, 8],
    [5, 0, 6, 8, 4, 1, 3, 7, 2, 9],
    [6, 9, 3, 1, 8, 5, 2, 4, 0, 7],
    [7, 2, 8, 6, 9, 4, 0, 5, 1, 3],
    [8, 5, 4, 3, 0, 9, 1, 6, 7, 2],
    [9, 6, 7, 4, 2, 8, 5, 3, 0, 1]
]


def verhoeff_check(number: str) -> bool:
    """
    Validate a number using Verhoeff algorithm.
    Returns True if the checksum is valid, False otherwise.
    
    Args:
        number: String representation of the number (should be 12 digits for Aadhaar)
    
    Returns:
        bool: True if valid, False otherwise
    
    Examples:
        >>> verhoeff_check("275492384017")  # Valid Aadhaar example
        True
        >>> verhoeff_check("123456789012")  # Invalid
        False
    """
    if not number or not number.isdigit():
        return False
    
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = D[c][P[(i % 8)][int(digit)]]
    
    return c == 0


def generate_verhoeff_digit(number: str) -> str:
    """
    Generate the Verhoeff check digit for a given number.
    
    Args:
        number: String of digits (without check digit)
    
    Returns:
        str: Single digit check digit
    
    Example:
        For Aadhaar, first 11 digits -> generates 12th check digit
    """
    number = number + '0'  # Append 0 temporarily
    c = 0
    for i, digit in enumerate(reversed(number)):
        c = D[c][P[(i % 8)][int(digit)]]
    
    # Find the digit that makes the checksum 0
    for i in range(10):
        test_c = c
        test_c = D[test_c][P[(len(number) % 8)][i]]
        if test_c == 0:
            return str(i)
    
    return '0'


def validate_aadhaar(aadhaar_number: str) -> dict:
    """
    Comprehensive Aadhaar validation.
    
    Args:
        aadhaar_number: 12-digit Aadhaar number
    
    Returns:
        dict with validation results:
        {
            'valid': bool,
            'format_valid': bool,
            'length_valid': bool,
            'verhoeff_valid': bool,
            'message': str
        }
    """
    result = {
        'valid': False,
        'format_valid': False,
        'length_valid': False,
        'verhoeff_valid': False,
        'message': ''
    }
    
    # Clean input
    cleaned = aadhaar_number.strip().replace(' ', '').replace('-', '')
    
    # Check if all digits
    if not cleaned.isdigit():
        result['message'] = 'Aadhaar must contain only digits'
        return result
    
    result['format_valid'] = True
    
    # Check length
    if len(cleaned) != 12:
        result['length_valid'] = False
        result['message'] = f'Aadhaar must be 12 digits, got {len(cleaned)}'
        return result
    
    result['length_valid'] = True
    
    # Verhoeff validation
    if verhoeff_check(cleaned):
        result['verhoeff_valid'] = True
        result['valid'] = True
        result['message'] = 'Valid Aadhaar number'
    else:
        result['verhoeff_valid'] = False
        result['message'] = 'Verhoeff checksum validation failed'
    
    return result


def extract_aadhaar_from_text(text: str) -> str:
    """
    Extract potential Aadhaar number from text.
    
    Args:
        text: Input text that may contain Aadhaar number
    
    Returns:
        Extracted Aadhaar number or empty string
    """
    if not text:
        return ''
    
    # Remove spaces and special chars
    cleaned = text.strip()
    
    # Try to find 12 consecutive digits
    import re
    match = re.search(r'\d{12}', cleaned)
    if match:
        candidate = match.group()
        if verhoeff_check(candidate):
            return candidate
    
    # Try with separators (spaces, dashes)
    match_sep = re.search(r'\d{4}[\s\-]?\d{4}[\s\-]?\d{4}', cleaned)
    if match_sep:
        candidate = re.sub(r'[\s\-]', '', match_sep.group())
        if len(candidate) == 12 and verhoeff_check(candidate):
            return candidate
    
    return ''


if __name__ == '__main__':
    # Test examples
    test_cases = [
        ('275492384017', True),   # Valid example
        ('123456789012', False),  # Invalid
        ('012345678901', False),  # Invalid checksum
    ]
    
    print("Verhoeff Algorithm Test Cases:")
    print("-" * 50)
    for number, expected in test_cases:
        result = verhoeff_check(number)
        status = "✓" if result == expected else "✗"
        print(f"{status} {number}: {result} (expected {expected})")
    
    # Validate Aadhaar
    print("\nAadhaar Validation Examples:")
    print("-" * 50)
    for aadhaar in ['275492384017', '123456789012']:
        result = validate_aadhaar(aadhaar)
        print(f"{aadhaar}: {result['valid']} - {result['message']}")
