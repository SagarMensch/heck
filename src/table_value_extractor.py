"""
Advanced Table Value Extractor
Extracts label->value pairs from PaddleX table HTML
"""
import re
from typing import List, Dict, Tuple, Optional
from bs4 import BeautifulSoup

def parse_table_to_pairs(html_str: str) -> List[Dict[str, str]]:
    """
    Parse HTML table and extract label-value pairs.
    CRITICAL: Extract actual VALUES from second cell, not just labels!
    
    Input: HTML table from PaddleX
    Output: List of dicts with {'label': str, 'value': str}
    """
    if not html_str or not html_str.strip().startswith('<'):
        return []
    
    pairs = []
    
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return parse_table_regex(html_str)
        
        # Process each row
        for tr in table.find_all('tr'):
            cells = tr.find_all(['td', 'th'])
            
            if len(cells) >= 2:
                # Cell 0 = label, Cell 1 = VALUE (handwritten data!)
                label = cells[0].get_text(strip=True)
                value = cells[1].get_text(strip=True)
                
                # ONLY add if we have actual value (not empty)
                if value and len(value.strip()) > 0:
                    pairs.append({
                        'label': label,
                        'value': value,
                        'row_type': 'standard'
                    })
                elif label and len(label.strip()) > 0:
                    # If no value in cell 1, check cell 0 for label-only
                    pairs.append({
                        'label': label,
                        'value': '',
                        'row_type': 'label_only'
                    })
            
            elif len(cells) == 1:
                # Single cell - section header or standalone label
                text = cells[0].get_text(strip=True)
                if text and len(text) > 3:
                    pairs.append({
                        'label': text,
                        'value': '',
                        'row_type': 'header'
                    })
    
    except Exception as e:
        print(f"Table parsing error: {e}")
        return parse_table_regex(html_str)
    
    return pairs


def parse_table_regex(html_str: str) -> List[Dict[str, str]]:
    """Fallback regex-based table parser"""
    pairs = []
    
    # Match all <td>...</td> or <th>...</th>
    pattern = r'<t[hd][^>]*>(.*?)</t[hd]>'
    cells = re.findall(pattern, html_str, re.IGNORECASE | re.DOTALL)
    
    # Clean cells
    cleaned_cells = []
    for cell in cells:
        text = re.sub(r'<[^>]+>', '', cell)  # Remove tags
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
        if text:
            cleaned_cells.append(text)
    
    # Pair them up
    for i in range(0, len(cleaned_cells) - 1, 2):
        pairs.append({
            'label': cleaned_cells[i],
            'value': cleaned_cells[i + 1] if i + 1 < len(cleaned_cells) else ''
        })
    
    return pairs


def extract_field_value_from_table(html_str: str, field_keywords: List[str]) -> Optional[str]:
    """
    Extract value for a specific field from table.
    
    Args:
        html_str: HTML table string
        field_keywords: List of keywords to match (e.g., ['name', 'full name'])
    
    Returns:
        Extracted value or None
    """
    pairs = parse_table_to_pairs(html_str)
    
    for pair in pairs:
        label = pair['label'].lower()
        value = pair['value']
        
        # Check if label matches any keyword
        for keyword in field_keywords:
            if keyword.lower() in label:
                return value.strip() if value else None
    
    return None


def extract_all_key_values(html_str: str) -> Dict[str, str]:
    """
    Extract all key-value pairs from table.
    
    Returns:
        Dict mapping labels to values
    """
    pairs = parse_table_to_pairs(html_str)
    result = {}
    
    for pair in pairs:
        label = pair['label']
        value = pair['value']
        
        if label and value:
            result[label] = value
    
    return result


if __name__ == "__main__":
    # Test
    test_html = '<html><body><table><tr><td>Name</td><td>John Doe</td></tr><tr><td>Age</td><td>30</td></tr></table></body></html>'
    pairs = parse_table_to_pairs(test_html)
    
    print("Extracted pairs:")
    for pair in pairs:
        print(f"  {pair['label']}: {pair['value']}")
