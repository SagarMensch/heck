"""
SOTA TABLE EXTRACTOR FOR LIC FORM 300
Handles 2-column AND 3-column tables dynamically
Extracts handwritten VALUES, not just labels
"""
import os
import re
from typing import List, Dict, Tuple
from bs4 import BeautifulSoup

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"


def parse_table_sota(html_str: str) -> List[Dict[str, str]]:
    """
    Parse HTML table with intelligent column detection.
    
    LIC Form 300 table structure:
    Option A (2-col): | Label | Value |
    Option B (3-col): | # | Label | Value |
    
    Returns: List of {label, value, confidence}
    """
    if not html_str or not html_str.strip().startswith('<'):
        return []
    
    pairs = []
    
    try:
        soup = BeautifulSoup(html_str, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return parse_table_regex(html_str)
        
        # Analyze first row to detect column structure
        first_row = table.find('tr')
        if not first_row:
            return []
        
        cells = first_row.find_all(['td', 'th'])
        num_cols = len(cells)
        
        # Detect structure
        if num_cols >= 3:
            # 3-column format: [#] [Label] [Value]
            pairs = parse_3col_table(table)
        elif num_cols == 2:
            # 2-column format: [Label] [Value]
            pairs = parse_2col_table(table)
        else:
            # Single column - treat as label only
            pairs = parse_1col_table(table)
        
        return pairs
    
    except Exception as e:
        print(f"Table parsing error: {e}")
        return parse_table_regex(html_str)


def parse_3col_table(table) -> List[Dict]:
    """Parse 3-column table: [#] [Label] [Value]
    CRITICAL: We ONLY want Cell 2 (Value). Cell 1 is printed label.
    """
    pairs = []
    
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        
        if len(cells) >= 3:
            # Cell 0 = index (ignore)
            # Cell 1 = LABEL (ignore - it's printed)
            # Cell 2 = VALUE (handwritten - THIS IS WHAT WE WANT!)
            label = cells[1].get_text(strip=True)
            value = cells[2].get_text(strip=True)
            
            # Return ONLY the value column
            if value and len(value) > 0:
                pairs.append({
                    'label': label,
                    'value': value,
                    'structure': '3col_value_only'
                })
        elif len(cells) == 2:
            # Fallback
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            if value:
                pairs.append({
                    'label': label,
                    'value': value,
                    'structure': '2col_fallback'
                })
    
    return pairs


def parse_2col_table(table) -> List[Dict]:
    """Parse 2-column table: [Label] [Value]"""
    pairs = []
    
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True)
            value = cells[1].get_text(strip=True)
            
            # CRITICAL: Only extract if value cell has content
            if label and value and len(value) > 0:
                pairs.append({
                    'label': label,
                    'value': value,
                    'structure': '2col'
                })
            elif label:
                # Label with no value yet
                pairs.append({
                    'label': label,
                    'value': '',
                    'structure': 'label_only'
                })
    
    return pairs


def parse_1col_table(table) -> List[Dict]:
    """Parse single-column table (labels only)"""
    pairs = []
    
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        
        for cell in cells:
            text = cell.get_text(strip=True)
            if text and len(text) > 2:
                pairs.append({
                    'label': text,
                    'value': '',
                    'structure': '1col'
                })
    
    return pairs


def parse_table_regex(html_str: str) -> List[Dict]:
    """Fallback regex parser"""
    pairs = []
    
    # Match all <td>...</td>
    pattern = r'<t[hd][^>]*>(.*?)</t[hd]>'
    cells = re.findall(pattern, html_str, re.IGNORECASE | re.DOTALL)
    
    # Clean cells
    cleaned = []
    for cell in cells:
        text = re.sub(r'<[^>]+>', '', cell)
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            cleaned.append(text)
    
    # Pair them (assume 3-col first, then 2-col)
    if len(cleaned) % 3 == 0:
        # 3-col structure
        for i in range(0, len(cleaned), 3):
            pairs.append({
                'label': cleaned[i+1] if i+1 < len(cleaned) else '',
                'value': cleaned[i+2] if i+2 < len(cleaned) else '',
                'structure': '3col_regex'
            })
    elif len(cleaned) >= 2:
        # 2-col structure
        for i in range(0, len(cleaned)-1, 2):
            pairs.append({
                'label': cleaned[i],
                'value': cleaned[i+1] if i+1 < len(cleaned) else '',
                'structure': '2col_regex'
            })
    
    return pairs


def extract_field_value_pairs(html_str: str) -> Dict[str, str]:
    """
    Extract all label->value pairs from table HTML.
    Returns dict mapping labels to values.
    """
    pairs = parse_table_sota(html_str)
    
    result = {}
    for pair in pairs:
        label = pair.get('label', '')
        value = pair.get('value', '')
        
        if label and value:
            result[label] = value
    
    return result


if __name__ == "__main__":
    # Test with sample HTML
    test_html = '''
    <html><body><table>
    <tr><td>1</td><td>Name</td><td>Sanvika</td></tr>
    <tr><td>2</td><td>DOB</td><td>18/10/2026</td></tr>
    <tr><td>3</td><td>Gender</td><td>Female</td></tr>
    </table></body></html>
    '''
    
    print("Testing 3-column table:")
    pairs = parse_table_sota(test_html)
    for pair in pairs:
        print(f"  {pair['label']}: {pair['value']}")
    
    print("\nExtracted dict:")
    result = extract_field_value_pairs(test_html)
    for label, value in result.items():
        print(f"  {label} → {value}")
