"""
HTML Table Parser for LIC Form 300
Extracts structured field->value pairs from PaddleX table HTML
"""
import re
from typing import List, Dict, Optional

# Try to import BeautifulSoup, fall back to regex if not available
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

def parse_table_html_regex(html_str: str) -> List[Dict[str, str]]:
    """Fallback table parser using regex (no dependencies)"""
    rows = []
    # Extract all <td>...</td> or <th>...</th> content
    cells = re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', html_str, re.IGNORECASE | re.DOTALL)
    
    # Group into pairs (label, value)
    for i in range(0, len(cells) - 1, 2):
        label = re.sub(r'<[^>]+>', '', cells[i]).strip()
        value = re.sub(r'<[^>]+>', '', cells[i+1]).strip()
        if label or value:
            rows.append({'label': label, 'value': value})
    
    return rows


def parse_table_html(html_str: str) -> List[Dict[str, str]]:
    """
    Parse HTML table from PaddleX output into structured rows
    
    Args:
        html_str: HTML table string from PaddleX
    
    Returns:
        List of dicts with 'label' and 'value' keys
    """
    if not html_str or not html_str.strip().startswith('<'):
        return []
    
    if HAS_BS4:
        try:
            soup = BeautifulSoup(html_str, 'html.parser')
            rows = []
            table = soup.find('table')
            if not table:
                return parse_table_html_regex(html_str)
            
            for tr in table.find_all('tr'):
                cells = tr.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].get_text(strip=True)
                    value = cells[1].get_text(strip=True)
                    if label or value:
                        rows.append({'label': label, 'value': value})
                elif len(cells) == 1:
                    text = cells[0].get_text(strip=True)
                    if text:
                        rows.append({'label': text, 'value': ''})
            
            return rows
        except:
            return parse_table_html_regex(html_str)
    else:
        return parse_table_html_regex(html_str)


def extract_key_values_from_table(html_str: str) -> Dict[str, str]:
    """
    Extract key-value pairs from table HTML
    Returns a dict mapping labels to values
    """
    rows = parse_table_html(html_str)
    result = {}
    
    for row in rows:
        label = row.get('label', '')
        value = row.get('value', '')
        
        if label and value:
            # Store in result
            result[label] = value
    
    return result


if __name__ == "__main__":
    # Test
    test_html = '<html><body><table><tr><td>Name</td><td>John</td></tr><tr><td>Age</td><td>30</td></tr></table></body></html>'
    print(parse_table_html(test_html))
