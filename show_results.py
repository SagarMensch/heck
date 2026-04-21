import json

with open('data/accuracy_output/p02_accurate_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('='*70)
print('EXTRACTED DATA FROM P02.pdf (First 5 Pages)')
print('='*70)
print()

# Show fields that were corrected or have interesting values
for i, item in enumerate(data['extracted_fields']):
    val = item.get('value', '')
    raw = item.get('raw', '')
    method = item.get('method', '')
    
    # Show non-basic_clean methods or fields with actual values
    if method != 'basic_clean' or (val and len(val) > 3):
        field_name = item.get('field', 'unknown')[:40]
        value_short = str(val)[:50].replace('\n', ' ')
        raw_short = str(raw)[:60].replace('\n', ' ') if raw else ''
        conf = item.get('confidence', 0)
        status = item.get('status', '')
        
        # Encode to ASCII to avoid unicode errors
        field_safe = field_name.encode('ascii', 'replace').decode('ascii')
        value_safe = value_short.encode('ascii', 'replace').decode('ascii')
        raw_safe = raw_short.encode('ascii', 'replace').decode('ascii')
        
        print(f"{i}. Field: {field_safe}")
        print(f"   Value: {value_safe}")
        print(f"   Raw:   {raw_safe}")
        print(f"   Conf: {conf:.2f}, Status: {status}, Method: {method}")
        print()

print()
print('='*70)
print('STATISTICS')
print('='*70)
stats = data.get('statistics', {})
print(f"Total fields: {stats.get('total_fields', 0)}")
print(f"High confidence: {stats.get('high_conf', 0)}")
print(f"Corrected: {stats.get('corrected', 0)}")
print(f"Failed: {stats.get('failed', 0)}")
print(f"Time: {stats.get('processing_time_s', 0):.1f}s")
