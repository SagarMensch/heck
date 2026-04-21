#!/usr/bin/env python
"""
Generate AUDIT REPORT for P05.pdf extraction
Shows every extracted field with value, confidence, and validation status
"""
import json
import os

OUTPUT_FILE = r'data/test_run/p05_full_result.json'
AUDIT_TXT = r'data/test_run/audit_report.txt'
AUDIT_CSV = r'data/test_run/audit_report.csv'

with open(OUTPUT_FILE, encoding='utf-8') as f:
    data = json.load(f)

fields = data['extracted_fields']
stats = data['statistics']

# Generate audit report
lines = []
lines.append('='*100)
lines.append('LIC FORM 300 - P05.pdf EXTRACTION AUDIT REPORT')
lines.append('='*100)
lines.append('')
lines.append('SUMMARY STATISTICS')
lines.append('-'*100)
lines.append(f"Total Fields Extracted: {stats['total_fields']}")
lines.append(f"High Confidence (>0.85): {stats['high_conf']} ({stats['high_conf']/stats['total_fields']*100:.1f}%)")
lines.append(f"Corrected by Encyclopedia: {stats['corrected']}")
lines.append(f"Failed/Invalid: {stats['failed']}")
lines.append(f"Processing Time: {stats['processing_time_s']:.1f}s")
lines.append(f"Pages Processed: 28")
lines.append(f"Speed: {stats['processing_time_s']/28:.1f}s/page")
lines.append('')
lines.append('='*100)
lines.append('DETAILED FIELD-BY-FIELD EXTRACTION')
lines.append('='*100)
lines.append('')

# Group by page
by_page = {}
for item in fields:
    page = item.get('source_page', 0)
    if page not in by_page:
        by_page[page] = []
    by_page[page].append(item)

for page_num in sorted(by_page.keys()):
    page_fields = by_page[page_num]
    lines.append(f'PAGE {page_num}')
    lines.append('-'*100)
    
    for i, item in enumerate(page_fields, 1):
        field_name = item['field']
        value = item['value'][:80] if item['value'] else '(empty)'
        raw = item.get('raw', '')[:60] if item.get('raw') else ''
        conf = item['confidence']
        status = item['status']
        method = item.get('method', 'unknown')
        notes = item.get('notes', '')
        
        icon = '✓' if status in ['valid', 'corrected'] else '!'
        
        lines.append(f'{i:3}. {icon} Field: {field_name}')
        lines.append(f'     Value: {value}')
        if raw and raw != value:
            lines.append(f'     Raw:   {raw}')
        lines.append(f'     Conf: {conf:.2f} | Status: {status:<15} | Method: {method}')
        if notes:
            lines.append(f'     Notes: {notes}')
        lines.append('')
    
    lines.append('')

lines.append('='*100)
lines.append('END OF AUDIT REPORT')
lines.append('='*100)

# Write text report
with open(AUDIT_TXT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

print(f'Text audit report saved to: {AUDIT_TXT}')

# Write CSV report
import csv
with open(AUDIT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Page', 'Field_Name', 'Value', 'Raw_Value', 'Confidence', 'Status', 'Method', 'Notes'])
    
    for page_num in sorted(by_page.keys()):
        page_fields = by_page[page_num]
        for i, item in enumerate(page_fields, 1):
            writer.writerow([
                page_num,
                item['field'],
                item['value'],
                item.get('raw', ''),
                item['confidence'],
                item['status'],
                item.get('method', ''),
                item.get('notes', '')
            ])

print(f'CSV audit report saved to: {AUDIT_CSV}')
print(f'\nTotal fields audited: {len(fields)}')
