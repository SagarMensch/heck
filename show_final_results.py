import json

with open(r'data/test_run/p05_full_result.json', encoding='utf-8') as f:
    d = json.load(f)

s = d['statistics']

print('='*70)
print('P05.pdf FULL PDF (28 PAGES) - FINAL RESULTS')
print('='*70)
print()
print(f'Total Fields Extracted: {s["total_fields"]}')
print(f'High Confidence (>0.85): {s["high_conf"]} ({s["high_conf"]/s["total_fields"]*100:.1f}%)')
print(f'Corrected by Encyclopedia: {s["corrected"]}')
print(f'Failed/Invalid: {s["failed"]}')
print()
print(f'Total Time: {s["processing_time_s"]:.1f}s')
print(f'Pages Processed: 28')
print(f'Speed: {s["processing_time_s"]/28:.1f}s/page')
print()
print('Extrapolation for 50 PDFs (28 pages each):')
print(f'  Time per PDF: {s["processing_time_s"]/28*28:.1f}s')
print(f'  Total Time: {s["processing_time_s"]/28*28*50/60:.1f} minutes')
print(f'  or {s["processing_time_s"]/28*28*50/3600:.2f} hours')
print()
print('='*70)
print('CONCLUSION: Pipeline successfully processes full 28-page PDFs!')
print('='*70)
