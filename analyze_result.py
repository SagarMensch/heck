import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\aigcp_gpuadmin\Downloads\LICRFP\LICF\data\output\e2e_p02_result.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print('form_status:', data.get('form_status'))
print('pages_processed:', data.get('pages_processed'))
print('processing_time:', round(data.get('processing_time_seconds', 0), 1), 's')
kpis = data.get('kpis', {})
for k, v in kpis.items():
    print(f'  KPI {k}: {v}')
print()
fields = data.get('fields', {})
extracted = {k: v for k, v in fields.items() if v.get('value') and v.get('confidence', 0) > 0.3}
missing = {k: v for k, v in fields.items() if not v.get('value') or v.get('confidence', 0) <= 0.3}
hallucinated = {k: v for k, v in fields.items() if v.get('validation_status') == 'hallucination'}
print(f'Fields with value (conf>0.3): {len(extracted)}/{len(fields)}')
print(f'Missing/empty: {len(missing)}')
print(f'Hallucinated: {len(hallucinated)}')
print()
print('--- EXTRACTED FIELDS ---')
for fname, fdata in sorted(extracted.items()):
    val = fdata.get('value', '')[:80]
    conf = fdata.get('confidence', 0)
    src = fdata.get('selected_model', '?')
    cat = fdata.get('category', '')
    print(f'  {fname}: val="{val}" conf={conf:.2f} src={src} cat={cat}')
print()
print('--- HALLUCINATED FIELDS ---')
for fname, fdata in sorted(hallucinated.items()):
    val = fdata.get('value', '')[:80]
    conf = fdata.get('confidence', 0)
    msg = fdata.get('validation_message', '')[:80]
    print(f'  {fname}: val="{val}" conf={conf:.2f} msg="{msg}"')
print()
print('--- MISSING FIELDS (first 20) ---')
for i, (fname, fdata) in enumerate(sorted(missing.items())):
    if i >= 20:
        break
    tname = fdata.get('template_name', '')
    fam = fdata.get('field_family', '')
    src = fdata.get('source', '')
    print(f'  {fname} (template={tname}, family={fam}, src={src})')
