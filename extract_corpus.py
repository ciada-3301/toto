import os
import sys
import glob
import json
import re
import zipfile
import xml.etree.ElementTree as ET

sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
JSONL_OUTPUT = os.path.join(DATA_DIR, "parallel_corpus.jsonl")

def is_bengali(text):
    if not text:
        return False
    # Check if any character falls in Bengali Unicode range 0980-09FF
    return any(0x0980 <= ord(c) <= 0x09FF for c in text)

def is_english_or_latin(text):
    if not text:
        return False
    latin_chars = sum(1 for c in text if ('a' <= c.lower() <= 'z'))
    bengali_chars = sum(1 for c in text if 0x0980 <= ord(c) <= 0x09FF)
    return latin_chars > 0 and latin_chars > bengali_chars

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def parse_xlsx_sheets(filepath):
    records = []
    try:
        with zipfile.ZipFile(filepath) as z:
            shared_strings = []
            if 'xl/sharedStrings.xml' in z.namelist():
                tree = ET.fromstring(z.read('xl/sharedStrings.xml'))
                for elem in tree.iter():
                    if elem.tag.endswith('}t'):
                        shared_strings.append(elem.text or '')

            sheet_files = [f for f in z.namelist() if f.startswith('xl/worksheets/sheet')]
            for sf in sheet_files:
                tree = ET.fromstring(z.read(sf))
                sheet_rows = {}
                for c in tree.iter():
                    if c.tag.endswith('}c'):
                        cell_ref = c.attrib.get('r', '')
                        cell_type = c.attrib.get('t', '')
                        row_num = int(re.search(r'\d+', cell_ref).group()) if re.search(r'\d+', cell_ref) else 0
                        col_name = re.search(r'[A-Z]+', cell_ref).group() if re.search(r'[A-Z]+', cell_ref) else ''

                        v_elem = c.find('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v')
                        val = ""
                        if v_elem is not None and v_elem.text:
                            val = v_elem.text
                            if cell_type == 's' and val.isdigit() and int(val) < len(shared_strings):
                                val = shared_strings[int(val)]

                        if val:
                            if row_num not in sheet_rows:
                                sheet_rows[row_num] = {}
                            sheet_rows[row_num][col_name] = clean_text(val)

                for r_num in sorted(sheet_rows.keys()):
                    row_data = sheet_rows[r_num]
                    vals = list(row_data.values())
                    eng_candidates = [v for v in vals if is_english_or_latin(v) and len(v) > 1 and not v.startswith('/')]
                    ben_candidates = [v for v in vals if is_bengali(v) and len(v) > 1]
                    
                    if eng_candidates and ben_candidates:
                        eng_txt = eng_candidates[0]
                        ben_txt = ben_candidates[0]
                        
                        roman_candidates = [v for v in eng_candidates if v != eng_txt and not any(kw in v.lower() for kw in ['noun', 'verb', 'adj', 'section', 'category', 'english', 'bengali'])]
                        roman_txt = roman_candidates[0] if roman_candidates else ""

                        records.append({
                            'english': eng_txt,
                            'toto_bengali': ben_txt,
                            'toto_roman': roman_txt,
                            'source': f"{os.path.basename(filepath)}:row{r_num}"
                        })
    except Exception as e:
        print(f"Error parsing xlsx {filepath}: {e}")
    return records

def parse_docx_files(filepath):
    records = []
    try:
        with zipfile.ZipFile(filepath) as z:
            tree = ET.fromstring(z.read('word/document.xml'))
            paragraphs = []
            for p in tree.iter():
                if p.tag.endswith('}p'):
                    txt = ''.join([e.text for e in p.iter() if e.tag.endswith('}t') and e.text])
                    cleaned = clean_text(txt)
                    if cleaned:
                        paragraphs.append(cleaned)
            
            for i in range(len(paragraphs) - 1):
                p1 = paragraphs[i]
                p2 = paragraphs[i+1]
                if is_english_or_latin(p1) and is_bengali(p2):
                    records.append({
                        'english': p1,
                        'toto_bengali': p2,
                        'toto_roman': "",
                        'source': f"{os.path.basename(filepath)}:p{i}"
                    })
                elif is_bengali(p1) and is_english_or_latin(p2):
                    records.append({
                        'english': p2,
                        'toto_bengali': p1,
                        'toto_roman': "",
                        'source': f"{os.path.basename(filepath)}:p{i}"
                    })
    except Exception as e:
        print(f"Error parsing docx {filepath}: {e}")
    return records

def main():
    all_records = []

    excel_files = glob.glob("**/*.xlsx", recursive=True)
    print(f"Processing {len(excel_files)} Excel files...")
    for ef in excel_files:
        recs = parse_xlsx_sheets(ef)
        print(f"  Extracted {len(recs)} pairs from {ef}")
        all_records.extend(recs)

    docx_files = [f for f in glob.glob("**/*.docx", recursive=True) if not os.path.basename(f).startswith("PROGRESS")]
    print(f"\nProcessing {len(docx_files)} Word document files...")
    for df in docx_files:
        recs = parse_docx_files(df)
        print(f"  Extracted {len(recs)} pairs from {df}")
        all_records.extend(recs)

    filtered = []
    seen = set()
    for r in all_records:
        eng = r['english']
        ben = r['toto_bengali']
        
        if any(h in eng.upper() for h in ['ENGLISH', 'BENGALI', 'SECTION', 'CATEGORY', 'GRAMMATICAL', 'IPA', 'CHANGES', 'DENOTATIVE']):
            continue
        if len(eng) < 2 or len(ben) < 2:
            continue
            
        pair_key = (eng.strip().lower(), ben.strip())
        if pair_key not in seen:
            seen.add(pair_key)
            filtered.append(r)

    print(f"\nTotal unique valid parallel pairs extracted: {len(filtered)}")

    with open(JSONL_OUTPUT, "w", encoding="utf-8") as f:
        for r in filtered:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved dataset to {JSONL_OUTPUT}")

if __name__ == "__main__":
    main()
