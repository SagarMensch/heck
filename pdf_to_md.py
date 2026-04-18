import os
import sys
import pypdfium2 as pdfium

def pdf_to_md(pdf_path, md_path):
    """Convert PDF to Markdown text file."""
    try:
        pdf = pdfium.PdfDocument(pdf_path)
        text = ""
        for page in pdf:
            textpage = page.get_textpage()
            text += textpage.get_text_range() + "\n\n"
        pdf.close()
        
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Converted: {pdf_path} -> {md_path}")
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")

def main():
    # Get current directory
    current_dir = os.getcwd()
    
    # Find all PDF files
    pdf_files = [f for f in os.listdir(current_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in the current directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s).")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(current_dir, pdf_file)
        # Create MD filename by replacing .pdf with .md
        md_file = os.path.splitext(pdf_file)[0] + '.md'
        md_path = os.path.join(current_dir, md_file)
        
        pdf_to_md(pdf_path, md_path)

if __name__ == "__main__":
    main()