import os

def find_markdown_files(base_dir='.'):
    md_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith('.md'):
                md_files.append(os.path.join(root, file))
    return md_files

# 사용 예시
if __name__ == "__main__":
    markdown_files = find_markdown_files()
    print(f"총 {len(markdown_files)}개의 md 파일 발견:")
    for f in markdown_files:
        print(f)

from PyPDF2 import PdfMerger

pdfs = ['file1.pdf', 'file2.pdf', 'file3.pdf']
merger = PdfMerger()

for pdf in pdfs:
    merger.append(pdf)

merger.write("merged.pdf")
merger.close()