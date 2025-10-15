import os
from PyPDF2 import PdfMerger, PdfReader

def find_files(ext=None, base_dir='.'):
    result = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if ext:
                if file.endswith(ext):
                    result.append(os.path.join(root, file))
            else: result.append(os.path.join(root, file))
    return result

pdf_files = find_files('pdf')
print(f"총 {len(pdf_files)}개의 파일 발견:")
for pdf in pdf_files:
    print(f"  {pdf}")
'''
merger = PdfMerger()
for pdf in pdf_files:
    merger.append(PdfReader(open(pdf, 'rb')))
    print(f"  {pdf}")

merger.write("merged.pdf")
merger.close()
print("merged.pdf 생성 완료")
'''