import os
from PyPDF2 import PdfMerger, PdfReader

# 하위 디렉터리까지 포함해서 파일 검색하기 (ext로 확장자 지정 가능)
def find_files(ext=None, base_dir='.'):
    result = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if ext:
                if file.endswith(ext):
                    result.append(os.path.join(root, file))
            else: result.append(os.path.join(root, file))
    return result
# pdf 파일 검색 후 파일 갯수 출력(파일명은 주석처리)
pdf_files = find_files('pdf')
print(f"총 {len(pdf_files)}개의 파일 발견:")
# for pdf in pdf_files: print(f"  {pdf}")

# pdf 파일 병합
merger = PdfMerger()
for pdf in pdf_files:
    merger.append(PdfReader(open(pdf, 'rb')))
    print(f"  {pdf}") # 병합 파일 출력

# 병합된 pdf 파일 저장 / 저장 완료 출력
fileName = "제35회ADP실기오픈북[251015].pdf"
merger.write(fileName)
merger.close()
print(fileName, "생성 완료")
