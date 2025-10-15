from PyPDF2 import PdfMerger, PdfReader

pdf_files = ["제35회 데이터분석전문가(ADP) 실기 응시가이드.pdf", "제35회ADP실기오픈북[251015].pdf"]

# pdf 파일 병합
merger = PdfMerger()

# 1️⃣ 첫 번째 PDF → 마지막 2페이지 제외하고 추가
first_pdf = PdfReader(open(pdf_files[0], 'rb'))
num_pages = len(first_pdf.pages)
merger.append(first_pdf, pages=(0, num_pages - 2))  # 마지막 2페이지 제외
print(f"{pdf_files[0]} (총 {num_pages}p → {num_pages - 2}p만 추가됨)")

# 2️⃣ 두 번째 PDF → 전체 추가
second_pdf = PdfReader(open(pdf_files[1], 'rb'))
merger.append(second_pdf)
print(f"{pdf_files[1]} (전체 추가됨)")

# 병합된 pdf 파일 저장 / 저장 완료 출력
fileName = "제35회ADP실기오픈북-출력용.pdf"
merger.write(fileName)
merger.close()
print(fileName, "생성 완료")
