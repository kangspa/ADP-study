from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter  # 페이지 크기: 필요시 A4로 교체
from io import BytesIO

# 1️⃣ 페이지 넘버용 오버레이 PDF 만들기
def create_page_number_pdf(total_pages, page_size=letter):
    packet = BytesIO()
    c = canvas.Canvas(packet, pagesize=page_size)
    width, height = page_size

    for i in range(total_pages):
        page_num = i + 1
        # 홀수: 우측 하단 / 짝수: 좌측 하단
        if page_num % 2 == 1:
            x = width - 45   # 오른쪽 여백
        else:
            x = 15           # 왼쪽 여백
        y = 15  # 아래쪽 여백

        c.setFont("Helvetica", 10)
        c.drawString(x, y, str(page_num))
        c.showPage()

    c.save()
    packet.seek(0)
    return PdfReader(packet)

# 2️⃣ 원본 PDF + 페이지 넘버 합치기
def add_page_numbers(input_pdf_path, output_pdf_path):
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    number_overlay = create_page_number_pdf(len(reader.pages))

    for i, page in enumerate(reader.pages):
        overlay_page = number_overlay.pages[i]
        page.merge_page(overlay_page)
        writer.add_page(page)

    with open(output_pdf_path, "wb") as f:
        writer.write(f)

    print(f"✅ '{output_pdf_path}' 생성 완료 (홀수→오른쪽 / 짝수→왼쪽 페이지 번호)")

# 3️⃣ 실행 예시
input_pdf = "제35회ADP실기오픈북-출력용.pdf"
output_pdf = "제35회ADP실기오픈북-페이지추가.pdf"
add_page_numbers(input_pdf, output_pdf)
