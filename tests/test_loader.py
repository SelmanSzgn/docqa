import pymupdf

from docqa.loader import extract_pages


def test_extract_pages(tmp_path):
    pdf_path = tmp_path / "file.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "alpha")
    page2 = doc.new_page()
    page2.insert_text((72, 72), "beta")
    doc.save(pdf_path)
    doc.close()

    pages = extract_pages(pdf_path)
    assert pages[0][0] == 1
    assert pages[0][1] == "alpha\n"
    assert pages[1][0] == 2
    assert pages[1][1] == "beta\n"
