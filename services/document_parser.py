from io import BytesIO
from pathlib import Path

import pandas as pd
from docx import Document as DocxDocument
from pypdf import PdfReader


class DocumentParser:
    def parse(self, file_name, file_bytes):
        suffix = Path(file_name).suffix.lower()
        if suffix == ".txt":
            text = self._decode_text(file_bytes)
        elif suffix == ".pdf":
            text = self._parse_pdf(file_bytes)
        elif suffix == ".docx":
            text = self._parse_docx(file_bytes)
        elif suffix == ".xlsx":
            text = self._parse_excel(file_bytes)
        elif suffix == ".csv":
            text = self._parse_csv(file_bytes)
        else:
            raise ValueError(f"暂不支持的文件类型: {suffix}")

        normalized = self._normalize_text(text)
        if not normalized:
            raise ValueError("文档中没有提取到可用文本，请检查文件内容。")
        return normalized, suffix

    def _decode_text(self, file_bytes):
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                return file_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("文本文件编码无法识别，建议转为 UTF-8 后再上传。")

    def _parse_pdf(self, file_bytes):
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    def _parse_docx(self, file_bytes):
        document = DocxDocument(BytesIO(file_bytes))
        paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return "\n".join(paragraphs)

    def _parse_excel(self, file_bytes):
        excel_file = pd.ExcelFile(BytesIO(file_bytes))
        sheet_texts = []
        for sheet_name in excel_file.sheet_names:
            dataframe = excel_file.parse(sheet_name, dtype=str).fillna("")
            sheet_texts.append(f"工作表：{sheet_name}\n{self._dataframe_to_text(dataframe)}")
        return "\n\n".join(sheet_texts)

    def _parse_csv(self, file_bytes):
        dataframe = None
        for encoding in ("utf-8", "utf-8-sig", "gbk"):
            try:
                dataframe = pd.read_csv(BytesIO(file_bytes), dtype=str, encoding=encoding).fillna("")
                break
            except UnicodeDecodeError:
                continue
        if dataframe is None:
            raise ValueError("CSV 文件编码无法识别，建议转为 UTF-8 后再上传。")
        return self._dataframe_to_text(dataframe)

    def _dataframe_to_text(self, dataframe):
        headers = [str(column).strip() for column in dataframe.columns.tolist()]
        lines = [" | ".join(headers)] if headers else []
        for _, row in dataframe.iterrows():
            values = [str(value).strip() for value in row.tolist()]
            if any(values):
                lines.append(" | ".join(values))
        return "\n".join(lines)

    def _normalize_text(self, text):
        clean_text = text.replace("\ufeff", "")
        lines = [line.strip() for line in clean_text.replace("\r\n", "\n").split("\n")]
        filtered = [line for line in lines if line]
        return "\n".join(filtered)
