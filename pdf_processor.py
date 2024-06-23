import PyPDF2
import nltk
from openai import OpenAI
import numpy as np
import json
import os
import logging
from dotenv import load_dotenv
from datetime import datetime
import tiktoken

# ロギングの設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルから環境変数を読み込む
load_dotenv()

# OpenAI クライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# APIキーが設定されていない場合のエラーチェック
if not client.api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

# NLTKのpunkt tokenizerのダウンロード
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class PDFProcessor:
    def __init__(self):
        self.chunks = []
        self.output_folder = "processed_pdfs"
        self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def process_pdf(self, pdf_file):
        logging.info(f"Processing PDF file: {pdf_file.name}")
        try:
            # 出力フォルダの作成（存在しない場合）
            os.makedirs(self.output_folder, exist_ok=True)

            # PDFファイルからテキストを抽出
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                return "Error: No text could be extracted from the PDF file."
            
            self.chunks = self.chunk_text(text)
            
            if not self.chunks:
                return "Error: No valid text chunks could be created from the PDF content."
            
            # 埋め込みの生成
            embeddings = self.create_embeddings()
            
            # 出力ファイルの作成
            output_path = self.create_output_file(pdf_file.name, self.chunks, embeddings)
            
            logging.info(f"Processed PDF and saved results to {output_path}")
            return f"Processed PDF and saved results to {output_path}"
        
        except Exception as e:
            logging.error(f"An error occurred while processing the PDF: {str(e)}", exc_info=True)
            return f"An error occurred while processing the PDF: {str(e)}"

    def extract_text_from_pdf(self, pdf_file):
        logging.info("Extracting text from PDF")
        text = ""
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            if len(reader.pages) == 0:
                raise ValueError("The uploaded PDF file is empty.")
            for page in reader.pages:
                text += page.extract_text()
            return text
        except PyPDF2.errors.PdfReadError as e:
            logging.error(f"Error reading PDF: {str(e)}")
            raise ValueError("The uploaded file is not a valid PDF or is corrupted.")

    def chunk_text(self, text, max_tokens=500):
        logging.info("Chunking text")
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        sentences = nltk.sent_tokenize(text)
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            if current_tokens + sentence_tokens > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        logging.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_embeddings(self):
        logging.info("Creating embeddings")
        try:
            response = client.embeddings.create(
                input=self.chunks,
                model="text-embedding-3-small"
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logging.error(f"Error creating embeddings: {str(e)}", exc_info=True)
            return [[] for _ in self.chunks]  # 空の埋め込みを返す

    def create_output_file(self, original_filename, chunks, embeddings):
        logging.info("Creating output file")
        now = datetime.now()
        date_time = now.strftime("%Y%m%d_%H%M%S")
        original_filename = os.path.splitext(os.path.basename(original_filename))[0]
        output_filename = f"{original_filename}_{date_time}_processed.json"
        output_path = os.path.join(self.output_folder, output_filename)
        
        output = [
            {
                "chunk_id": i,
                "text": chunk,
                "embedding": embedding
            } for i, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1)
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        return output_path

# 使用例
if __name__ == "__main__":
    processor = PDFProcessor()
    result = processor.process_pdf("example.pdf")
    print(result)