import os
import logging
import glob
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from discussion_generator import DiscussionGenerator
import gradio as gr

# AWSの認証情報チェックをスキップ
os.environ['AWS_EC2_METADATA_DISABLED'] = 'true'

# ロギングの設定
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# .envファイルから環境変数を読み込む
load_dotenv()

# PDFProcessorの初期化
pdf_processor = PDFProcessor()

def process_pdf(pdf_file):
    logging.info(f"Processing PDF file: {pdf_file.name}")
    try:
        result = pdf_processor.process_pdf(pdf_file)
        logging.info(f"PDF processing result: {result}")
        return result
    except Exception as e:
        logging.error(f"Error processing PDF: {str(e)}", exc_info=True)
        return f"Error processing PDF: {str(e)}"

def generate_discussion(topic, rounds):
    logging.info(f"Generating discussion for topic: {topic}, rounds: {rounds}")
    try:
        latest_json = get_latest_json_file()
        if latest_json is None:
            return "Error: No processed PDF files found. Please process a PDF first.", None

        discussion_generator = DiscussionGenerator(latest_json)
        discussion_data, filepath = discussion_generator.generate_discussion(topic, rounds=int(rounds))
        
        # Format the discussion for display
        formatted_discussion = f"Discussion Topic: {discussion_data['topic']}\n\n"
        for i, round_data in enumerate(discussion_data['rounds'], 1):
            formatted_discussion += f"Round {i}:\n"
            for model, data in round_data.items():
                formatted_discussion += f"{model}:\n{data['data']['response']}\n\n"
        
        logging.info(f"Discussion generated and saved to {filepath}")
        return formatted_discussion, filepath
    except Exception as e:
        logging.error(f"Error generating discussion: {str(e)}", exc_info=True)
        return f"An error occurred while generating the discussion: {str(e)}", None

def get_latest_json_file():
    logging.info("Getting latest JSON file")
    if not hasattr(pdf_processor, 'output_folder') or pdf_processor.output_folder is None:
        logging.error("PDF processor output folder is not set")
        return None
    json_files = glob.glob(os.path.join(pdf_processor.output_folder, "*.json"))
    if not json_files:
        logging.warning("No JSON files found")
        return None
    latest_file = max(json_files, key=os.path.getctime)
    logging.info(f"Latest JSON file: {latest_file}")
    return latest_file

with gr.Blocks() as demo:
    with gr.Tab("PDF Processing"):
        pdf_input = gr.File(label="Upload PDF")
        process_button = gr.Button("Process PDF")
        pdf_output = gr.Textbox(label="Processing Result")
        
        process_button.click(
            process_pdf,
            inputs=[pdf_input],
            outputs=[pdf_output]
        )
    
    with gr.Tab("Generate Discussion"):
        topic_input = gr.Textbox(label="Enter the discussion topic")
        rounds_input = gr.Slider(minimum=1, maximum=5, step=1, label="Number of discussion rounds", value=1)
        generate_button = gr.Button("Generate Discussion")
        discussion_output = gr.Textbox(label="Generated Discussion")
        json_output = gr.Textbox(label="JSON Output File")
        
        generate_button.click(
            generate_discussion,
            inputs=[topic_input, rounds_input],
            outputs=[discussion_output, json_output]
        )

if __name__ == "__main__":
    logging.info("Starting main program")
    try:
        demo.launch()
    except Exception as e:
        logging.error(f"An error occurred in the main program: {str(e)}", exc_info=True)
    logging.info("Main program completed")





