# Academic Paper Discussion Generator

This application processes academic papers and generates AI-powered discussions on specified topics. It uses a combination of vector and keyword search techniques to analyze paper content and simulate a discussion between two AI models.

## Features

- PDF processing and text extraction
- Vector search using OpenAI's embedding model
- Keyword search with intelligent keyword extraction
- AI-powered discussion generation using GPT models
- Comparison of vector and keyword search methods in information retrieval
- User-friendly Gradio interface for PDF processing and discussion generation
- JSON output of processed papers and generated discussions

## Requirements

- Python 3.7+
- OpenAI API key
- Various Python libraries (see `requirements.txt`)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/academic-paper-discussion-generator.git
   cd academic-paper-discussion-generator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up your OpenAI API key:
   Create a `.env` file in the project root and add your API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```
   python main.py
   ```
   This command launches a Gradio interface with two tabs:

   a. **PDF Processing**:
   - Upload your academic paper in PDF format.
   - Click "Process PDF" to extract and process the text.
   - The processed data will be saved in the `processed_pdfs` directory.

   b. **Generate Discussion**:
   - Enter a discussion topic.
   - Set the number of discussion rounds.
   - Click "Generate Discussion" to create an AI-powered discussion based on the processed papers.
   - The generated discussion will be saved as a JSON file in the `discussion_results` directory.

2. Interact with the Gradio interface:
   - Use the PDF Processing tab to process new academic papers as needed.
   - Use the Generate Discussion tab to create discussions on various topics.
   - You can process multiple PDFs and generate multiple discussions in a single session.

3. View the results:
   - Processed PDF data: Check the `processed_pdfs` directory for JSON files containing extracted and processed text from the PDFs.
   - Generated discussions: Look in the `discussion_results` directory for JSON files containing the AI-generated discussions.

Note: Ensure you have processed at least one PDF before generating a discussion, as the discussion generation relies on the processed data.

## How it Works

1. **PDF Processing**: 
   - Extracts text from uploaded PDF files.
   - Splits the text into manageable chunks.
   - Generates embeddings for each chunk using OpenAI's embedding model.
   - Saves the processed data (text chunks and embeddings) as JSON files.

2. **Discussion Generation**:
   - Uses two AI models, each employing a different search method:
     - Model 1: Vector Search
     - Model 2: Keyword Search
   - For each round of discussion:
     - Generates search queries based on the discussion topic.
     - Retrieves relevant information using respective search methods.
     - Generates responses using GPT models, considering the search results and previous conversation.
   - Saves the entire discussion, including queries, search results, and AI responses, in a structured JSON format.

## Project Structure

- `main.py`: Entry point of the application, sets up the Gradio interface.
- `pdf_processor.py`: Handles PDF text extraction and processing.
- `discussion_generator.py`: Contains the DiscussionGenerator class for generating AI discussions.
- `processed_pdfs/`: Directory for storing processed PDF data.
- `discussion_results/`: Directory for storing generated discussions.

## Customization

You can modify the following files to customize the application's behavior:

- `pdf_processor.py`: Adjust PDF processing parameters (e.g., chunk size, embedding model).
- `discussion_generator.py`: Modify search algorithms, AI model parameters, or discussion generation logic.
- `main.py`: Customize the Gradio interface or add new features.

## Troubleshooting

- If you encounter issues with PDF processing, ensure you have the necessary dependencies installed for PyPDF2.
- For OpenAI API errors, check your API key and internet connection.
- If discussions are not generating properly, verify that you have processed PDFs available in the `processed_pdfs` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT and embedding models
- Gradio for the easy-to-use interface framework
- The creators and maintainers of PyPDF2, sklearn, and other libraries used in this project

