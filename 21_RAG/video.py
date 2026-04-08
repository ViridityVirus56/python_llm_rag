import os
import sys
import time
import argparse
from dotenv import load_dotenv
from google import genai
from fpdf import FPDF

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("Error: Could not find GOOGLE_API_KEY or GEMINI_API_KEY in .env file.")
    sys.exit(1)

client = genai.Client(api_key=API_KEY)

def generate_pdf(text, filename="consti.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    # Ensure compatible encoding
    safe_text = text.encode('windows-1252', 'replace').decode('windows-1252')
    pdf.multi_cell(0, 5, text=safe_text)
    pdf.output(filename)
    print(f"Output successfully written to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Extract video information and transcript using Google Gemini.")
    parser.add_argument("file_path", help="Path to the video file")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File '{args.file_path}' not found.")
        sys.exit(1)

    print(f"Uploading {args.file_path} to Gemini...")
    video_file = client.files.upload(file=args.file_path)
    
    print("Waiting for video processing", end='')
    while str(video_file.state).upper() == 'PROCESSING':
        print('.', end='', flush=True)
        time.sleep(5)
        video_file = client.files.get(name=video_file.name)
        
    if str(video_file.state).upper() == 'FAILED':
        print("\nError: Video processing failed.")
        sys.exit(1)

    print("\nGenerating video information...")
    prompt = "Provide all information of the video along with the audio transcript and any information on screen"
    
    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=[video_file, prompt]
    )
    
    print("Video information generated successfully! Saving to PDF...")
    generate_pdf(response.text, "consti.pdf")
    
    print("Cleaning up file from Gemini servers...")
    client.files.delete(name=video_file.name)

if __name__ == "__main__":
    main()
