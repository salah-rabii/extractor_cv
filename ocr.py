import os
import base64
import json
from pathlib import Path
from google import genai
from dotenv import load_dotenv

load_dotenv()

class CVExtractor:
    def __init__(self, input_folder="cv_inp", output_folder="cv_out", prompt_file="prompt.txt"):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.prompt_file = prompt_file
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)
        
        # Create folders if they don't exist
        os.makedirs(self.input_folder, exist_ok=True)
        os.makedirs(self.output_folder, exist_ok=True)
    
    def load_prompt(self) -> str:
        """Load prompt from external text file"""
        if not os.path.exists(self.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file}")
        
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    
    def get_mime_type(self, file_path: str) -> str:
        """Determine MIME type based on file extension"""
        ext = Path(file_path).suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(ext, "image/jpeg")
    
    def extract_cv_info(self, file_path: str) -> dict:
        """Extract CV information using Gemini API"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        mime_type = self.get_mime_type(file_path)
        
        # Read and encode file
        with open(file_path, "rb") as f:
            file_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        # Load prompt from file
        prompt = self.load_prompt()
        
        # Call Gemini API with proper format
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": file_data
                            }
                        }
                    ]
                }
            ]
        )
        
        # Parse response
        response_text = response.text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        cv_data = json.loads(response_text.strip())
        return cv_data
    
    def process_all_cvs(self):
        """Process all CV files in the input folder"""
        cv_files = [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
        
        if not cv_files:
            print(f"No files found in {self.input_folder} folder")
        else:
            print(f"Found {len(cv_files)} file(s) to process...\n")
            
            for cv_filename in cv_files:
                cv_file = os.path.join(self.input_folder, cv_filename)
                output_filename = Path(cv_filename).stem + "_extracted.json"
                output_file = os.path.join(self.output_folder, output_filename)
                
                print(f"Processing: {cv_filename}")
                
                try:
                    cv_info = self.extract_cv_info(cv_file)
                    
                    # Save to output folder
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(cv_info, f, indent=2, ensure_ascii=False)
                    print(f"✓ Saved to: {output_file}\n")
                
                except Exception as e:
                    print(f"✗ Error: {e}\n")

if __name__ == "__main__":
    extractor = CVExtractor()
    extractor.process_all_cvs()