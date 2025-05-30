import docx
import pandas as pd
import re
import os

def extract_text_from_docx(file_path):
    """Extract full text from a docx file"""
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():  # Skip empty paragraphs
            full_text.append(para.text.strip())
    return '\n'.join(full_text)

def parse_document_into_records(text):
    """Split the document text into separate records based on 'Citation:' marker"""
    # Split by "Citation:" but keep the delimiter with the following content
    records_raw = re.split(r'(?=Citation:)', text)
    # Remove any empty records (like possibly the first one)
    records = [r for r in records_raw if r.strip()]
    return records

def extract_fields_from_record(record_text):
    """Extract structured fields from a single record"""
    # Define all possible field headers
    headers = [
        "Citation", "Proposed Title", "Date", "Sender", "Sender Place",
        "Receiver", "Receiver Place", "Transcription"
    ]
    
    # Initialize an empty dictionary for the record
    record_data = {header.lower().replace(" ", "_"): "" for header in headers}
    
    # Current field being processed
    current_field = None
    current_content = []
    
    # Process each line
    for line in record_text.split('\n'):
        matched = False
        for header in headers:
            if line.startswith(f"{header}:"):
                # Save the previous field if there was one
                if current_field:
                    record_data[current_field] = '\n'.join(current_content).strip()
                
                # Start a new field
                current_field = header.lower().replace(" ", "_")
                current_content = [line[len(header)+1:].strip()]
                matched = True
                break
        
        if not matched and current_field:
            # Continue with the current field
            current_content.append(line)
    
    # Don't forget to save the last field being processed
    if current_field:
        record_data[current_field] = '\n'.join(current_content).strip()
    
    return record_data

def main():
    # Adjust the file path as needed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use script_dir directly instead of parent_dir since the data folder is in the same directory
    doc_path = os.path.join(script_dir, "data", "SBB_data.docx")
    
    # Extract text from the document
    print(f"Extracting text from {doc_path}...")
    document_text = extract_text_from_docx(doc_path)
    
    # Split into individual records
    print("Parsing records...")
    records = parse_document_into_records(document_text)
    print(f"Found {len(records)} records")
    
    # Extract structured data from each record
    structured_records = []
    for i, record in enumerate(records):
        print(f"Processing record {i+1}/{len(records)}")
        record_data = extract_fields_from_record(record)
        structured_records.append(record_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(structured_records)
    
    # Save to CSV
    output_path = os.path.join(script_dir, "data", "parsed_records.csv")
    df.to_csv(output_path, index=False)
    print(f"Data successfully saved to {output_path}")
    
    # Also save as Parquet for more efficient storage/reading
    parquet_path = os.path.join(script_dir, "data", "parsed_records.parquet")
    df.to_parquet(parquet_path, index=False)
    print(f"Data also saved to {parquet_path}")
    
    return df

if __name__ == "__main__":
    df = main()
    # Display first few records
    print("\nSample of parsed data:")
    print(df.head())