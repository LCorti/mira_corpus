#!/usr/bin/env python3
"""
Extract titles and abstracts from the text field in selected_papers.jsonl.
Creates separate title and abstract fields while keeping them in the text field.
Titles always start with "Title:" and abstracts start with "ABSTRACT"
"""

import json
import re

def extract_title(text):
    """
    Extract the title from the text field.
    The title starts with "Title:" and ends before "ABSTRACT" or "SECTION 1:"
    """
    # Find the title using regex
    # Match "Title:" followed by any text until we hit "ABSTRACT" or "SECTION 1:"
    match = re.search(r'Title:\s*(.*?)\s*(?:ABSTRACT|SECTION 1:)', text, re.DOTALL)
    
    if match:
        title = match.group(1).strip()
        # Remove any extra whitespace and newlines
        title = ' '.join(title.split())
        return title
    return None

def extract_abstract(text):
    """
    Extract the abstract from the text field.
    The abstract starts with "ABSTRACT" and ends before "SECTION 1:"
    """
    # Find the abstract using regex
    # Match "ABSTRACT" followed by any text until we hit "SECTION 1:"
    match = re.search(r'ABSTRACT\s*(.*?)\s*SECTION 1:', text, re.DOTALL)
    
    if match:
        abstract = match.group(1).strip()
        # Remove any extra whitespace but preserve paragraph breaks
        # Replace multiple spaces/newlines with single space, but keep intentional breaks
        abstract = re.sub(r'[ \t]+', ' ', abstract)
        abstract = re.sub(r'\n+', '\n', abstract)
        return abstract.strip()
    return None

def process_file(input_file, output_file):
    """
    Process the JSONL file to extract titles and abstracts and create new fields.
    Keeps titles and abstracts in the text field as well.
    """
    processed_count = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
        open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Extract title from text
                title = extract_title(data['text'])
                
                if title:
                    # Add title field
                    data['title'] = title
                    processed_count += 1
                else:
                    print(f"Warning: Could not extract title from line {line_num} (paper_id: {data.get('paper_id', 'unknown')})")
                    error_count += 1
                
                # Extract abstract from text
                abstract = extract_abstract(data['text'])
                
                if abstract:
                    # Add abstract field (keep it in text field too)
                    data['abstract'] = abstract
                
                # Write the modified data (text field remains unchanged)
                outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_num}: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                error_count += 1
    
    print(f"\nProcessing complete:")
    print(f"  Successfully processed: {processed_count}")
    print(f"  Errors: {error_count}")

if __name__ == '__main__':
    input_file = 'data_scisumm/corpus.jsonl'
    
    # Process in place by using a temporary file
    temp_file = 'data_scisumm/corpus_temp.jsonl'
    
    process_file(input_file, temp_file)
    
    # Replace original with temp file
    import shutil
    shutil.move(temp_file, input_file)
    
    print(f"\nFile updated: {input_file}")
