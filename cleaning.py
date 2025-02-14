import json
import re

def validate_json_object(json_string):
    try:
        json_obj = json.loads(json_string)
        return json_obj, None
    except json.JSONDecodeError as e:
        return None, str(e)

def fix_json_object(json_string):
    try:
        # Attempt to fix common errors
        # 1. Remove trailing commas
        json_string = re.sub(r",\s*([\}\]])", r"\1", json_string)
        
        # 2. Add missing brackets (if possible)
        open_brackets = json_string.count("{")
        close_brackets = json_string.count("}")
        if open_brackets > close_brackets:
            json_string += "}" * (open_brackets - close_brackets)
        
        # Validate and parse the fixed JSON
        json_obj = json.loads(json_string)
        return json_obj, None
    except json.JSONDecodeError as e:
        return None, str(e)

def process_json_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    
    # Split into potential JSON objects
    json_objects = re.split(r'(?<=\})(\s*[\r\n]+)', data)
    
    fixed_objects = []
    for idx, json_str in enumerate(json_objects):
        if not json_str.strip():
            continue
        print(f"Processing JSON object {idx + 1}...")
        valid_json, error = validate_json_object(json_str)
        if valid_json:
            fixed_objects.append(valid_json)
        else:
            print(f"Error: {error}. Attempting to fix...")
            fixed_json, fix_error = fix_json_object(json_str)
            if fixed_json:
                fixed_objects.append(fixed_json)
                print(f"Object {idx + 1} fixed successfully.")
            else:
                print(f"Failed to fix object {idx + 1}: {fix_error}")
    
    # Save fixed JSON objects
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump(fixed_objects, output_file, indent=4, ensure_ascii=False)
    print(f"Fixed JSON saved to {output_path}")

if __name__ == "__main__":
    # Replace 'input.json' with your actual input file path
    # Replace 'output.json' with your desired output file path
    input_file = "/Users/shubhamfufal/Chatbot(v2)/MentalHealthChatbot-v2-/mental_health_counseling_conversations.json"
    output_file = "output.json"
    
    process_json_file(input_file, output_file)
