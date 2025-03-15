from utils import run_inference


response = run_inference("Hi!")
print("response:", response)

from utils import extract_xml
extracted_xml_data = extract_xml(response)
