from utils import run_inference, extract_xml

response = run_inference("Hi!")
print("response:", response)

extracted_xml_data = extract_xml(response)
print("extracted xml:", extracted_xml_data)
