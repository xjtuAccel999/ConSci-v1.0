import re
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python3 split_modules.py <filename>")
    sys.exit(1)

filename = sys.argv[1]
directory = os.path.dirname(filename)

with open(filename, 'r') as file:
    content = file.read()

modules = re.findall(r"module\s+(\w+)\s*\(.*?endmodule", content, re.DOTALL)

for module_name in modules:
    module_content = re.search(r"(module\s+" + module_name + r"\s*\(.*?endmodule)", content, re.DOTALL)
    if module_content:
        output_file = os.path.join(directory, module_name + ".v")
        with open(output_file, 'w') as file:
            file.write(module_content.group(1))
