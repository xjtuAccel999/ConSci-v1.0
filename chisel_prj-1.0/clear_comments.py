import os
import re
import sys

def remove_c_comments(file_path):

    if not os.path.isfile(file_path):
        print(f"The file {file_path} does not exist.")
        return

    with open(file_path, 'r') as file:
        content = file.read()

        # delete（/* ... */）
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

        # delete（// ...）
        content = re.sub(r'//.*', '', content)

    with open(file_path, 'w') as file:
        file.write(content)

    # print(f"Comments removed from {file_path}")

if len(sys.argv) != 2:
    print("Usage: python3 remove_comments.py <filename>")
    sys.exit(1)

filename = sys.argv[1]

remove_c_comments(filename)
