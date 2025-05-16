import os

# Specify the directory containing the txt files
directory = '/path/to/directory'

# Initialize an empty string to store the combined content
combined_content = ''

# Loop through each file in the directory
for filename in sorted(os.listdir(directory)):
    if filename.endswith('.txt'):
        with open(os.path.join(directory, filename), 'r') as file:
            combined_content += file.read() + '\n'

# Write the combined content to a new file
with open('combined_file.txt', 'w') as combined_file:
    combined_file.write(combined_content)
