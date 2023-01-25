import json
import os
import re

if __name__ == '__main__':
    all_characters = []
    for dirname in os.listdir("data"):
        this_directory_character = None
        filename = os.listdir(os.path.join("data", dirname))[0]
        found_character = re.search(r'(.)_[0-9]*.png', filename).group(1)
        all_characters.append(found_character)
    mapping = {all_characters[i]: i for i in range(len(all_characters))}
    dump = json.dumps(mapping, indent=4)
    with open('label_to_int.json', 'w') as file:
        file.write(dump)
    characters_dump = json.dumps(all_characters, indent=4)
    with open('all_characters.json', 'w') as file:
        file.write(characters_dump)
