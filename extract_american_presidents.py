import os
import re

def remove_XML_tags(text):
    """Remove XML tags from a string"""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

directory_name = "Corpus_of_Presential_Speeches"
directory = os.fsencode(directory_name)

for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".txt"):
         # print(os.path.join(directory, filename))

         continue
     else:
         continue
