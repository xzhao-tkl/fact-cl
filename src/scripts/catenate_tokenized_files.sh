#!/bin/bash

# Set the path to the top-level directory containing subdirectories
abstract_folder="../../datasets/2018_dump_wiki_cache/abstracts_tokenized"
title_folder="../../datasets/2018_dump_wiki_cache/titles_tokenized"

# Loop through subdirectories
for lang_folder in "$abstract_folder"/*; do
    if [ -d "$lang_folder" ]; then
        # Initialize a variable to store concatenated file name
        concatenated_file="$lang_folder/all_abstracts.txt"

        # Use find to locate all files in the lang_folder and concatenate them
        find "$lang_folder" -type f -exec cat {} + > "$concatenated_file"
        echo "Concatenated files in $lang_folder to $concatenated_file"
    fi
done


# # Loop through subdirectories
# for lang_folder in "$title_folder"/*; do
#     if [ -d "$lang_folder" ]; then
#         # Initialize a variable to store concatenated file name
#         concatenated_file="$lang_folder/all_titles.txt"
#         # Use find to locate all files in the lang_folder and concatenate them
#         find "$lang_folder" -type f -exec cat {} + > "$concatenated_file"
#         echo "Concatenated files in $lang_folder to $concatenated_file"
#     fi
# done