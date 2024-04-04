#!/bin/bash

# Define the paths to folders A and B
abstract_folder="../../datasets/2018_dump_wiki_cache/abstracts_tokenized"
title_folder="../../datasets/2018_dump_wiki_cache/titles_tokenized"
sub_obj_folder="../../datasets/2018_dump_wiki_cache/subject_object_tokenized"
result_folder="../../datasets/2018_dump_wiki_cache/ggrep_result"

# Iterate through subdirectories in folder A
for subdirectoryA in "$sub_obj_folder"/*; do
    # Extract the subdirectory name
    subdirname=$(basename "$subdirectoryA")
    targetdir="$result_folder/$subdirname"
    
    if [ ! -d "$targetdir" ]; then
        mkdir "$targetdir"
    fi
    
    abstractSubdirectory="$abstract_folder/$subdirname"

    srcSubKeywords="$subdirectoryA"/subject.text
    srcObjKeywords="$subdirectoryA"/object.text
    abstractDocuments="$abstractSubdirectory"/all_abstracts.txt
    
    subjectMatchFile="$targetdir"/subject_matches.txt
    objectMatchFile="$targetdir"/object_matches.txt
    echo "Matching for language "$subdirname
    if [ -d "$abstractSubdirectory" ]; then
        ./ggrep $srcSubKeywords < $abstractDocuments > $subjectMatchFile
        ./ggrep $srcObjKeywords < $abstractDocuments > $objectMatchFile
    fi
done