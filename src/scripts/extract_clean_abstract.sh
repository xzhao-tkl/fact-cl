#!/bin/bash

# Define the directory containing the XML files
indir_root="/disk/xzhao/datasets/wikipedia_2018_octnov/abstract"
outdir_root="/disk/xzhao/datasets/wikipedia_2018_octnov/clean_abstract"


folders=($(find "$indir_root"/* -type d))

process_xml() {
    local file="$1"
    local outdir="$2"
    local outfn="$outdir/clean_abstract.txt"

    grep -ho '<abstract>.*</abstract>' "$file" | sed 's/<abstract>\(.*\)<\/abstract>/\1/' > "$outfn"
}

for folder in "${folders[@]}"; do
    last_folder=$(basename "$folder")
    # echo ${last_folder}
    indir="$indir_root/$last_folder"
    outdir="$outdir_root/$last_folder"

    if [ ! -d "$outdir" ]; then
        mkdir "$outdir"
    fi

    # Loop through each XML file in the directory
    for file in "$indir"/*.xml; do
        # Extract the abstract text and save it to a new file
        process_xml "$file" "$outdir" &
    done

    # wait
done