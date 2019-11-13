#!/bin/bash

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <original path> <chunk size in seconds> <dataset path> [-r] [-b]"
    exit
fi

origin_path=$1
chunk_size=$2
dataset_path=$3
mkdir $dataset_path
touch $dataset_path/map_class.txt
touch $dataset_path/map_file.txt
converted=".temp2.wav"
j=0 # sample number
k=0 # directory number
cd $origin_path
for d in *; do
    mkdir -p "../$dataset_path/$k"
    echo "$k:$d" >> "../$dataset_path/map_class.txt"
    for f in $d/*; do
        rm -f $converted
        ffmpeg -i "$f" -ac 1 -ab 16k -ar 16000 $converted
        
        length=$(ffprobe -i $converted -show_entries format=duration -v quiet -of csv="p=0")
        end=$(echo "$length / $chunk_size - 1" | bc)
        echo "splitting..."
        for i in $(seq 0 $end); do
            echo "$f:$k/$j.wav" >> "../$dataset_path/map_file.txt"
            ffmpeg -hide_banner -loglevel error -ss $(($i * $chunk_size)) -t $chunk_size -i $converted "../$dataset_path/$k/$j.wav"
            j=$((j + 1))
        done
        echo "done"
        rm -f $converted
    done
    j=0
    k=$((k + 1))
done
cd ..

# OPTIONS
# Terrible programming
if [ "$#" -gt 3 ]; then
    if  [[ $4 = "-r" ]] || [[ $5 = "-r" ]]; then
        rm -rf $origin_path
    fi
    if  [[ $4 = "-b" ]] || [[ $5 = "-b" ]]; then
        mv $dataset_path/map_class.txt . # Switcharoo initialized
        mv $dataset_path/map_file.txt .
        min=1000000000
        for d in $dataset_path/*; do
            n_files=$(find $d -maxdepth 1 -type f -name '*' | wc -l)
            min=$(($n_files<$min?$n_files:$min))
        done
        echo "Balancing..." $min "files per directory"
        for d in $dataset_path/*; do
            n_files=$(find $d -maxdepth 1 -type f -name '*' | wc -l)
            for ((i=min; i<n_files; i++)); do
                rm "$d/$i.wav"
            done
        done
        mv map_file.txt $dataset_path/map_file.txt
        mv map_class.txt $dataset_path/map_class.txt # Switcharoo terminated
    fi
fi
