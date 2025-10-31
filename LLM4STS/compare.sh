#!/bin/bash
# Based on https://unix.stackexchange.com/questions/397655/two-files-comparison-in-bash-script
# sh compare.sh "../Dataset/Log1/testApache" "../Results/Log1/testApache/gpt2_256_1_1/decompress"
# sh compare.sh "../Results1/1Cyber-Vehicle/syndata_vehicle0/Llama_256_1_1/decompress" "../Dataset/Cyber-Vehicle/syndata_vehicle0"



file1=$1
file2=$2

if cmp -s $file1 $file2; then
    # printf 'The file "%s" is the same as "%s"\n' "$file1" "$file2"
    printf 'Same'
else
    # printf 'The file "%s" is different from "%s"\n' "$file1" "$file2"
    printf 'Error'
fi
