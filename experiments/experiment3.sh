#!/bin/bash
cd ..

declare -A files=(
  ["data/gs_test.csv"]="related_concept"
  ["data/rekt.csv"]="None"
  ["data/watchyourback.csv"]="None")       

#
# Remote Models
#
set -a 
source .env
set +a

declare -A rmodels_t=(
  ["gpt-4o-2024-05-13"]=7
  ["gpt-3.5-turbo-0125"]=9)

# Five-shot for each model and each file
for file in "${!files[@]}"; do
  filter=${files[$file]}
  for model in "${!rmodels_t[@]}"; do
    t=${rmodels_t[$model]}
    python3 e2e.py -f "$file" -ft "$filter" -b "bm25_3" -k 5 -m "$model" -t $t -s 5
  done
done

#
# Local Models
#
declare -A lmodels_t=(
  ["Jellyfish-7B"]=3
  ["Jellyfish-13B-awq"]=2
  ["Mistral-7B-Instruct-v0.3"]=0
  ["Mistral-7B-v0.3"]=8
  ["Meta-Llama-3-8B-Instruct"]=3
  ["Meta-Llama-3-8B"]=0
)

# Five-shot for each model and each file
for file in "${!files[@]}"; do
  filter=${files[$file]}
  for model in "${!lmodels_t[@]}"; do
    t=${lmodels_t[$model]}
    python3 e2e.py -f "$file" -ft "$filter" -b "bm25_3" -k 5 -m "$model" -t $t -s 5
  done
done

#
# Baseline Models
#
for file in "${!files[@]}"; do
  filter=${files[$file]}
  python3 e2e.py -f $file -ft "$filter" -b "bm25_3" -k 5 -m "bm25_3" -t 0 -s 0
  python3 e2e.py -f $file -ft "$filter" -b "bm25_3" -k 5 -m "UnicornPlus" -t 0 -s 0
  python3 e2e.py -f $file -ft "$filter" -b "bm25_3" -k 5 -m "UnicornPlusFT" -t 0 -s 0
done

# systemctl poweroff