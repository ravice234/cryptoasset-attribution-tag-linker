#!/bin/bash
cd ..

fn="data/candidate_sets/gs_validation.json"

#
# Remote Models
#
set -a 
source .env
set +a
rmodels=("gpt-4o-2024-05-13" "gpt-3.5-turbo-0125")

# Zero-shot for each model and each template
for model in "${rmodels[@]}"; do
  for t in {0..9}; do
      python3 el.py -f $fn -m "$model" -t $t -s 0
  done
done

# Five-shot for each model and each template
for model in "${rmodels[@]}"; do
  for t in {0..9}; do
      python3 el.py -f $fn -m "$model" -t $t -s 5
  done
done

#
# Local Models
#
lmodels=("Jellyfish-7B"
         "Jellyfish-13B-awq" 
         "Mistral-7B-Instruct-v0.3"
         "Mistral-7B-v0.3" 
         "Meta-Llama-3-8B-Instruct" 
         "Meta-Llama-3-8B")

# Zero-shot for each model and each template 5 times
for model in "${lmodels[@]}"; do
  for t in {0..9}; do
    for i in {1..5}; do
      python3 el.py -f $fn -m "$model" -t $t -s 0
    done
  done
done

# Five-shot for each model and each template 5 times
for model in "${lmodels[@]}"; do
  for t in {0..9}; do
    for i in {1..5}; do
      python3 el.py -f $fn -m "$model" -t $t -s 5
    done
  done
done

systemctl poweroff