#!/bin/bash
cd ..
fn="data/gs_validation.csv"
filter=("None" "same_concept" "related_concept")
blocker=("overlap_3" "bm25_3")
k=(1 5 10 25)

for f in "${filter[@]}"; do
  for b in "${blocker[@]}"; do
    for k in "${k[@]}"; do
      python3 cg.py -f $fn -ft "$f" -b "$b" -k $k
    done
  done
done

# systemctl poweroff