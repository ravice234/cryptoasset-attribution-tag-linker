# Cryptoasset Attribution Tag Linker

## Description

This repository contains the code for the paper: "Linking Cryptoasset Attribution Tags to Knowledge Graph Entities: An LLM-based Approach"

## Installation

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```

2. For access to OpenAI models, add a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY_ATL=YOUR_KEY
   ```

3. For access to remote models, download models from sources like Hugging Face.

4. Configure models in `config.yaml`.

## Usage

### Reproducing the Paper

Run experiments 1, 2 & 3 in the `experiment` folder.

### End-to-End Linking

```
python3 e2e.py -f $file -ft $filter -b $blocker -k $k -m $model -t $t -s $s
```

### Candidate Set Generation

```
python3 cg.py -f $file -ft $filter -b $blocker -k $k
```

### Entity Linking

```
python3 el.py -f $file -m $model -t $t -s $s
```

### Example

```
python3 e2e.py -f "data/gs_test.csv" -ft "related-concept" -b "bm25_3" -k 5 -m "Jellyfish-7B" -t 7 -s 5
```

### Parameters

- `-f`: File Path (CSV format. See examples in data folder)
- `-ft`: Filter (Valid choices: "None", "same-concept", "related-concept")
- `-b`: Blocker (Valid choices: "overlap_3", "bm25_3")
- `-k`: Candidate Set Size (e.g., 5)
- `-m`: Large Language Model (Can be configured in config.yaml, 8 models preconfigured, new models can be simply added via config)
- `-t`: Prompt Template (Templates are in attribution_tag_linker/src/templates.py)
- `-s`: Shots (Number of few-shot examples)

## Citation

Please consider citing our paper:

```
@misc{avice2025linkingcryptoassetattributiontags,
      title={Linking Cryptoasset Attribution Tags to Knowledge Graph Entities: An LLM-based Approach}, 
      author={Régnier Avice and Bernhard Haslhofer and Zhidong Li and Jianlong Zhou},
      year={2025},
      eprint={2502.10453},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2502.10453}, 
}
```
