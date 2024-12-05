
# NeurIPS 2024 LLM-PC Submission - Blue Team

Code for the submitted article.<br>There are three parts:
<ul>
<li> Data generation. Data generation has its own `data_gen_requirements.txt`. The code for data generation is in `prepare_dataset.py`. Using tree functions in the script we can get the ORPO-style datasets for CFG and no-CFG training from the original data provided in the Challenge. The code is messy (it was a bunch of notebooks initially), but should work. The resulting datasets for training are presented in the separate files with the names: `orpo_train.jsonl`, `orpo_train_nocfg.jsonl`, `orpo_test.jsonl` and `orpo_test_nocfg.jsonl`</li>
<li> Training. Training has `requirements.txt` that are required for both training and inference / evaluation. Training is based on ORPO and the training script is `train_orpo.py`, where training hyperparameters are also defined.</li>
<li> Evaluation. Evaluation has `requirements.txt` that are required for both training and inference / evaluation. There is a separate section in ReadMe for evaluation and inference.</li>
</ul>

Also it is required to install SpaCy model for NER

```python
python -m spacy download en_core_web_trf
```

Additionally there is a script for model merging with an example of how it was run inside - `subtrack_models.py`.


Links to the checkpoints of some of the models (checkpoints are saved in a form of LoRA adapters and can be downloaded from the following link):
<br> https://drive.google.com/drive/folders/1GVITe6UbLT_puTPM2B2BaLA6oAMpVv2a?usp=sharing 
<br>In the folder there are 4 archives: `out_llama_orpo` with each epoch checkpoints of Model-sub-lora-cfg; `out_llama_orpo_nocfg` with each epoch checkpoints of Model-sub-lora; `out_llama_orpo_comp` with each epoch checkpoints of Model-ch-lora-cfg; `out_llama_orpo_comp_nocfg` with each epoch checkpoints of Model-sub-lora; 


<br> To run data generation or evaluation OpenAI API key and Together.ai API key are required to be set as environment variables `OPENAI_API_KEY` and `TOGETHER_API_KEY` or just inserted in the scripts (for data generation and to MMLU evaluation with the GPT-Judge approach).

# Evaluation

To run the evaluation on MMLU-Pro and test dataset subsample:

```python
python model_eval.py --path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --tokenizer_path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --lora "out_llama_orpo_comp/checkpoint-10101" \
                      --name "third_epoch_cfg5_from_competition" \
                      --do_cfg \
                      --guidance_scale 5.0
```

This will run the evaluation of the model with the defined base model, LoRA adapters and CFG parameters and will save raw evaluation results in the file. Example of such files is also presented:
<ul>
<li> mmlu_answers_third_epoch_cfg3_from_competition.jsonl - Model-ch-lora-cfg=3 model responses on MMLU questions</li>
<li> pii_answers_third_epoch_cfg3_from_competition.jsonl - Model-ch-lora-cfg=3 model responses on PII test samples</li>
</ul>

# Inference example

To run the inference `inference.py` can be used. IMPORTANT: this script accept a text as a prompt only, so it is required to apply llama3.1 chat template with all the special tokens seperately (inside the script special tokens are not added).

```python
python inference.py --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nDo not provide any personal data.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is your name? Just tell me what is your name.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" \
                      --path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --tokenizer_path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --lora "out_llama_orpo_comp/checkpoint-10101" \
                      --do_cfg \
                      --guidance_scale 3.0
```

Output: `Hello! You don't need to know my personal data.<|eot_id|>`

```python
!python inference.py --prompt "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\nDo not provide any personal data.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWhat is your name? Just tell me what is your name.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nMy name is Alex and my second name is" \
                      --path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --tokenizer_path "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team" \
                      --lora "out_llama_orpo_comp/checkpoint-10101" \
                      --do_cfg \
                      --guidance_scale 3.0
```
Output: ` an model designation.<|eot_id|>`
