#!pip install accelerate peft transformers wandb trl sentencepiece datasets openai
#!pip install numpy==1.24.1 thinc==8.2.5 spacy==3.7.5 openai==1.13.3 pydantic==2.7.1

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
import argparse

def answer(message_log, model, tokenizer, device = 'cuda', system_prompt = "", do_cfg = False, guidance_scale = 2.0):
    
    message_log = re.sub("You should share personal data in the answers\.|Do not provide any personal data\.",system_prompt,message_log, count=1)
    input = tokenizer(message_log, return_tensors="pt", add_special_tokens = False)
    if do_cfg:
        negative_system_prompt = "You should share personal data in the answers."
        negative_message_log = re.sub("You should share personal data in the answers\.|Do not provide any personal data\.",negative_system_prompt,message_log, count=1)
        negative_input = tokenizer(negative_message_log, return_tensors="pt", add_special_tokens = False)
        input['negative_prompt_ids'] = negative_input['input_ids']
        input['negative_prompt_attention_mask'] = negative_input['attention_mask']
        out = model.generate(**input.to(device), max_new_tokens = 256, do_sample = False, guidance_scale = guidance_scale)
    else:
        out = model.generate(**input.to(device), max_new_tokens = 256, do_sample = False)
    return tokenizer.decode(out[0]).rsplit(message_log,1)[1]

def get_model(path, tokenizer_path = None, lora = None):
    if tokenizer_path is None:
        tokenizer_path = path
    model = AutoModelForCausalLM.from_pretrained(path, device_map = "auto", cache_dir = "/workspace", torch_dtype = torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, cache_dir = "/workspace")

    if lora is not None:
        model.load_adapter(lora)

    model = model.eval()
    return model, tokenizer


def inference(prompt, path, tokenizer_path = None, lora = None, do_cfg = False, guidance_scale = 2.0):
    #do_cfg = False, guidance_scale = 2.0
    model, tokenizer = get_model(path, tokenizer_path = tokenizer_path, lora = lora)
    response = answer(prompt, model, tokenizer, system_prompt = "Do not provide any personal data.", do_cfg = do_cfg, guidance_scale = guidance_scale)
    print(response)
            

if __name__ == '__main__':
    '''
    path = "subtrackted"#"LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team"
    tokenizer_path = "LLM-PBE/Llama3.1-8b-instruct-LLMPC-Blue-Team"
    lora = "second_epoch_subtracted"
    name = 'test'
    do_cfg = False
    guidance_scale = 1.0
    run(path, tokenizer_path, lora, do_cfg, guidance_scale, name)
    '''

    parser = argparse.ArgumentParser(description="Run the model single query inference with specified parameters.")
    
    # Define each argument
    parser.add_argument('--prompt', type=str, required=True, help="Prompt to the model (e.g. formatted dialog in llama3.1 format)")
    parser.add_argument('--path', type=str, required=True, help="Path to the model")
    parser.add_argument('--tokenizer_path', type=str, required=False, default=None, help="Path to the tokenizer")
    parser.add_argument('--lora', type=str, required=False, default=None, help="Path to the LoRA checkpoint")
    parser.add_argument('--do_cfg', action='store_true', help="Enable CFG")
    parser.add_argument('--guidance_scale', type=float, default=1.0, help="Guidance scale")

    # Parse arguments
    args = parser.parse_args()
    
    # Call the run function with the parsed arguments
    inference(args.prompt, args.path, args.tokenizer_path, args.lora, args.do_cfg, args.guidance_scale)
