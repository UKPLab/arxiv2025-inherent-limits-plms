import config
from huggingface_hub import login
from peft import LoraConfig
import torch
from transformers import (
                          AutoModelForCausalLM, 
                          AutoTokenizer, 
                          BitsAndBytesConfig
                          )
from vllm import LLM
from vllm.lora.request import LoRARequest

login(token=config.hf_key)

def init_model(model_id, 
               do_train, 
               saved_adapter=None, 
               use_vllm=False
               ):
    print("INIT ID:", model_id)
    print("USE VLLM:", use_vllm)
    bnbconfig = BitsAndBytesConfig(
                                    load_in_8bit=False,  # 8-bit is used for <= 13B models
                                    load_in_4bit=True    # 4-bit is used for 70B models
                                  )
    if saved_adapter == None:
        tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                  use_fast=True
                                                  )
    else:
        tokenizer = AutoTokenizer.from_pretrained(saved_adapter, 
                                                  use_fast=True
                                                  )
    print("LOADING TOKENIZER:", tokenizer)

    if do_train:
    # We don't use VLLM during training
        if "mistral" in model_id:
            base_model = AutoModelForCausalLM.from_pretrained(
                                                        model_id, 
                                                        device_map="auto", 
                                                        quantization_config=bnbconfig
                                                        )
            if saved_adapter == None:
                tokenizer.add_special_tokens({
                        'pad_token': '<PAD>',
                        })
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side="right"

        elif "llama" in model_id:
            base_model = AutoModelForCausalLM.from_pretrained(
                                                          model_id, 
                                                          device_map="auto", 
                                                          quantization_config=bnbconfig
                                                          )
            if saved_adapter == None:
                tokenizer.add_special_tokens({
                        'pad_token': '<PAD>',
                        })
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side="right"
            base_model.resize_token_embeddings(len(tokenizer))
            base_model.config.pad_token_id = tokenizer.pad_token_id
            base_model.generation_config.pad_token_id = tokenizer.pad_token_id

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            #modules_to_save= ["embed_tokens", "lm_head"]
        )
        base_model.add_adapter(peft_config)
        return base_model, tokenizer, peft_config, None
    
    elif not do_train and saved_adapter != None:
        # Only load an adapter if not testing the base model.
        if not use_vllm:
            base_model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                          device_map="auto", 
                                                          load_in_8bit=False,   # 8-bit is used for <= 13B models
                                                          load_in_4bit=True     # 4-bit is used for 70B models
                                                          )
            model = PeftModel.from_pretrained(base_model, 
                                              saved_adapter, 
                                              is_trainable=False
                                              )
            model.eval()
            lora_request = None
        elif use_vllm:

            model = LLM(model=model_id,
                        tokenizer=config.path + "/saved_models/" + config.args.run_name,
                        enable_lora=True,
                        enforce_eager=True,
                        max_lora_rank=64,
                        dtype=torch.bfloat16,
                        quantization="bitsandbytes", 
                        load_format="bitsandbytes",
                        )
            lora_request=LoRARequest("trained_adapter", 1, saved_adapter)

        return model, tokenizer, None, lora_request
    
    elif not do_train and saved_adapter == None:
        # Load a base model for testing, e.g. when from_exemplar_gen
        if use_vllm:
            model = LLM(model=model_id, 
                        enforce_eager=True
                        )
        elif not use_vllm:
            model = AutoModelForCausalLM.from_pretrained(
                                                          model_id, 
                                                          device_map="auto",
                                                          quantization_config=bnbconfig,
                                                          )
        tokenizer.add_special_tokens({
                        'pad_token': '<PAD>',
                        })
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side="right"
        return model, tokenizer, None, None