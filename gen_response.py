import torch
import config
from data_utils import *
from model_init import *
from transformers import GenerationConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_response(model, tokenizer, prompt, use_vllm=False, lora_request=None):
    if use_vllm:
        from vllm import SamplingParams
        inputs = tokenizer(text=prompt,
                           max_length=4096,
                           padding='max_length', 
                           truncation=True, 
                           return_tensors="pt")
        if lora_request != None:
            sampling_params = SamplingParams(temperature=0,
                                             max_tokens=1024,
                                            )
            response = model.generate(prompt, 
                                      lora_request=lora_request, 
                                      sampling_params=sampling_params
                                    )
        else:
            # Base models don't need to generate exemplars; we can lower the max_tokens for speed
            sampling_params = SamplingParams(temperature=0,
                                             max_tokens=250,
                                            )
            response = model.generate(prompt,
                                      sampling_params=sampling_params)
    elif not use_vllm:
        inputs = tokenizer(text=prompt, 
                           max_length=4096,
                           padding="max_length", 
                           truncation=True, 
                           return_tensors="pt")                         
        generate_ids = model.generate(inputs.input_ids.to(device), 
                                  max_new_tokens=config.max_new_tokens,
                                  generation_config=GenerationConfig(
                                        temperature=config.temperature,
                                        do_sample=True,        # this will use greedy decoding and will be deterministic (no need to set temp)
                                        top_k=config.top_k,
                                        seed=config.seed
                                        )
                                  )
        response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response