from djl_python import Input, Output
from djl_python.streaming_utils import StreamingUtils
import os
import deepspeed
import torch
import logging
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoModel
from torch import autocast

model = None
tokenizer = None

'''
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

model_path = "LinkSoul/Chinese-Llama-2-7b"

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path,cache_dir='/data').half().cuda()
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

prompt = instruction.format("用英文回答，什么是夫妻肺片？")
generate_ids = model.generate(tokenizer(prompt, return_tensors='pt').input_ids.cuda(), max_new_tokens=4096, streamer=streamer)
'''


    
def preprocess(text):
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text

def postprocess(text):
    return text.replace("\\n", "\n").replace("\\t", "\t")

def answer(prompt, temperature, model):
    inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
    inputs1 = {'input_ids':inputs['input_ids']}
    tokens = model.generate(
     **inputs1,
     max_new_tokens=512,
     do_sample=True,
     temperature=temperature,
     top_p=1.0,
    )
    answer = tokenizer.decode(tokens[0], skip_special_tokens=True)
    # print(answer)
    return answer


def inference(inputs):
    try:
        input_map = inputs.get_as_json()
        data = input_map.pop("ask", input_map)
        # history = input_map.pop("history", [])
        temperature = 0.1 #input_map.pop("temperature", {})
        outputs = Output()
        ##
        instruction = """[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

            If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n{} [/INST]"""

        prompt = instruction.format(data)
        print('prompt',prompt)
        response = answer(prompt, temperature, model)
        outputs.add_as_json({"answer": response})
        return outputs
    
    except Exception as e:
        logging.exception("Inference failed")
        # error handling
        outputs = Output().error(str(e))


def get_model(properties):
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    tensor_parallel_degree = properties["tensor_parallel_degree"]
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_location, trust_remote_code=True).half().cuda()
    model = model.eval()
    return model, tokenizer


def handle(inputs: Input) -> None:
    global model, tokenizer
    if not model:
        model, tokenizer = get_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    return inference(inputs)