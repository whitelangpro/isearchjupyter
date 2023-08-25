from djl_python import Input, Output
import torch
import logging
import math
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_location, trust_remote_code=True)
    model = model.eval().half().cuda()
    
    return model, tokenizer


model = None
tokenizer = None
generator = None


def handle(inputs: Input):
    global model, tokenizer
    if not model:
        model, tokenizer = load_model(inputs.get_properties())

    if inputs.is_empty():
        return None
    data = inputs.get_as_json()
    
    prompt = data["ask"]
    parameters = {'temperature':data["temperature"]

}
    # history = data["history"]

    response, _ = model.chat(tokenizer, prompt, history=[],**parameters) #history=history, 
    
    result = {"answer": response} #, "history" : history
    return Output().add_as_json(result)
