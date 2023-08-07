from djl_python import Input, Output
import os
import logging
import torch
import deepspeed
import transformers
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
# from transformers.models.llama.tokenization_llama import LlamaTokenizer

predictor = None
#here, we need to set the global variable batch_size according to the batch_size in the serving.properties file.
batch_size = 8

def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True,torch_dtype=torch.float16)

    #for deepspeed inference 
    model = AutoModelForCausalLM.from_pretrained(model_location, trust_remote_code=True).half().cuda()

#     print("----------model dtype is {0}---------".format(model.dtype))
#     model = deepspeed.init_inference(
#         model,
#         mp_size=tensor_parallel,
#         dtype=torch.half,
#         replace_method="auto",
#         replace_with_kernel_inject=True,
#     )
        
#     local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, use_cache=True, device=local_rank)
    
    
    #for HF accelerate inference
    '''
    model = AutoModelForCausalLM.from_pretrained(model_location, device_map="auto", torch_dtype=torch.float16)
    print("----------model dtype is {0}---------".format(model.dtype))
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer, use_cache=True)
    '''
    
    #for llama model, maybe the followiong code is need when you invoke the pipleline API for batch input prompts.
    # generator.tokenizer.pad_token_id = model.config.eos_token_id
    return model, tokenizer


def handle(inputs: Input) -> None:
    global model, tokenizer
    try:
        model,tokenizer = load_model(inputs.get_properties())

        #print(inputs)
        if inputs.is_empty():
            # Model server makes an empty call to warmup the model on startup
            return None
        
        if inputs.is_batch():
            #the demo code is just suitable for single sample per client request
            bs = inputs.get_batch_size()
            logging.info(f"Dynamic batching size: {bs}.")
            batch = inputs.get_batches()
            #print(batch)
            tmp_inputs = []
            for _, item in enumerate(batch):
                tmp_item = item.get_as_json()
                tmp_inputs.append(tmp_item.get("input"))
            
            #For server side batch, we just use the custom generation parameters for single Sagemaker Endpoint.
            #result = predictor(tmp_inputs, batch_size = bs, max_new_tokens = 128, min_new_tokens = 128, temperature = 1.0, do_sample = True)
            result, history = model.chat(tokenizer, tmp_inputs, history=[])
            print(result)
            
            outputs = Output()
            for i in range(len(result)):
                outputs.add(result[i], key="generate_text", batch_index=i)
            return outputs
        else:
            inputs = inputs.get_as_json()
            if not inputs.get("input"):
                return Output().add_as_json({"code":-1,"msg":"input field can't be null"})

            #input data
            data = inputs.get("input")
            params = inputs.get("params",{})

            #for pure client side batch
            if type(data) == str:
                bs = 1
            elif type(data) == list:
                if len(data) > batch_size:
                    bs = batch_size
                else:
                    bs = len(data)
            else:
                return Output().add_as_json({"code":-1,"msg": "input has wrong type"})
                
            print("client side batch size is ", bs)
            #predictor
            result = predictor(data, batch_size = bs, **params)

            #return
            return Output().add({"code":0,"msg":"ok","data":result})
    except Exception as e:
        return Output().add_as_json({"code":-1,"msg":e})
    
    
