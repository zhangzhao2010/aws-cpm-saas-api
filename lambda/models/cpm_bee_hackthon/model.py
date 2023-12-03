from djl_python import Input, Output
import os
import logging
import torch
from collections import OrderedDict
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import dispatch_model
from accelerate.utils import get_balanced_memory, infer_auto_device_map

model = None
tokenizer = None

def load_delta(delta_path):
    delta_dict = torch.load(delta_path)
    delta_with_prefix = OrderedDict()
    for k, v in delta_dict.items():
        # CpmBeeModel -> CpmBeeForCasualLM
        if k.startswith("encoder.") or k.startswith("input_embedding.") or k.startswith("position_bias."):
            delta_with_prefix["cpmbee."+k] = v
        else:
            delta_with_prefix[k] = v
    del delta_dict
    return delta_with_prefix

def load_model(properties):
    tensor_parallel = properties["tensor_parallel_degree"]
    model_location = properties['model_dir']
    if "model_id" in properties:
        model_location = properties['model_id']
    logging.info(f"Loading model in {model_location}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_location, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_location, trust_remote_code=True).cuda()

    if os.path.exists('delta.pt'):
        from opendelta import LoraModel
        delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="hf")
        model.load_state_dict(load_delta('delta.pt'), strict=False)

    device_map = {
        "cpmbee.input_embedding": 0,
        "cpmbee.position_bias": 0,
        "lm_head": 0,
        "cpmbee.encoder.output_layernorm": 0
    }
    for i in range(48):
        device_map["cpmbee.encoder.layers.{}".format(i)] = i % int(tensor_parallel)

    model = dispatch_model(model, device_map=device_map)
    return model, tokenizer

def handle(inputs: Input) -> None:
    global model, tokenizer
    try:
        if not model:
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
                tmp_inputs.append(tmp_item.get("inputs"))
            
            #For server side batch, we just use the custom generation parameters for single Sagemaker Endpoint.
            result = model.generate(tmp_inputs, tokenizer)
            
            outputs = Output()
            for i in range(len(result)):
                outputs.add(result[i], key="generate_text", batch_index=i)
            return outputs
        else:
            inputs = inputs.get_as_json()
            if not inputs.get("inputs"):
                return Output().add_as_json({"code":-1,"msg":"input field can't be null"})

            #input data
            data = inputs.get("inputs")
            params = inputs.get("parameters",{})

            #for pure client side batch
            if type(data) == str:
                bs = 1
            elif type(data) == list:
                bs = len(data)
            else:
                return Output().add_as_json({"code":-1,"msg": "input has wrong type"})
                
            print("client side batch size is ", bs)
            #predictor
            result = model.generate(data, tokenizer, **params)

            #return
            return Output().add({"code":0,"msg":"ok","data":result})
    except Exception as e:
        return Output().add_as_json({"code":-1,"msg":e})
