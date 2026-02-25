#!/bin/env python3

from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer

#
# Explanation of what's being tested:
#  - transformers integration:
#      * AutoTokenizer ensures tokenization functionality from the
#        Transformers library is correctly integrated.
#
#  - optimum.intel.openvino integration:
#      * OVModelForCausalLM ensures Transformers models can be loaded,
#        converted, and executed with OpenVINO optimizations via optimum.intel.
#

# Model name compatible with OpenVINO optimizations
model_name = "gpt2"

# Load tokenizer (Transformers API)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load optimized model (Optimum Intel API with OpenVINO backend)
model = OVModelForCausalLM.from_pretrained(model_name, export=True)

# Prepare input text
prompt = "Testing transformers and optimum.intel integration"
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
input_ids = inputs.input_ids
attention_mask = inputs.attention_mask

# Generate output (testing both transformers tokenization & OpenVINO inference)
output_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=40)
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Prompt:", prompt)
print("Generated text:", generated_text)
