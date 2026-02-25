from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

#
# Explanation of What This Verifies:
#
#  * Tokenizer compatibility: ensures tokenization and decoding work properly.
#  * Core model loading: confirms transformers properly load models and weights.
#  * Inference via pipeline: ensures the transformers pipeline mechanism
#    is working properly.
#

# Load tokenizer and model directly from transformers
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)

# Test tokenization explicitly
test_text = "The transformers library on RHEL 9"

encoded = tokenizer.encode(text=test_text, return_tensors='pt')
decoded = tokenizer.decode(token_ids=encoded[0])


print("Original text:", test_text)
print("Decoded text after tokenization:", decoded)

# Test text-generation pipeline
text_generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
generated_text = text_generator(text_inputs=test_text, max_length=30, num_return_sequences=1)

print("\nGenerated text example:")
print(generated_text[0]["generated_text"])
