import torch
import transformers
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoConfig
from jetmoe import JetMoEForCausalLM, JetMoEConfig, JetMoEForSequenceClassification

print(transformers.__version__)
# Initialize the model and tokenizer
AutoConfig.register("jetmoe", JetMoEConfig)
model_name = "./chat-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = JetMoEForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, attn_implementation="eager", trust_remote_code=True)
# Check if a GPU is available and move the model to GPU if it is
if torch.cuda.is_available():
    model = model.cuda()
    print("Using GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("GPU is not available, using CPU instead.")
# Encode input context
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    },
    {"role": "user", "content": "How many helicopters can a human eat in one sitting?"},
 ]
tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
print(tokenized_chat)
# If using a GPU, move the input IDs to the GPU
input_ids = tokenized_chat
if torch.cuda.is_available():
    input_ids = tokenized_chat.cuda()
    # Generate text
output = model.generate(input_ids, max_length=500, num_return_sequences=1, no_repeat_ngram_size=2)
# If the output is on the GPU, move it back to CPU for decoding
if torch.cuda.is_available():
    output = output.cpu()    
# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
