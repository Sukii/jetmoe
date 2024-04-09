from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSequenceClassification
from jetmoe import JetMoEForCausalLM, JetMoEConfig, JetMoEForSequenceClassification

AutoConfig.register("jetmoe", JetMoEConfig)
AutoModelForCausalLM.register(JetMoEConfig, JetMoEForCausalLM)
AutoModelForSequenceClassification.register(JetMoEConfig, JetMoEForSequenceClassification)

tokenizer = AutoTokenizer.from_pretrained('./model')
model = AutoModelForCausalLM.from_pretrained('./model')

# Generate text
for i in range(10):
    input_text = input("Query:")
    input_text = "Question: " + input_text + " \nAnswer:"
    encoded_input = tokenizer(input_text, return_tensors="pt")
    output = model.generate(**encoded_input, max_new_tokens=200)

    # Decode the output
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Print the generated text
    print(decoded_output)
