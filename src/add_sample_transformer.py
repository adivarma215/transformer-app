from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", device_map="auto", load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(
						"mistralai/Mistral-7B-v0.1", padding_side="left"
						)

model_inputs = tokenizer(["My beautiful wife is"], return_tensors="pt").to("mps")
generate_ids = model.generate(**model_inputs)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True))


