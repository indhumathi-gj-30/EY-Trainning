from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

model_id = "vectara/hallucination_evaluation_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    trust_remote_code=True
)

context_text = "The sky is blue today because of Rayleigh scattering, and it is 72 degrees Fahrenheit outside."
response_text = "The weather is completely cloudy and raining."

model_inputs = tokenizer(
    context_text,
    response_text,
    return_tensors="pt"
)

with torch.no_grad():
    result = classifier_model(**model_inputs)
    scores = torch.softmax(result.logits, dim=-1)

consistency_probability = scores[0][1].item()

print(f"HHEM Factual Consistency Score: {consistency_probability:.4f}")
print(f"Hallucination Probability: {1 - consistency_probability:.4f}")