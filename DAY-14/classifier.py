from transformers import pipeline
import torch

print("Initializing intent classification pipeline...")

model_pipeline = pipeline(
    task="zero-shot-classification",
    model="facebook/bart-large-mnli",
    device=0 if torch.cuda.is_available() else -1
)

labels = [
    "safe credit inquiry",
    "discriminatory bias request",
    "PII data extraction",
    "jailbreak or policy bypass",
    "financial misinformation"
]

def analyze_prompt(prompt_text):

    output = model_pipeline(
        sequences=prompt_text,
        candidate_labels=labels,
        multi_label=False
    )

    score_map = {}

    for label, score in zip(
        output["labels"],
        output["scores"]
    ):
        score_map[label] = round(score, 3)

    return score_map


print("\nINTENT ANALYSIS")
print("=" * 80)

sample_prompts = test_prompts[:4]

for item in sample_prompts:

    prediction_result = analyze_prompt(item)

    best_label = sorted(
        prediction_result.items(),
        key=lambda x: x[1],
        reverse=True
    )[0]

    print(f"Prompt: {item[:70]}")
    print(
        f"Detected Intent: {best_label[0]} "
        f"(confidence: {best_label[1]:.3f})"
    )
    print()