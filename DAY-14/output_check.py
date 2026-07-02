import spacy
from selfcheckgpt.modeling_selfcheck import SelfCheckBERTScore
reference_text = (
    "Michael Jordan was born in Brooklyn, New York. "
    "He played for the Chicago Bulls."
)
comparison_responses = [
    "Michael Jordan is a famous basketball player who spent most of his career with the Bulls. He was born in Brooklyn.",
    "Jordan was born in New York City and is regarded as one of the greatest basketball players ever.",
    "Born in Brooklyn, Michael Jordan became an international sports icon while playing for the Chicago Bulls."
]
language_model = spacy.load("en_core_web_sm")
sentence_list = [
    item.text.strip()
    for item in language_model(reference_text).sents
]
consistency_checker = SelfCheckBERTScore(
    rescale_with_baseline=True
)
consistency_scores = consistency_checker.predict(
    sentences=sentence_list,
    sampled_passages=comparison_responses
)
for text_line, value in zip(sentence_list, consistency_scores):
    print(f"Text: {text_line}")
    print(
        f"Consistency Score: {value:.4f} "
        "(Lower = More Consistent, Higher = Possible Hallucination)\n"
    )