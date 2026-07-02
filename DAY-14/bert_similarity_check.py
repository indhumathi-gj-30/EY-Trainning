from bert_score import score
predicted_sentence = [
    "Artificial intelligence is transforming many industries through automation and data analysis."
]
actual_sentence = [
    "AI is changing various sectors by automating tasks and analyzing large amounts of data."
]
precision, recall, f1 = score(
    predicted_sentence,
    actual_sentence,
    lang="en",
    verbose=False
)
print("Precision :", round(precision.mean().item(), 4))
print("Recall    :", round(recall.mean().item(), 4))
print("F1 Score  :", round(f1.mean().item(), 4))