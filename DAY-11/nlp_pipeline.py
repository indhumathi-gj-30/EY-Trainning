import os
import re
import json
import pandas as pd
from rouge_score import rouge_scorer
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

text_generator = pipeline("text-generation", model="gpt2")
target_llm = HuggingFacePipeline(pipeline=text_generator)
output_parser = StrOutputParser()
print("\n" + "#" * 80)
print("MODULE 1: EVALUATING ZERO-SHOT AND FEW-SHOT EXPERIMENTS")
print("#" * 80 + "\n")

generation_blueprint = """
You are an expert financial researcher.

Synthesize exactly 3 custom, distinct corporate earnings call excerpts.
Each corporate excerpt must be between 4 and 5 sentences in length containing:
1. Historical financial trajectory
2. Primary operational catalyst
3. Near-term forward management projections

Output the result using this strict schema template format:

Excerpt 1:
...

Excerpt 2:
...

Excerpt 3:
...
"""

generation_prompt = PromptTemplate.from_template(generation_blueprint)
generation_pipeline = generation_prompt | target_llm | output_parser

raw_synthesized_response = generation_pipeline.invoke({})
extracted_excerpts_text = raw_synthesized_response

print("Generated Financial Data Excerpts:\n")
print(extracted_excerpts_text)
print("\n" + "." * 80 + "\n")
excerpt_regex_pattern = r"Excerpt\s*\d+\s*:\s*(.*?)(?=Excerpt\s*\d+\s*:|$)"
regex_matches = re.findall(excerpt_regex_pattern, extracted_excerpts_text, re.DOTALL)
evaluated_excerpts = [match.strip() for match in regex_matches]

if len(evaluated_excerpts) != 3:
    evaluated_excerpts = [
        """Revenue grew 12% year-over-year, driven by strong cloud subscription demand.
Operating margin improved to 21% because of lower infrastructure costs.
Management said enterprise renewals remained healthy across key regions.
The company expects steady momentum in the next quarter, though it remains cautious on global macro headwinds.""",

        """Quarterly sales declined 4% due to weaker smartphone shipments in Asia.
However, services revenue hit a record level and gross margin stayed stable.
Executives highlighted strong user retention and rising recurring revenue.
Management expects gradual recovery as supply chain conditions improve in the coming months.""",

        """Advertising revenue fell 6% because retail clients reduced spending.
Despite this, the firm added over 150 new enterprise AI customers during the quarter.
Operating expenses declined following restructuring efforts.
Leadership expects profitability to improve in the second half of the year as AI adoption grows."""
    ]

# --- Zero-Shot Engine Architecture ---
zero_shot_blueprint = """
You are acting as a senior financial audit bot.

Condense the given earnings call document down into exactly 3 clear bullets.

Target Focus Areas:
1. Numerical metrics / Performance
2. Dynamic Business Drivers
3. Future Executive Path

Document:
{document}

Bullet Summary:
"""

zero_shot_prompt = PromptTemplate.from_template(zero_shot_blueprint)
zero_shot_pipeline = zero_shot_prompt | target_llm | output_parser
few_shot_blueprint = """
You are acting as a senior financial audit bot.

Review the reference structural examples detailing the precise bullet summarization formatting:

Example A
Document:
Revenue increased 10% year-over-year due to strong software demand.
Operating profit improved because of lower marketing spend.
Management expects moderate growth next quarter.

Bullet Summary:
- Revenue rose 10% year-over-year, mainly supported by software demand.
- Profitability improved due to lower marketing expenses.
- Management expects moderate growth in the upcoming quarter.

Example B
Document:
Hardware sales declined, but cloud revenue grew 18%.
Margins were pressured by logistics costs.
Executives remain optimistic about retention and product expansion.

Bullet Summary:
- Hardware performance was weak, while cloud revenue increased 18%.
- Margins were affected by higher logistics expenses.
- Leadership remains positive about customer retention and future expansion.

Process the target incoming document using the exact benchmark paradigm shown above.

Document:
{document}

Bullet Summary:
"""

few_shot_prompt = PromptTemplate.from_template(few_shot_blueprint)
few_shot_pipeline = few_shot_prompt | target_llm | output_parser

# --- Ground Truth Gold References ---
gold_standard_references = [
    """- Revenue grew strongly year-over-year due to cloud subscription demand.
- Margins improved because of reduced infrastructure costs and healthy renewals.
- Management expects steady next-quarter performance but remains cautious on macro conditions.""",

    """- Sales declined because of weaker smartphone demand, especially in Asia.
- Services revenue remained strong and supported stability in margins.
- Management expects supply chain improvements and gradual recovery ahead.""",

    """- Advertising revenue declined as retail clients cut spending.
- AI customer growth and lower operating expenses were major positives.
- Leadership expects profitability improvement in the second half of the year."""
]

if len(evaluated_excerpts) != len(gold_standard_references):
    gold_standard_references = [
        "Revenue performance, key business driver, and management outlook were discussed.",
        "Revenue performance, business trends, and future guidance were highlighted.",
        "Financial results, growth drivers, and future outlook were summarized."
    ]

zero_shot_predictions = []
few_shot_predictions = []

print("RUNNING ZERO-SHOT EXECUTIONS:\n")
for idx, doc in enumerate(evaluated_excerpts, start=1):
    z_res = zero_shot_pipeline.invoke({"document": doc}).strip()
    zero_shot_predictions.append(z_res)
    print(f"Excerpt Index {idx}:\n{z_res}\n")

print("\n" + "." * 80 + "\n")

print("RUNNING FEW-SHOT EXECUTIONS:\n")
for idx, doc in enumerate(evaluated_excerpts, start=1):
    f_res = few_shot_pipeline.invoke({"document": doc}).strip()
    few_shot_predictions.append(f_res)
    print(f"Excerpt Index {idx}:\n{f_res}\n")

# --- ROUGE METRIC ACCURACY RUN ---
print("\n" + "#" * 80)
print("METRIC BENCHMARKING: ROUGE-L EVALUATION")
print("#" * 80 + "\n")

metric_calculator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
score_table_records = []

for idx in range(len(evaluated_excerpts)):
    baseline_ref = gold_standard_references[idx] if idx < len(gold_standard_references) else ""
    
    z_score_f = metric_calculator.score(baseline_ref, zero_shot_predictions[idx])["rougeL"].fmeasure
    f_score_f = metric_calculator.score(baseline_ref, few_shot_predictions[idx])["rougeL"].fmeasure

    score_table_records.append({
        "Index_ID": idx + 1,
        "Score_0_Shot_ROUGE_L": round(z_score_f, 4),
        "Score_Few_Shot_ROUGE_L": round(f_score_f, 4)
    })

analytics_dataframe = pd.DataFrame(score_table_records)
print(analytics_dataframe)

print("\nMean ROUGE-L Metric Score (0-Shot):", round(analytics_dataframe["Score_0_Shot_ROUGE_L"].mean(), 4))
print("Mean ROUGE-L Metric Score (Few-Shot):", round(analytics_dataframe["Score_Few_Shot_ROUGE_L"].mean(), 4))



print("\n" + "#" * 80)
print("MODULE 2: 5-WAY CLASSIFICATION CORRELATION ENGINE")
print("#" * 80 + "\n")

inbound_customer_tickets = [
    "I was charged twice for my monthly subscription and need help fixing the invoice.",
    "The mobile app crashes every time I try to upload a document.",
    "I canceled my order yesterday and want my money returned.",
    "How do I update my profile details in the portal?",
    "My account has been locked for a week and no one from support has responded. This is urgent."
]

triage_classifier_blueprint = """
You are an intelligent customer ticket triage classification model.

Categorize the text input down below exclusively inside exactly 1 of these 5 designated labels:
1. Billing
2. Tech
3. Refund
4. General
5. Escalate

Processing Guidelines:
- Break down and establish the structural root problem.
- Formulate a compressed logical argument summary.
- Assign the deterministic target class string.
- Keep the analytical argument direct and short.

Inbound Ticket Text:
{ticket_content}

Strict Output Serialization Structure:

Logical Argument Summary:
...

Assigned Category Label:
...
"""

triage_prompt = PromptTemplate.from_template(triage_classifier_blueprint)
triage_pipeline = triage_prompt | target_llm | output_parser

triage_processed_rows = []

for idx, user_ticket in enumerate(inbound_customer_tickets, start=1):
    raw_triage_result = triage_pipeline.invoke({"ticket_content": user_ticket}).strip()

    parsed_logic = re.search(r"Logical Argument Summary:\s*(.*?)\s*Assigned Category Label:", raw_triage_result, re.DOTALL)
    parsed_class = re.search(r"Assigned Category Label:\s*(.*)", raw_triage_result, re.DOTALL)

    argument_text = parsed_logic.group(1).strip() if parsed_logic else "Metadata parsing missing"
    category_label = parsed_class.group(1).strip() if parsed_class else "Metadata parsing missing"

    triage_processed_rows.append({
        "ID": idx,
        "Inbound_Text": user_ticket,
        "Model_Argument": argument_text,
        "Resolved_Class": category_label
    })

triage_dataframe = pd.DataFrame(triage_processed_rows)
print(triage_dataframe.to_string(index=False))
analytics_dataframe.to_csv("rouge_l_scores.csv", index=False)
triage_dataframe.to_csv("ticket_classification_results.csv", index=False)

with open("generated_earnings_snippets.txt", "w", encoding="utf-8") as file_writer:
    file_writer.write(extracted_excerpts_text)

with open("zero_shot_summaries.txt", "w", encoding="utf-8") as file_writer:
    for idx, text in enumerate(zero_shot_predictions, start=1):
        file_writer.write(f"Snippet {idx} Zero-shot Summary:\n{text}\n\n")

with open("few_shot_summaries.txt", "w", encoding="utf-8") as file_writer:
    for idx, text in enumerate(few_shot_predictions, start=1):
        file_writer.write(f"Snippet {idx} Few-shot Summary:\n{text}\n\n")

print("\nAll pipeline logs exported successfully onto system disk storage.")