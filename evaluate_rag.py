import json
from rag_pipeline import generate_answer, download_and_extract
from difflib import SequenceMatcher
from pathlib import Path

# 🧠 Helper for similarity scoring
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# 📂 Load JSON data
def load_eval_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

# 🔍 Evaluate RAG accuracy
def evaluate(questions, full_text):
    print("🔍 Evaluating RAG Pipeline...\n")
    correct = 0
    total = len(questions)
    low_score_logs = []

    for idx, item in enumerate(questions, 1):
        question = item["question"]
        expected = item["ground_truth"]

        predicted = generate_answer(question, full_text)

        sim_score = similarity(expected, predicted)
        match = sim_score > 0.75

        print(f"Q{idx}: {question}")
        print(f"✅ Ground Truth: {expected}")
        print(f"🤖 Prediction  : {predicted}")
        print(f"📊 Similarity  : {sim_score:.2f} --> {'✔️' if match else '❌'}\n")

        if not match:
            low_score_logs.append({
                "question": question,
                "expected": expected,
                "predicted": predicted,
                "similarity": sim_score
            })

        if match:
            correct += 1

    accuracy = correct / total * 100
    print(f"\n🧮 Final Accuracy: {accuracy:.2f}% ({correct}/{total})")

    if low_score_logs:
        print("\n⚠️ Low Accuracy Predictions (for tuning):")
        for i, log in enumerate(low_score_logs, 1):
            print(f"\n{i}. {log['question']}")
            print(f"   - Expected : {log['expected']}")
            print(f"   - Predicted: {log['predicted']}")
            print(f"   - Score    : {log['similarity']:.2f}")

# 🚀 Run evaluation
if __name__ == "__main__":
    path = Path("data/eval_questions.json")
    doc_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"  # 🔁 Replace this with your actual PDF URL
    questions = load_eval_data(path)
    full_text = download_and_extract(doc_url)
    evaluate(questions, full_text)
