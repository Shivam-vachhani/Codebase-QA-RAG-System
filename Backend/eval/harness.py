import json,pathlib
from app.services.rag_service import get_rag_service

GOLDEN_SET_PATH = pathlib.Path(__file__).parent / "golden_questions.json"
COLLECTED_PATH = pathlib.Path(__file__).parent / "collected_samples.json"

model = "gpt-4o"

def load_golden_questions()-> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)

def collect_samples()-> list[dict]:
     """Runs REAL pipeline for every golden question and captures
     exactly what RAGAS needs: the question, retrieved contexts, and answer.
     Deliberately calls get_rag_service()/run() — the same entrypoint
     queryRoute.py uses — so eval measures the real system, not a stand-in."""

     questions = load_golden_questions()
     samples = []

     for i,q in enumerate(questions):
         print(f"[{i+1}/{len(questions)}] {q["question"]}")
         pipeline = get_rag_service(q["repo_id"],model)
         result = pipeline.run(q["question"],include_context=True)

         samples.append({
             "user_input": q["question"],
             "retrived_context":result["context"],
             "response":result["answer"],
             "repo_id":q["repo_id"],
             "expected_classification":q.get("expected_classification"),
             "actual_classification":q.get("classification")
         })

     return samples

if __name__ == "__main__":
    samples = collect_samples()
    with open (COLLECTED_PATH,"w") as f:
        json.dump(samples,f,indent=2)
    print(f"\nCollected {len(samples)} samples -> {COLLECTED_PATH}")

     
    