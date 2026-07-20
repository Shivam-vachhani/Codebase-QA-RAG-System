from eval import _ragas_compat
import json,pathlib
from datetime import datetime
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness,ResponseRelevancy
from eval.config import get_judge_embeddings, get_judge_llm, EVAL_RUN_CONFIG

SAMPLES_PATH = pathlib.Path(__file__).parent / "collected_samples.json"
RESULTS_DIR = pathlib.Path(__file__).parent / "results"

def load_samples()->list[dict]:
    with open(SAMPLES_PATH) as f:
        return json.load(f)

def main():
    raw_samples = load_samples()
    ragas_row = [
        {
            "user_input":s["user_input"],
            "retrieved_contexts":s["retrived_context"],
            "response":s["response"]
        }
        for s in raw_samples 
    ]

    dataset = EvaluationDataset.from_list(ragas_row)
    judge_llm = get_judge_llm()
    judge_embeddings = get_judge_embeddings()

    metrics = [
        Faithfulness(llm=judge_llm),
        ResponseRelevancy(llm=judge_llm,embeddings=judge_embeddings),
    ]

    result = evaluate(
        dataset= dataset,
        metrics= metrics,
        llm = judge_llm,
        embeddings= judge_embeddings,
        run_config=EVAL_RUN_CONFIG,
    )

    df = result.to_pandas()
    df["repo_id"] = [s["repo_id"] for s in raw_samples]

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp= datetime.now().strftime("Y%m%d_%H%M%S")
    out_path = RESULTS_DIR/f"eval_{timestamp}.csv"
    df.to_csv(out_path,index=False)

    print(df[["user_input","faithfulness","answer_relevancy"]].sort_values("faithfulness"))
    print(f"\nFull results saved -> {out_path}")

if __name__ == "__main__":
    main()