'''
python scripts/evaluation/score_existing_csv.py > final_score.txt 2>&1
'''


import pandas as pd
import sys
import ast
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

# Import the existing scoring function and patch logic
from scripts.evaluation.eval_rag_metrics import run_ragas_scoring

def main():
    csv_path = "eval_results.csv"
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # When reading from CSV, the 'contexts' column is a string representation of a list.
    # We must convert it back to an actual Python list for RAGAS.
    if 'contexts' in df.columns:
        df['contexts'] = df['contexts'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [str(x)]
        )
        
    print(f"Loaded {len(df)} samples from {csv_path}")
    print("Running RAGAS scoring (judge: gpt-4o-mini)...")
    
    # Run the evaluation using the imported function
    result = run_ragas_scoring(df, judge_model="gpt-4o-mini")
    
    print("\n" + "=" * 40)
    print("FINAL RAGAS SCORES")
    print("=" * 40)
    print(result)
    
    if "category" in df.columns and result is not None:
        scores_df = result.to_pandas()
        if len(scores_df) == len(df):
            scores_df["category"] = df["category"].values
            print("\nPer-category breakdown:")
            metric_cols = [
                c for c in scores_df.columns 
                if c != "category" and pd.api.types.is_numeric_dtype(scores_df[c])
            ]
            grouped = scores_df.groupby("category")[metric_cols].mean()
            print(grouped.to_string(float_format="{:.3f}".format))

if __name__ == "__main__":
    main()
