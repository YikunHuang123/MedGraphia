import sys
import random
import time
from pathlib import Path
import torch

# Ensure src is in PYTHONPATH
src_path = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.append(str(src_path))

from medgraphia.data.mesh import MeSHLoader
from medgraphia.ingestion.entity_linker import EntityLinker

def main():
    print("=== SapBERT Entity Linking: Synonym Hold-Out Evaluation ===")
    print("1. Loading MeSH Database...")
    loader = MeSHLoader(storage_dir="data/mesh")
    # Load all concepts to make the retrieval space realistic (19k+ concepts)
    raw_index = loader.load()
    print(f"   Loaded {len(raw_index)} concepts.")

    # Build Test Set
    test_cases = []
    modified_index = {}
    
    random.seed(42)
    for cui, concept in raw_index.items():
        syns = concept.get("synonyms", [])
        if syns and len(syns) > 0:
            # Pick a random synonym to hold out
            holdout = random.choice(syns)
            new_syns = [s for s in syns if s != holdout]
            
            # Create modified concept (hide the holdout from the linker)
            mod_concept = dict(concept)
            mod_concept["synonyms"] = new_syns
            modified_index[cui] = mod_concept
            
            test_cases.append({
                "mention": holdout,
                "target_cui": cui,
                "entity_type": concept.get("entity_type", "Unknown")
            })
        else:
            modified_index[cui] = dict(concept)
            
    print(f"2. Generated {len(test_cases)} hold-out test cases.")
    
    # Sample down to 1000 to save evaluation time
    if len(test_cases) > 1000:
        test_cases = random.sample(test_cases, 1000)
        
    print(f"3. Initializing Entity Linker with modified database...")
    # Initialize linker with the modified index (where the test synonyms are deleted)
    linker = EntityLinker(concept_index=modified_index)
    linker.build_index()
    
    print(f"4. Running dense retrieval for {len(test_cases)} test cases...")
    start_time = time.time()
    
    # Encode all held-out mentions
    mentions = [tc["mention"] for tc in test_cases]
    mention_embs = linker._sapbert.encode(
        mentions, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True
    )
    
    # Calculate cosine similarity against all MeSH concepts
    dict_embs = linker._concept_embs
    scores = torch.matmul(mention_embs, dict_embs.T)
    topk_scores, topk_idxs = torch.topk(scores, k=50, dim=1)
    
    topk_idxs = topk_idxs.cpu().tolist()
    
    hit1 = 0
    hit5 = 0
    mrr_sum = 0.0
    
    # Note: SapBERT flat index might contain multiple entries for the same CUI 
    # (e.g. English label and Chinese label). We want to find the highest rank of the target CUI.
    for i, tc in enumerate(test_cases):
        target = tc["target_cui"]
        rank = -1
        
        # Traverse top-50 results
        for r, idx in enumerate(topk_idxs[i]):
            cui = linker._entries[idx].cui
            if cui == target:
                rank = r + 1  # 1-based rank
                break
                
        if rank == 1:
            hit1 += 1
        if 0 < rank <= 5:
            hit5 += 1
            
        if rank > 0:
            mrr_sum += 1.0 / rank
            
    end_time = time.time()
    
    print("\n=== Evaluation Results ===")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Inference Time:   {end_time - start_time:.2f}s")
    print(f"Accuracy@1:       {hit1 / len(test_cases) * 100:.2f}%")
    print(f"Accuracy@5:       {hit5 / len(test_cases) * 100:.2f}%")
    print(f"MRR:              {mrr_sum / len(test_cases):.4f}")

if __name__ == "__main__":
    main()
