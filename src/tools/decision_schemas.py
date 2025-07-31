from __future__ import annotations
import numpy as np
from pydantic import Field
from typing import Literal, Dict, List, ClassVar
# ... (other imports)
from .signal_processing_schemas import register_op, DecisionOp, OP_REGISTRY
from ..states.phm_states import PHMState, ProcessedData, InputData, get_node_data

@register_op
class SimilarityScorer(DecisionOp):
    """
    Calculates similarity scores between sets of reference and test signals.
    The output is a dictionary mapping each test node to its similarity scores against all reference nodes.
    """
    op_name: ClassVar[str] = "score_similarity"
    description: ClassVar[str] = "Computes similarity scores between multiple reference and test signals."
    
    reference_node_ids: List[str] = Field(..., description="A list of reference node IDs from the DAG.")
    test_node_ids: List[str] = Field(..., description="A list of test node IDs from the DAG.")
    metric: Literal["cosine", "mae"] = Field("cosine", description="The similarity metric to use.")

    def execute(self, state: PHMState, **_) -> Dict[str, Dict[str, float]]:
        """
        Returns a nested dictionary: {test_node_id: {ref_node_id: score, ...}, ...}
        """
        results = {}
        for test_id in self.test_node_ids:
            test_data = get_node_data(state, test_id)
            if test_data is None: continue
            
            scores = {}
            for ref_id in self.reference_node_ids:
                ref_data = get_node_data(state, ref_id)
                if ref_data is None: continue

                # Simple cosine similarity for demonstration
                score = np.dot(ref_data, test_data) / (np.linalg.norm(ref_data) * np.linalg.norm(test_data))
                scores[ref_id] = float(score)
            
            results[test_id] = scores
        return results

@register_op
class TopKClassifier(DecisionOp):
    """
    Makes a diagnosis for each test signal by checking if its most similar reference signal is in the top K matches.
    This avoids hard-coded thresholds.
    """
    op_name: ClassVar[str] = "classify_by_top_k"
    description: ClassVar[str] = "Classifies test signals based on top-K similarity scores against references."

    scores_node_id: str = Field(..., description="The node ID containing the nested dictionary of similarity scores.")
    k: int = Field(1, description="The number of top matches to consider for a positive classification.")
    label_if_match: str = Field("Healthy", description="The diagnosis label if a test signal's best match is among the top K references.")
    label_if_mismatch: str = Field("Faulty", description="The diagnosis label if a test signal's best match is not in the top K.")

    def execute(self, state: PHMState, **_) -> Dict[str, Dict]:
        """
        Returns a dictionary mapping each test node ID to its diagnosis.
        """
        scores_node = state.dag_state.nodes.get(self.scores_node_id)
        if not scores_node or not isinstance(scores_node.result, dict):
            return {"error": "Scores node not found or result is not a dictionary."}
        
        scores_data = scores_node.result
        diagnoses = {}

        for test_id, ref_scores in scores_data.items():
            if not ref_scores:
                diagnoses[test_id] = {"diagnosis": "Undetermined", "reason": "No similarity scores available."}
                continue

            # Sort references by similarity score in descending order
            sorted_refs = sorted(ref_scores.items(), key=lambda item: item[1], reverse=True)
            
            # For this example, we assume the "true" reference is known by name convention
            # In a real scenario, this logic might be more complex
            # Let's assume a test signal 'test_01' should match a reference 'ref_01'
            expected_ref_id = test_id.replace("test", "ref") 

            # Get the IDs of the top K matches
            top_k_ref_ids = [ref_id for ref_id, score in sorted_refs[:self.k]]

            if expected_ref_id in top_k_ref_ids:
                diagnosis = self.label_if_match
            else:
                diagnosis = self.label_if_mismatch
            
            diagnoses[test_id] = {
                "diagnosis": diagnosis,
                "best_match": sorted_refs[0][0],
                "best_score": round(sorted_refs[0][1], 4)
            }
        
        return diagnoses
