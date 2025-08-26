import numpy as np
from typing import Dict, List

from src.states.phm_states import PHMState, ProcessedData, InputData


def _calc_metric(a: np.ndarray, b: np.ndarray, metric: str) -> float:
    if metric == "cosine":
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom else 0.0
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "pearson":
        r = np.corrcoef(a, b)[0, 1]
        return float(1 - r)
    raise ValueError(f"unknown metric {metric}")


def inquirer_agent(state: PHMState, metrics: List[str]) -> Dict[str, List[str]]:
    """
    Calculates a similarity matrix between reference and test signals for each leaf node
    and stores it in the node's `sim` attribute.
    """
    leaf_ids = list(state.dag_state.leaves)
    print(f"Calculating similarity for {len(leaf_ids)} leaf nodes with metrics: {metrics}")

    for leaf_id in leaf_ids:
        node = state.dag_state.nodes.get(leaf_id)

        if not isinstance(node, (ProcessedData, InputData)) or not node.results:
            print(f"Node {leaf_id} is not a ProcessedData or InputData node or has no results.")
            continue

        ref_data_dict = node.results.get("ref")
        tst_data_dict = node.results.get("tst")

        if not isinstance(ref_data_dict, dict) or not isinstance(tst_data_dict, dict):
            state.dag_state.error_log.append(f"ref/tst data in node {leaf_id} is not a dictionary.")
            print(f"Node {leaf_id} has invalid ref/tst data.")
            continue

        # Initialize the similarity dictionary for the node
        node.sim = {}

        for metric in metrics:
            # Create a similarity matrix (dict of dicts) for the current metric
            sim_matrix = {}
            for ref_key, ref_val in ref_data_dict.items():
                sim_matrix[ref_key] = {}
                a = np.asarray(ref_val).ravel()
                
                for tst_key, tst_val in tst_data_dict.items():
                    b = np.asarray(tst_val).ravel()
                    
                    if a.shape != b.shape:
                        state.dag_state.error_log.append(f"Shape mismatch between {ref_key} and {tst_key} in node {leaf_id}")
                        print(f"Shape mismatch between {ref_key} and {tst_key} in node {leaf_id}")
                        continue
                    
                    try:
                        # Calculate the similarity and store it in the matrix
                        val = _calc_metric(a, b, metric)
                        sim_matrix[ref_key][tst_key] = val
                    except Exception as exc:
                        print(f"Error calculating {metric} between {ref_key} and {tst_key} in node {leaf_id}: {exc}")
                        state.dag_state.error_log.append(f"{metric} fail between {ref_key} and {tst_key}: {exc}")
            
            # Store the complete similarity matrix for the metric in the node's `sim` attribute
            node.sim[metric] = sim_matrix
            print(f"Calculated {metric} similarity for node {leaf_id}")

    # No new nodes are created. The agent modifies existing nodes.
    return {"new_nodes": []}


if __name__ == "__main__":
    import os
    import sys
    import numpy as np
    from scipy.stats import kurtosis
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from src.states.phm_states import PHMState, DAGState, InputData

    # --- 1. Setup: Mock data and operators ---
    ref_sig = {'id1': np.random.randn(1, 1024, 2),
               'id2': np.random.randn(1, 1024, 2) * 1.5,
               'id3': np.random.randn(1, 1024, 2) * 2.0}
    tst_sig = {'id4': np.random.randn(1, 1024, 2) * 1.2,
               'id5': np.random.randn(1, 1024, 2) * 1.8,
               'id6': np.random.randn(1, 1024, 2) * 2.5}

    # Mock operators
    def mock_fft(data):
        return np.fft.fft(data).real

    def mock_kurtosis(data):
        return kurtosis(data, axis=None)

    # --- 2. Build the complete DAG for testing ---
    nodes = {}
    
    # Step A: Raw Signal Nodes
    ch1_node = InputData(
        node_id="ch1",
        results={"ref": ref_sig, "tst": tst_sig},
        parents=[],
        shape=(1, 1024, 2),
        meta={"channel": "ch1"}
    )
    ch2_node = InputData(
        node_id="ch2",
        results={"ref": ref_sig, "tst": tst_sig}, # Use same data for ch2 for simplicity
        parents=[],
        shape=(1, 1024, 2),
        meta={"channel": "ch2"}
    )
    nodes.update({"ch1": ch1_node, "ch2": ch2_node})

    # Step B: FFT Nodes
    fft_nodes = {}
    for parent_id in ["ch1", "ch2"]:
        parent_node = nodes[parent_id]
        ref_in, tst_in = parent_node.results["ref"], parent_node.results["tst"]
        
        out_ref = {key: mock_fft(value) for key, value in ref_in.items()}
        out_tst = {key: mock_fft(value) for key, value in tst_in.items()}

        fft_node = ProcessedData(
            node_id=f"fft_{parent_id}",
            parents=[parent_id],
            source_signal_id=parent_id,
            method="fft",
            processed_data=out_tst,  # Add the required field
            results={"ref": out_ref, "tst": out_tst},
            meta={"channel": parent_node.meta["channel"], "method": "fft"},
            shape=next(iter(out_ref.values())).shape
        )
        fft_nodes[fft_node.node_id] = fft_node
    nodes.update(fft_nodes)

    # Step C: Kurtosis Nodes
    kurtosis_nodes = {}
    for parent_id, parent_node in fft_nodes.items():
        ref_in, tst_in = parent_node.results["ref"], parent_node.results["tst"]
        
        out_ref = {key: mock_kurtosis(value) for key, value in ref_in.items()}
        out_tst = {key: mock_kurtosis(value) for key, value in tst_in.items()}

        kurt_node = ProcessedData(
            node_id=f"kurtosis_{parent_id}",
            parents=[parent_id],
            source_signal_id=parent_node.source_signal_id,
            method="kurtosis",
            processed_data=out_tst,  # Add the required field
            results={"ref": out_ref, "tst": out_tst},
            meta={"channel": parent_node.meta["channel"], "method": "kurtosis"},
            shape=() # Scalar output
        )
        kurtosis_nodes[kurt_node.node_id] = kurt_node
    nodes.update(kurtosis_nodes)
    
    # The leaves are the last processed nodes
    leaves = list(kurtosis_nodes.keys())

    # --- 3. Create the final PHMState ---
    dag = DAGState(
        user_instruction="inquirer_test",
        channels=["ch1", "ch2"],
        nodes=nodes,
        leaves=leaves,
    )
    st = PHMState(
        user_instruction="inquirer_test",
        dag_state=dag,
        reference_signal=nodes["ch1"],
        test_signal=nodes["ch1"],
    )

    print("--- Initial DAG for Inquirer ---")
    print(f"Nodes: {list(st.dag_state.nodes.keys())}")
    print(f"Leaves: {st.dag_state.leaves}")
    print("--------------------------------\n")

    # --- 4. Run the inquirer_agent ---
    result = inquirer_agent(st, metrics=["euclidean", "cosine"])

    print("\n--- Inquirer Agent Results ---")
    print(f"Result: {result}")
    print(f"Final Leaves: {st.dag_state.leaves}")
    kurt_node_ch1 = st.dag_state.nodes['kurtosis_fft_ch1']
    print(f"Updated node 'kurtosis_fft_ch1' sim attribute: {kurt_node_ch1.sim}")
    print(f"Node 'kurtosis_fft_ch1' results attribute remains unchanged: {kurt_node_ch1.results is not None}")
    print(f"Errors: {st.dag_state.error_log}")
    print("------------------------------\n")

    # --- 5. Verification ---
    initial_node_count = len(nodes)
    assert len(result["new_nodes"]) == 0, "No new nodes should be created"
    assert len(st.dag_state.nodes) == initial_node_count, "The number of nodes should not change"

    # Check that the leaf nodes have been updated with similarity matrices
    for leaf_id in leaves:
        updated_node = st.dag_state.nodes[leaf_id]
        assert "euclidean" in updated_node.sim
        assert "cosine" in updated_node.sim
        
        # Check the structure of the similarity matrix
        euclidean_matrix = updated_node.sim["euclidean"]
        assert isinstance(euclidean_matrix, dict)
        first_ref_key = next(iter(ref_sig.keys()))
        first_tst_key = next(iter(tst_sig.keys()))
        assert first_tst_key in euclidean_matrix[first_ref_key]
        assert isinstance(euclidean_matrix[first_ref_key][first_tst_key], float)

        # Check that original results are untouched
        assert "ref" in updated_node.results and "tst" in updated_node.results

    # Check that leaves were NOT changed
    assert "kurtosis_fft_ch1" in st.dag_state.leaves
    assert "sim_euclidean_kurtosis_fft_ch1" not in st.dag_state.nodes

    print("✅ Inquirer Agent test passed!")
