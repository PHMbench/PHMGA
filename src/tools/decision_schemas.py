# from __future__ import annotations
# import numpy as np
# import numpy.typing as npt
# from pydantic import Field
# from typing import Literal, Dict, List, ClassVar
# from scipy import signal
# import scipy.stats
# # ... (other imports)
# from .signal_processing_schemas import register_op, DecisionOp, OP_REGISTRY
# from ..states.phm_states import PHMState, ProcessedData, InputData, get_node_data

# @register_op
# class SimilarityScorer(DecisionOp):
#     """
#     Calculates similarity scores between sets of reference and test signals.
#     The output is a dictionary mapping each test node to its similarity scores against all reference nodes.
#     """
#     op_name: ClassVar[str] = "score_similarity"
#     description: ClassVar[str] = "Computes similarity scores between multiple reference and test signals."
#     input_spec: ClassVar[str] = "DAG State"
#     output_spec: ClassVar[str] = "Dict[test_id, Dict[ref_id, score]]"
    
#     reference_node_ids: List[str] = Field(..., description="A list of reference node IDs from the DAG.")
#     test_node_ids: List[str] = Field(..., description="A list of test node IDs from the DAG.")
#     metric: Literal["cosine", "mae"] = Field("cosine", description="The similarity metric to use.")

#     def execute(self, state: PHMState, **_) -> Dict[str, Dict[str, float]]:
#         """
#         Returns a nested dictionary: {test_node_id: {ref_node_id: score, ...}, ...}
#         """
#         results = {}
#         for test_id in self.test_node_ids:
#             test_data = get_node_data(state, test_id)
#             if test_data is None: continue
            
#             scores = {}
#             for ref_id in self.reference_node_ids:
#                 ref_data = get_node_data(state, ref_id)
#                 if ref_data is None: continue

#                 # Simple cosine similarity for demonstration
#                 score = np.dot(ref_data, test_data) / (np.linalg.norm(ref_data) * np.linalg.norm(test_data))
#                 scores[ref_id] = float(score)
            
#             results[test_id] = scores
#         return results

# @register_op
# class TopKClassifier(DecisionOp):
#     """
#     Makes a diagnosis for each test signal by checking if its most similar reference signal is in the top K matches.
#     This avoids hard-coded thresholds.
#     """
#     op_name: ClassVar[str] = "classify_by_top_k"
#     description: ClassVar[str] = "Classifies test signals based on top-K similarity scores against references."
#     input_spec: ClassVar[str] = "DAG State"
#     output_spec: ClassVar[str] = "Dict[test_id, Dict[diagnosis, score]]"

#     scores_node_id: str = Field(..., description="The node ID containing the nested dictionary of similarity scores.")
#     k: int = Field(1, description="The number of top matches to consider for a positive classification.")
#     label_if_match: str = Field("Healthy", description="The diagnosis label if a test signal's best match is among the top K references.")
#     label_if_mismatch: str = Field("Faulty", description="The diagnosis label if a test signal's best match is not in the top K.")

#     def execute(self, state: PHMState, **_) -> Dict[str, Dict]:
#         """
#         Returns a dictionary mapping each test node ID to its diagnosis.
#         """
#         scores_node = state.dag_state.nodes.get(self.scores_node_id)
#         if not scores_node or not isinstance(scores_node.result, dict):
#             return {"error": "Scores node not found or result is not a dictionary."}
        
#         scores_data = scores_node.result
#         diagnoses = {}

#         for test_id, ref_scores in scores_data.items():
#             if not ref_scores:
#                 diagnoses[test_id] = {"diagnosis": "Undetermined", "reason": "No similarity scores available."}
#                 continue

#             # Sort references by similarity score in descending order
#             sorted_refs = sorted(ref_scores.items(), key=lambda item: item[1], reverse=True)
            
#             # For this example, we assume the "true" reference is known by name convention
#             # In a real scenario, this logic might be more complex
#             # Let's assume a test signal 'test_01' should match a reference 'ref_01'
#             expected_ref_id = test_id.replace("test", "ref") 

#             # Get the IDs of the top K matches
#             top_k_ref_ids = [ref_id for ref_id, score in sorted_refs[:self.k]]

#             if expected_ref_id in top_k_ref_ids:
#                 diagnosis = self.label_if_match
#             else:
#                 diagnosis = self.label_if_mismatch
            
#             diagnoses[test_id] = {
#                 "diagnosis": diagnosis,
#                 "best_match": sorted_refs[0][0],
#                 "best_score": round(sorted_refs[0][1], 4)
#             }
        
#         return diagnoses

# # @register_op
# # class ThresholdOp(DecisionOp):
# #     """
# #     Checks if a scalar value exceeds a threshold.
# #     """
# #     op_name: ClassVar[str] = "threshold"
# #     description: ClassVar[str] = "Returns 'True' if the input value exceeds the threshold, else 'False'."
# #     input_spec: ClassVar[str] = "Scalar"
# #     output_spec: ClassVar[str] = "Dict[is_exceeded, value, threshold]"
# #     threshold: float = Field(..., description="The threshold value to compare against.")
    
# #     def execute(self, x: npt.NDArray, **_) -> Dict[str, bool]:
# #         # Ensure input is a scalar or can be treated as one
# #         if x.size != 1:
# #             raise ValueError("ThresholdOp requires a scalar input.")
        
# #         value = x.item()
# #         is_exceeded = value > self.threshold
        
# #         return {
# #             "is_exceeded": is_exceeded,
# #             "value": value,
# #             "threshold": self.threshold
# #         }

# # @register_op
# # class RuleBasedDecisionOp(DecisionOp):
# #     """
# #     Makes a decision based on a set of simple rules.
# #     Each rule is a tuple of (variable_name, operator, value).
# #     Example: [("rms_ratio", ">", 0.8), ("crest_factor", "<", 1.2)]
# #     """
# #     op_name: ClassVar[str] = "rule_based_decision"
# #     description: ClassVar[str] = "Makes a decision based on a set of simple rules."
# #     input_spec: ClassVar[str] = "Dict[feature_name, value]"
# #     output_spec: ClassVar[str] = "Dict[decision, rule_evaluations]"
    
# #     rules: List[tuple[str, Literal[">", "<", "==", "!=", ">=", "<="], float]] = Field(..., description="List of rules to evaluate.")
# #     logical_op: Literal["AND", "OR"] = Field("AND", description="Logical operator to combine the rules.")

# #     def execute(self, x: Dict[str, float], **_) -> Dict[str, any]:
# #         """
# #         Input `x` is a dictionary of named feature values, e.g., {"rms_ratio": 0.9, "crest_factor": 1.1}
# #         """
# #         results = []
# #         for var, op, val in self.rules:
# #             if var not in x:
# #                 results.append(False) # A rule on a non-existent variable is false
# #                 continue
            
# #             val_from_input = x[var]
# #             if op == ">":
# #                 results.append(val_from_input > val)
# #             elif op == "<":
# #                 results.append(val_from_input < val)
# #             elif op == "==":
# #                 results.append(val_from_input == val)
# #             elif op == "!=":
# #                 results.append(val_from_input != val)
# #             elif op == ">=":
# #                 results.append(val_from_input >= val)
# #             elif op == "<=":
# #                 results.append(val_from_input <= val)
        
# #         if self.logical_op == "AND":
# #             final_decision = all(results)
# #         else: # OR
# #             final_decision = any(results)
            
# #         return {
# #             "decision": final_decision,
# #             "rule_evaluations": results,
# #             "rules": self.rules,
# #             "inputs": x
# #         }

# @register_op
# class SimilarityClassifierOp(DecisionOp):
#     """
#     Classifies a test feature vector based on its similarity to a set of reference feature vectors.
#     The class with the highest similarity (e.g., lowest distance) is chosen.
#     """
#     op_name: ClassVar[str] = "similarity_classifier"
#     description: ClassVar[str] = "Classifies a test vector by finding the most similar reference vector."
#     input_spec: ClassVar[str] = "Dict['test_vector', 'reference_vectors']"
#     output_spec: ClassVar[str] = "Dict[decision, scores]"
    
#     metric: Literal["euclidean", "manhattan", "cosine"] = Field("euclidean", description="The distance metric to use for similarity.")

#     def execute(self, x: Dict[str, npt.NDArray], **_) -> Dict[str, any]:
#         """
#         Input `x` is a dictionary with:
#         - "test_vector": The vector to classify.
#         - "reference_vectors": A dictionary of {class_name: reference_vector}.
#         """
#         if "test_vector" not in x or "reference_vectors" not in x:
#             raise ValueError("SimilarityClassifierOp requires 'test_vector' and 'reference_vectors'.")
        
#         test_vec = x["test_vector"]
#         ref_vecs = x["reference_vectors"]
        
#         distances = {}
#         for class_name, ref_vec in ref_vecs.items():
#             if self.metric == "euclidean":
#                 dist = np.linalg.norm(test_vec - ref_vec)
#             elif self.metric == "manhattan":
#                 dist = np.sum(np.abs(test_vec - ref_vec))
#             elif self.metric == "cosine":
#                 dist = 1 - np.dot(test_vec, ref_vec) / (np.linalg.norm(test_vec) * np.linalg.norm(ref_vec))
#             else:
#                 raise ValueError(f"Unknown metric: {self.metric}")
#             distances[class_name] = dist
            
#         # For distance metrics, the minimum distance means highest similarity.
#         best_match_class = min(distances, key=distances.get)
        
#         return {
#             "decision": best_match_class,
#             "scores": distances,
#             "metric": self.metric
#         }

# @register_op
# class AnomalyScorerOp(DecisionOp):
#     """
#     Calculates an anomaly score based on the distance to a reference 'healthy' state.
#     """
#     op_name: ClassVar[str] = "anomaly_scorer"
#     description: ClassVar[str] = "Calculates an anomaly score based on the distance to a healthy reference."
#     input_spec: ClassVar[str] = "Dict['test_vector', 'healthy_vector']"
#     output_spec: ClassVar[str] = "Dict[anomaly_score, distance]"
    
#     metric: Literal["euclidean", "manhattan", "mahalanobis"] = Field("euclidean", description="The distance metric to use.")

#     def execute(self, x: Dict[str, npt.NDArray], **_) -> Dict[str, any]:
#         """
#         Input `x` is a dictionary with:
#         - "test_vector": The vector to score.
#         - "healthy_vector": The reference healthy vector.
#         - "covariance" (optional): The covariance matrix, required for Mahalanobis distance.
#         """
#         if "test_vector" not in x or "healthy_vector" not in x:
#             raise ValueError("AnomalyScorerOp requires 'test_vector' and 'healthy_vector'.")
        
#         test_vec = x["test_vector"]
#         healthy_vec = x["healthy_vector"]
        
#         if self.metric == "euclidean":
#             dist = np.linalg.norm(test_vec - healthy_vec)
#         elif self.metric == "manhattan":
#             dist = np.sum(np.abs(test_vec - healthy_vec))
#         elif self.metric == "mahalanobis":
#             if "covariance" not in x:
#                 raise ValueError("Mahalanobis distance requires 'covariance' matrix in input.")
#             cov_inv = np.linalg.inv(x["covariance"])
#             delta = test_vec - healthy_vec
#             dist = np.sqrt(delta.T @ cov_inv @ delta)
#         else:
#             raise ValueError(f"Unknown metric: {self.metric}")
            
#         return {
#             "anomaly_score": float(dist),
#             "distance": float(dist),
#             "metric": self.metric
#         }

# @register_op
# class FindPeaksOp(DecisionOp):
#     """
#     Finds peaks in a 1D signal or spectrum.
#     """
#     op_name: ClassVar[str] = "find_peaks"
#     description: ClassVar[str] = "Finds peaks in a 1D signal or spectrum."
#     input_spec: ClassVar[str] = "(L,)"
#     output_spec: ClassVar[str] = "Dict[peak_indices, peak_properties]"
    
#     height: float | None = Field(None, description="Required height of peaks.")
#     distance: float | None = Field(None, description="Required minimal horizontal distance between neighboring peaks.")
#     prominence: float | None = Field(None, description="Required prominence of peaks.")

#     def execute(self, x: npt.NDArray, **_) -> Dict[str, any]:
#         """
#         Input `x` is a 1D numpy array.
#         """
#         if x.ndim != 1:
#             # If we have (B,L,C), we can try to squeeze it if B and C are 1
#             if x.ndim == 3 and x.shape[0] == 1 and x.shape[2] == 1:
#                 x = x.squeeze()
#             else:
#                 raise ValueError(f"FindPeaksOp requires a 1D array, but got shape {x.shape}.")

#         peaks, properties = signal.find_peaks(x, height=self.height, distance=self.distance, prominence=self.prominence)
        
#         # Convert numpy arrays in properties to lists for JSON serialization
#         for key, value in properties.items():
#             if isinstance(value, np.ndarray):
#                 properties[key] = value.tolist()

#         return {
#             "peak_indices": peaks.tolist(),
#             "peak_properties": properties
#         }

# @register_op
# class HarmonicAnalysisOp(DecisionOp):
#     """
#     Identifies harmonics of a fundamental frequency in a spectrum.
#     """
#     op_name: ClassVar[str] = "harmonic_analysis"
#     description: ClassVar[str] = "Identifies harmonics of a fundamental frequency in a spectrum."
#     input_spec: ClassVar[str] = "Dict['spectrum', 'peak_indices']"
#     output_spec: ClassVar[str] = "Dict[fundamental_hz, harmonics[hz, amplitude]]"
    
#     fs: float = Field(..., description="Sampling frequency of the signal.")
#     n_fft: int = Field(..., description="FFT length used to generate the spectrum.")
#     fundamental_hz: float = Field(..., description="The fundamental frequency to search for harmonics of.")
#     tolerance_hz: float = Field(5.0, description="The frequency tolerance in Hz to match harmonics.")

#     def execute(self, x: Dict[str, any], **_) -> Dict[str, any]:
#         """
#         Input `x` is a dictionary with:
#         - "spectrum": The 1D magnitude spectrum.
#         - "peak_indices": A list of indices where peaks were found.
#         """
#         if "spectrum" not in x or "peak_indices" not in x:
#             raise ValueError("HarmonicAnalysisOp requires 'spectrum' and 'peak_indices'.")
        
#         spectrum = x["spectrum"]
#         peak_indices = x["peak_indices"]
        
#         freqs = np.fft.rfftfreq(self.n_fft, d=1./self.fs)
#         peak_freqs = freqs[peak_indices]
#         peak_amplitudes = spectrum[peak_indices]
        
#         harmonics = []
#         # Start checking from the 2nd harmonic
#         for n in range(2, 10): # Check up to the 9th harmonic
#             target_freq = self.fundamental_hz * n
            
#             # Find the closest peak within the tolerance
#             freq_diffs = np.abs(peak_freqs - target_freq)
#             closest_peak_idx = np.argmin(freq_diffs)
            
#             if freq_diffs[closest_peak_idx] <= self.tolerance_hz:
#                 harmonics.append({
#                     "harmonic_n": n,
#                     "frequency_hz": peak_freqs[closest_peak_idx],
#                     "amplitude": peak_amplitudes[closest_peak_idx]
#                 })

#         return {
#             "fundamental_hz": self.fundamental_hz,
#             "harmonics_found": harmonics
#         }

# @register_op
# class ChangePointDetectionOp(DecisionOp):
#     """
#     Detects change points in a signal, indicating abrupt shifts in its properties.
#     """
#     op_name: ClassVar[str] = "change_point_detection"
#     description: ClassVar[str] = "Detects change points in a signal using the ruptures library."
#     input_spec: ClassVar[str] = "(L, C)"
#     output_spec: ClassVar[str] = "Dict[change_points]"
    
#     model: Literal["l1", "l2", "rbf"] = Field("l2", description="Model for cost function (e.g., 'l2' for changes in mean).")
#     penalty: float = Field(3.0, description="Penalty value for the number of change points.")

#     def execute(self, x: npt.NDArray, **_) -> Dict[str, any]:
#         try:
#             import ruptures as rpt
#         except ImportError:
#             raise ImportError("ruptures is not installed. Please install it with 'pip install ruptures'.")

#         if x.ndim > 2:
#             x = x.squeeze()
#         if x.ndim != 2: # Expects (L, C)
#             raise ValueError(f"ChangePointDetectionOp requires a 2D array (L, C), but got shape {x.shape}.")

#         algo = rpt.Pelt(model=self.model).fit(x)
#         result = algo.predict(pen=self.penalty)
        
#         return {
#             "change_points": result
#         }

# @register_op
# class OutlierDetectionOp(DecisionOp):
#     """
#     Detects outliers in a feature set using Isolation Forest.
#     """
#     op_name: ClassVar[str] = "outlier_detection"
#     description: ClassVar[str] = "Detects outliers in a feature set using Isolation Forest."
#     input_spec: ClassVar[str] = "(B, C')"
#     output_spec: ClassVar[str] = "Dict[is_outlier, scores]"
    
#     contamination: float = Field(0.1, description="The proportion of outliers in the data set.")

#     def execute(self, x: npt.NDArray, **_) -> Dict[str, any]:
#         from sklearn.ensemble import IsolationForest

#         if x.ndim != 2:
#             raise ValueError(f"Input for OutlierDetectionOp must be 2D (B, C'), but got {x.ndim}D.")

#         clf = IsolationForest(contamination=self.contamination, random_state=42)
#         preds = clf.fit_predict(x)
#         scores = clf.decision_function(x)
        
#         # -1 for outliers, 1 for inliers. Convert to boolean.
#         is_outlier = (preds == -1).tolist()
        
#         return {
#             "is_outlier": is_outlier,
#             "scores": scores.tolist()
#         }

# @register_op
# class KolmogorovSmirnovTestOp(DecisionOp):
#     """
#     Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
#     Tests the hypothesis that two samples are drawn from the same distribution.
#     """
#     op_name: ClassVar[str] = "ks_test"
#     description: ClassVar[str] = "Performs the Kolmogorov-Smirnov test on two samples."
#     input_spec: ClassVar[str] = "Dict['sample1', 'sample2']"
#     output_spec: ClassVar[str] = "Dict[statistic, p_value]"

#     def execute(self, x: Dict[str, npt.NDArray], **_) -> Dict[str, float]:
#         if "sample1" not in x or "sample2" not in x:
#             raise ValueError("KSTestOp requires 'sample1' and 'sample2'.")
        
#         sample1 = x["sample1"].squeeze()
#         sample2 = x["sample2"].squeeze()
        
#         statistic, p_value = scipy.stats.ks_2samp(sample1, sample2)
        
#         return {"statistic": statistic, "p_value": p_value}


# if __name__ == "__main__":
#     print("--- Testing decision_schemas.py ---")

#     # 1. Test FindPeaksOp and HarmonicAnalysisOp
#     print("\n1. Testing FindPeaksOp and HarmonicAnalysisOp...")
#     fs = 1000
#     n_fft = 2048
#     t = np.linspace(0, n_fft/fs, n_fft, endpoint=False)
    
#     # Create a dummy spectrum with a fundamental and harmonics
#     fundamental_hz = 50
#     spectrum = np.zeros(n_fft // 2 + 1)
#     freqs = np.fft.rfftfreq(n_fft, d=1./fs)
    
#     # Add fundamental and harmonics
#     for n in [1, 2, 3, 5]: # Fundamental (1) and harmonics at 2, 3, 5
#         idx = np.argmin(np.abs(freqs - fundamental_hz * n))
#         spectrum[idx] = 1.0 / n
    
#     # Find peaks
#     peaks_op = FindPeaksOp(height=0.1)
#     peaks_result = peaks_op.execute(spectrum)
#     print(f"Found {len(peaks_result['peak_indices'])} peaks.")
    
#     # Analyze harmonics
#     harmonic_op = HarmonicAnalysisOp(fs=fs, n_fft=n_fft, fundamental_hz=fundamental_hz, tolerance_hz=2.0)
#     harmonic_result = harmonic_op.execute({
#         "spectrum": spectrum,
#         "peak_indices": peaks_result["peak_indices"]
#     })
#     print("Harmonic Analysis Result:")
#     import json
#     print(json.dumps(harmonic_result, indent=2))
#     assert len(harmonic_result["harmonics_found"]) == 3 # Should find harmonics 2, 3, 5

#     # 2. Test ChangePointDetectionOp
#     print("\n2. Testing ChangePointDetectionOp...")
#     # Create a signal with a change point
#     import ruptures as rpt
#     n_samples, n_dims = 1000, 1
#     n_bkps = 1
#     signal, bkps = rpt.pw_constant(n_samples, n_dims, n_bkps, noise_std=1.0)
#     cpd_op = ChangePointDetectionOp(model="l2", penalty=3)
#     cpd_result = cpd_op.execute(signal)
#     print(f"Detected change points: {cpd_result['change_points']}")
#     assert len(cpd_result['change_points']) >= 2  # At least start and end segments

#     # 3. Test OutlierDetectionOp
#     print("\n3. Testing OutlierDetectionOp...")
#     # Create some normal data and some outliers
#     inliers = np.random.randn(100, 2)
#     outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
#     features = np.vstack([inliers, outliers])
    
#     outlier_op = OutlierDetectionOp(contamination=0.15)
#     outlier_result = outlier_op.execute(features)
#     print(f"Detected {sum(outlier_result['is_outlier'])} outliers.")
#     assert len(outlier_result['is_outlier']) == 120

#     # 4. Test KolmogorovSmirnovTestOp
#     print("\n4. Testing KolmogorovSmirnovTestOp...")
#     # Create two samples from different distributions
#     sample1 = np.random.normal(0, 1, 1000)
#     sample2 = np.random.uniform(-1, 1, 1000)
#     ks_op = KolmogorovSmirnovTestOp()
#     ks_result = ks_op.execute({"sample1": sample1, "sample2": sample2})
#     print(f"KS test p-value: {ks_result['p_value']:.4f}")
#     # p-value should be small, indicating distributions are different
#     assert ks_result['p_value'] < 0.05

#     print("\n--- decision_schemas.py tests passed! ---")
