from __future__ import annotations
import uuid
import yaml
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Disable LangSmith for cleaner logs
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""

from src.phm_outer_graph import build_builder_graph, build_executor_graph
from src.utils import initialize_state, save_state, load_state, generate_final_report
# from src.utils.visualization import visualize_dag_feature_evolution_umap
from src.agents.reflect_agent import get_dag_depth

def run_case(config_path: str):
    """
    Runs a PHM Graph Agent analysis case based on a configuration file.
    
    Args:
        config_path (str): Path to the YAML configuration file
        
    Returns:
        PHMState: The final state after DAG construction
    """
    try:
        # 1. Load configuration from YAML file
        print(f"--- Loading configuration from {config_path} ---")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate required fields
        required_fields = ['name', 'state_save_path', 'user_instruction', 'metadata_path', 'h5_path', 'ref_ids', 'test_ids']
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

    state_save_path = config['state_save_path']
    builder_cfg = config.get('builder', {})
    min_depth = builder_cfg.get('min_depth', 0)
    max_depth = builder_cfg.get('max_depth', float('inf'))

    # --- Check for existing state ---
    if os.path.exists(state_save_path):
        print(f"\n--- Found existing state file at {state_save_path}. Skipping builder workflow. ---")
        built_state = load_state(state_save_path)
        if built_state is None:
            print("Failed to load state. Exiting.")
            return
    else:
        print(f"\n--- No existing state file found. Starting builder workflow. ---")
        try:
            # --- Part 0: Initialization ---
            print("\n--- [Part 0] Initializing State ---")
            
            # Check if data files exist
            if not os.path.exists(config['metadata_path']):
                print(f"⚠️  Warning: Metadata file not found at {config['metadata_path']}")
                print("    Proceeding with mock data for demo purposes...")
                
            if not os.path.exists(config['h5_path']):
                print(f"⚠️  Warning: H5 file not found at {config['h5_path']}")
                print("    Proceeding with mock data for demo purposes...")
            
            initial_phm_state = initialize_state(
                user_instruction=config['user_instruction'],
                metadata_path=config['metadata_path'],
                h5_path=config['h5_path'],
                ref_ids=config['ref_ids'],
                test_ids=config['test_ids'],
                case_name=config['name']
            )
            
            if initial_phm_state is None:
                raise RuntimeError("Failed to initialize PHM state")
                
        except Exception as e:
            print(f"❌ Error during initialization: {e}")
            return None

        # --- Part 1: Run DAG Builder Workflow ---
        print("\n--- [Part 1] Starting DAG Builder Workflow ---")
        builder_app = build_builder_graph()

        built_state = initial_phm_state.model_copy(deep=True)
        iteration = 0

        while True:
            iteration += 1
            print(f"\n--- Builder Iteration {iteration} ---")
            thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            for event in builder_app.stream(built_state, config=thread_config):
                for node_name, state_update in event.items():
                    print(f"--- Builder Node Executed: {node_name} ---")
                    if state_update is not None:
                        for key, value in state_update.items():
                            setattr(built_state, key, value)

            depth = get_dag_depth(built_state.dag_state)
            print(f"Current DAG depth: {depth}")

            if depth >= max_depth:
                print(f"Reached max depth {max_depth}. Stopping builder.")
                break

            if depth < min_depth:
                print(f"Depth {depth} below min_depth {min_depth}. Continuing regardless of reflection.")
                built_state.needs_revision = True

            if not built_state.needs_revision:
                print("Reflect agent indicated to stop.")
                break

        print("\n--- [Part 1] DAG Builder Workflow Finished ---")
        if not built_state:
            print("Builder workflow failed to produce a final state.")
            return
            
        print(f"Final leaves of the built DAG: {built_state.dag_state.leaves}")
        print(f"Total nodes in DAG: {len(built_state.dag_state.nodes)}")
        print(f"Errors during build: {built_state.dag_state.error_log}")

        # --- Save the built state ---
        save_state(built_state, state_save_path)

    # At this point, `built_state` is guaranteed to be a valid state object,
    # either loaded from file or newly created.
    
    print(f"\n--- [Case Complete] DAG Construction Finished ---")
    print(f"✅ Successfully built DAG with {len(built_state.dag_state.nodes)} nodes")
    print(f"✅ Final DAG depth: {get_dag_depth(built_state.dag_state)}")
    print(f"✅ State saved to: {state_save_path}")
    
    if built_state.dag_state.error_log:
        print(f"⚠️  Warnings: {len(built_state.dag_state.error_log)} issues logged")
    
    return built_state

if __name__ == "__main__":
    """
    Direct execution for testing. Run different experimental cases:
    - case_exp2: 5-state bearing fault diagnosis  
    - case_exp2.5: Alternative 5-state configuration
    - case_exp_ottawa: 3-state variable speed dataset
    """
    import sys
    
    # Default to ottawa case if no argument provided
    if len(sys.argv) > 1:
        case_name = sys.argv[1]
    else:
        case_name = "case_exp_ottawa"
    
    config_file = f"config/{case_name}.yaml"
    
    print(f"🚀 Running PHM Graph Agent Demo: {case_name}")
    print(f"📋 Configuration: {config_file}")
    print("="*50)
    
    result = run_case(config_file)
    
    if result:
        print("="*50)
        print("✅ Demo completed successfully!")
    else:
        print("="*50)
        print("❌ Demo failed. Check error messages above.")
        sys.exit(1)
