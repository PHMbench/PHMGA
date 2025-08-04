import argparse
import importlib
import os

def main():
    """
    Main entry point for running PHM analysis cases.
    Dynamically loads and runs a case module based on command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run PHM analysis cases.")
    parser.add_argument(
        "case_name", 
        type=str, 
        help="The name of the case to run (e.g., 'case1')."
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to the configuration file. Defaults to 'config/<case_name>.yaml'."
    )
    args = parser.parse_args()

    case_name = args.case_name
    config_path = args.config or f"config/{case_name}.yaml"

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at '{config_path}'")
        return

    try:
        # Dynamically import the case module
        case_module = importlib.import_module(f"src.cases.{case_name}")
        
        # Run the case
        case_module.run_case(config_path)
        
    except ImportError:
        print(f"Error: Case '{case_name}' not found. Make sure 'src/cases/{case_name}.py' exists.")
    except Exception as e:
        print(f"An error occurred while running case '{case_name}': {e}")

if __name__ == "__main__":
    main()
