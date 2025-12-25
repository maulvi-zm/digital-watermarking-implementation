"""Main script to run complete experiment pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import traceback
import config
from train import main as train_main
from verify import main as verify_main
from test_robustness import main as robustness_main
from evaluate import main as evaluate_main
from src.utils import save_results


def main():
    """Run complete experiment pipeline."""
    print("="*80)
    print("COMPLETE EXPERIMENT PIPELINE")
    print("="*80)
    
    results = {}
    
    try:
        # Step 1: Training
        print("\n" + "="*80)
        print("STEP 1: TRAINING")
        print("="*80)
        train_main()
        
        # Step 2: Verification
        print("\n" + "="*80)
        print("STEP 2: VERIFICATION")
        print("="*80)
        verification_results = verify_main()
        results["verification"] = verification_results
        
        # Step 3: Robustness Testing
        print("\n" + "="*80)
        print("STEP 3: ROBUSTNESS TESTING")
        print("="*80)
        robustness_results = robustness_main()
        results["robustness"] = robustness_results
        
        # Step 4: Comprehensive Evaluation
        print("\n" + "="*80)
        print("STEP 4: COMPREHENSIVE EVALUATION")
        print("="*80)
        evaluation_results = evaluate_main()
        
        # Merge all results
        results.update(evaluation_results)
        
        # Save complete results
        print("\n" + "="*80)
        print("SAVING COMPLETE RESULTS")
        print("="*80)
        save_results(
            results,
            config.PATHS["results_json"],
            config.PATHS["results_txt"]
        )
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        print(f"\nResults saved to:")
        print(f"  JSON: {config.PATHS['results_json']}")
        print(f"  Text: {config.PATHS['results_txt']}")
        
        return results
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

