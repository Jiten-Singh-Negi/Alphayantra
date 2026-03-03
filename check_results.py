import joblib
from pathlib import Path

def show_results(model_name="default"):
    path = Path(f"models/saved/{model_name}/metrics.pkl")
    
    if not path.exists():
        print(f"❌ Could not find {path}.")
        print("The laptop may have died before the final ensemble save.")
        return

    metrics = joblib.load(path)
    
    print("\n" + "="*50)
    print(f"✅ SAVED MODEL METRICS ('{model_name}')")
    print("="*50)
    
    # Format and print the dictionary beautifully
    for key, value in metrics.items():
        # Capitalize and clean up the keys for reading
        clean_key = key.replace("_", " ").title()
        print(f"{clean_key:<25}: {value}")
    print("="*50)

if __name__ == "__main__":
    show_results()