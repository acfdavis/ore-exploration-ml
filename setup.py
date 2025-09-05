import os
import sys
import subprocess
from pathlib import Path

def run_notebook_as_script(notebook_path):
    """
    Executes a Jupyter notebook as a Python script.
    """
    print(f"--- Running {notebook_path.name} ---")
    try:
        # Ensure the project root is in the python path
        project_root = Path(__file__).parent
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

        subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "script", "--execute", str(notebook_path)],
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
        print(f"--- Finished {notebook_path.name} ---")
    except subprocess.CalledProcessError as e:
        print(f"!!! Error running {notebook_path.name} !!!")
        print(e.stdout)
        print(e.stderr)
        raise

def main():
    """
    Runs all the necessary notebooks in order to set up the data for the app.
    """
    print("Starting data setup process...")
    
    notebooks_to_run = [
        "notebooks/01_data_download.ipynb",
        "notebooks/01b_sgmc_fetch.ipynb",
        "notebooks/01c_labels_from_mrds.ipynb",
        "notebooks/02_feature_engineering.ipynb",
        "notebooks/02b_geology_features.ipynb",
        "notebooks/02c_gravity_features.ipynb",
        "notebooks/02d_geochem_features.ipynb",
        "notebooks/2e_magnetic_features.ipynb",
        "notebooks/03_modeling_and_maps.ipynb",
        "notebooks/04_bayesian_logreg.ipynb",
    ]
    
    for notebook in notebooks_to_run:
        notebook_path = Path(notebook)
        if notebook_path.exists():
            run_notebook_as_script(notebook_path)
        else:
            print(f"Warning: Notebook not found at {notebook_path}")

    print("Data setup process complete.")

if __name__ == "__main__":
    main()
