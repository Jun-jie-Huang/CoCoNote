import os

path_nb_in = "../raw_notebooks"
# path_wrangling_cells_out = "../CoCoMine-saved_results"
path_wrangling_cells_out = "./saved_wrangling_cells"
path_final_examples = "../dataset-CSIO"

# Step 1: identify data-wrangling code cells from raw notebooks
os.system(f"python identify_my_pipeline_ok.py {path_nb_in} {path_wrangling_cells_out}")

# Step 2: create the code generation examples
os.system(f"python create_code_generation_examples.py {path_wrangling_cells_out} {path_final_examples}")

# The final examples is saved to ${path_final_examples}/task_data_dependency_qualified.pkl
