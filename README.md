# morphometrics
Morphometric modules of the EuroFab project

# To run:

1. Clone this repository.

2. Run `pixi install`, then `pixi run build` and optionally `pixi run tests` . These commands setup the enviroment and all the required packages.

    - Alternatively you can manually install the conda-forge dependencies, but you have to still run the pixi build and tests commands:
           `pixi add momepy umap-learn fast_hdbscan jupyterlab pyarrow matplotlib lonboard folium mapclassify datashader  dask pip sidecar glasbey scikit-image colorcet pandas holoviews bokeh=3.1 esda pytest hdbscan`

3. To run jupyter use either `pixi run jupyter lab` or pass extra arguments like `pixi run jupyter lab --port 8888`.

4. To run the analysis on the whole dataset - first, make sure you have the correct folder structure and the target label data in place. Then, run:

    - `notebooks/1_download_ms_data.ipynb` or `notebooks/1_download_ov_data.ipynb` to download all the Microsoft or Overture building footprints, split them into regions, and then download the overture streets. And then assign all data to the regions.
    - `notebooks/2_process_regions.ipynb` to run the entire processing pipeline from building, street preprocessing, element generation, character and spatial lag calculations.
    - `notebooks/3_assign_targets.ipynb` to assign target labels to each generated element.
    - `notebooks/4_assign_buildings_to_h3.ipynb` to assign buildings to h3 hexagons, in preparation for train/test splitting.
    - `notebooks/5_train_test_split.ipynb` to split the data into 5 training and test sets, based on the country of origin.
    - `notebooks/6_single_model_training.ipynb` to train and test a specific model. Alternatively, you can run the `generate_predictions.py` file.
    - `notebooks/7_figures_and_tables.ipynb` to generate paper/report figures and tables.
    - `notebooks/interactive_exploration.ipynb` to interactively explore the morphometric characters.
