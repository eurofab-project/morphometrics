# morphometrics
Morphometric modules of the EuroFab project

# To run:

1. Clone this repository.

2. Run `pixi install`, then `pixi run build` and optionally `pixi run tests` . These commands setup the enviroment and all the required packages.

    - Alternatively you can manually install the conda-forge dependencies, but you have to still run the pixi build and tests commands:
           `pixi add momepy umap-learn fast_hdbscan jupyterlab pyarrow matplotlib lonboard folium mapclassify datashader  dask pip sidecar glasbey scikit-image colorcet pandas holoviews bokeh=3.1 esda pytest hdbscan`

3. To run jupyter use either `pixi run jupyter lab` or pass extra arguments like `pixi run jupyter lab --port 8888`.

4. To run the analysis on the whole dataset - first, make sure you have the correct folder structure and the target label data in place. Then, run:

    - `notebooks/download_data.ipynb` to download all the Microsoft building footprints, split them into regions, and then download the overture streets.
    - `notebooks/process_regions.ipynb` to run the entire processing pipeline from building, street preprocessing, element generation and characters calculations.
    - `notebooks/eurofab_model.ipynb` to train and test the model.
