# VTFDetection

Datasets

1. Volcano_Data_Collected - contains 52 volcanoes and 7941 total TIF files (Box folder)
2. Merged_tables_processed.xlsx - latest pruned and processed excel files containing these volcanoes and correct volcano names

Procedure

1. Install DORA on your local machine following the steps from the vtf-dora branch in this forked repo
		-[https://github.com/nasaharvest/dora/tree/vtf-dora/dora_exp_pipeline](https://github.com/nasaharvest/dora/tree/vtf-dora/dora_exp_pipeline)

2. Navigate to directory inference_pipeline with `cd inference_pipeline/`

3. Once the results are obtained, run `pip install -r requirements.txt` to install all requirements for this app

4. Run `vtf_app.py` with your specified input directory and output file for **one volcano** at a time. For example: 
`python3 vtf_app.py --input="/data/Pacaya/" --output="Pacaya_res.csv"`
You can run `python3 vtf_app.py --help` if you want to specify any additional arguments

Note:
Your directory structure should be of the following format - {Volcano}/"all"/{ASTER file name}, for example:
Pacaya/all/AST_08_00302282019042806_20230914075707_28909.SurfaceKineticTemperature.KineticTemperature.tif

5. Run `run_all.py` with your specified input directory of a **volcanic region**. For example: 
`python3 run_all.py --region="Kuril_Islands"`
You can run `python3 run_all.py --help` if you want help specifying arguments
