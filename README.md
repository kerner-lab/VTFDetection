# VTFDetection

Datasets

Volcano_Data_Collected - contains 52 volcanoes and 7941 total TIF files
Merged_tables_processed.xlsx - latest pruned and processed excel files containing these volcanoes and correct volcano names

Procedure

1. Run the modified LRX code (**fork repo**) on the whole volcano dataset
		-https://github.com/adityamohan29/dora/tree/issue-lrx-mod

2. Once the results are obtained, run the classify_truth_label.py (mc.py) to separate the results into yes and no instance folders

3. Run the decision_score.py (mpred.py) to obtain the csv with their calculated scores

4. Run the binary_prediction.py with the csv obtained in step 3 as input to obtain the threshold and predict the binary value acc to the decision score


