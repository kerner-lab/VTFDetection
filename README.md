# VTFDetection

Datasets

Volcano_Data_Collected - contains 52 volcanoes and 7941 total TIF files
Merged_tables_processed.xlsx - latest pruned and processed excel files containing these volcanoes and correct volcano names

Procedure

Run the modified LRX code (**fork repo**) on the whole volcano dataset
		-https://github.com/adityamohan29/dora/tree/issue-lrx-mod

Inorder to execute this more efficiently, split folder into separate subfolders and submit jobs parallely

 Once the results are obtained, run the classify_truth_label.py (mc.py) to separate the results into yes and no instance folders

Run the decision_score.py (mpred.py) to obtain the csv with their calculated scores

Run the binary_prediction.py with the csv obtained in step 3 as input to obtain the threshold and predict the binary value acc to the decision score


