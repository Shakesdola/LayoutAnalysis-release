# LayoutAnalysis-release

For Challenge 3.

Colab: https://colab.research.google.com/drive/1DPLxNYHCsES5eA0Nfj7gy8LpaUouRXMb

Google drive with data: https://drive.google.com/drive/folders/13eXCmo8-EKIOvTUKDMxOoj2ZYhuQ_X0I?usp=drive_link 

【Part A】 For decorations and text area.

Preprocessing from groundtruth, decoration:(255,0,0); text_area:(255,255,0). 【preprocessing\decoration_text_area.py】

Generate label files(txt) class 0: decoration class 1:text_area format: <class_index> ... (they are normalized (0~1) by X/W, Y/H. 【preprocessing\mask_to_polygon.py】

File Folder Structure

train model [model = YOLO('yolov8n-seg.pt')] validate model 【Layout_analysis_training.ipynb】

Prediction 【predict_and_mask.py】

Evaluation 【evaluation_by_mask.py】
