# LayoutAnalysis-release

For Challenge 3.

Colab: https://colab.research.google.com/drive/1DPLxNYHCsES5eA0Nfj7gy8LpaUouRXMb

Google drive with data: https://drive.google.com/drive/folders/13eXCmo8-EKIOvTUKDMxOoj2ZYhuQ_X0I?usp=drive_link 

【Part A】 For decorations and text area.

Preprocessing From groundtruth, decoration:(255,0,0); text_area:(255,255,0).
↓
Generate label files(txt) class 0: decoration class 1:text_area format: <class_index> ... (they are normalized (0~1) by X/W, Y/H.
↓
File Folder Structure
↓
train model [model = YOLO('yolov8n-seg.pt')] validate model
↓
Prediction
↓
Evaluation
