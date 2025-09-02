# PN23_DAC
This repo is for anomaly detection for aluminium tiles.

## 🔧 Environments
Create a new conda environment and install required packages.
```
conda create -n uninet_env python=3.9.7
conda activate uninet_env
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113
```

## 🚀 Run Experiments
<summary><strong> 1. Unsupervised AD</strong></summary>

Run the following command for industrial domain, such as MVTec AD dataset:
```
python main.py --setting oc --dataset "MVTecAD"
```
<br>
<summary><strong> 2. Testing</strong></summary>
    
With training on your own or saving the weight file we upload, you can test the model using the following command:
```
python main.py --setting oc --dataset "MVTecAD" --load_ckpts
```
