#python3 test.py
# cd fastapi
# sh start.sh
apt-get install graphviz -y
wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
unzip ngrok-stable-linux-amd64.zip
python3 test_preprocess.py
# python3 train.py --config_path "train.yaml"
python3 inference.py --model_path "/content/machine_translate/outputs/model_Lstm_Embd.h5" --text "new jersey is sometimes quiet"
