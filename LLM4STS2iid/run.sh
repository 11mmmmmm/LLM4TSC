
python main.py --model_name 'gpt2' --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
echo "end"
python main.py --model_name 'gpt2-medium' --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
echo "end"
python main.py --model_name 'gpt2-large' --gpu 0 --win_size 256 --pred_len 1 --batch_size 1
echo "end"

