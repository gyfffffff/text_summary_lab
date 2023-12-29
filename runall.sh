python main.py --model lstm2lstm --version v1 --batch_size 32 --lr 0.1 --num_layers 2
python main.py --model lstmatt --version v1 --batch_size 28 --lr 0.05 
python main.py --model gruatt --version v1 --batch_size 28 --lr 0.01 
python main.py --model transformer --version v1 --batch_size 64 --device cpu 
python main.py --model t5-base --version v1 --batch_size 8 --device cpu
python main.py --model t5-medical --version v1 --batch_size 8 
