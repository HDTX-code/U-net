# source /Home/atr2/homefun/zhf/SS-Former/train_sh/train_mutil_GPU.sh
# source /devdata/home/homefun/SS-Former/train_sh/txt.sh
conda activate homefun
cd /devdata/home/homefun/SS-Former/
#cd /Home/atr2/homefun/zhf/SS-Former
python csv_to_txt.py  --data_path /devdata/home/homefun/DATA/UW/ \
                      --csv_path /devdata/home/homefun/DATA/UW/data_csv.csv