
save_path=/Users/huangdayi/project/github/AgentSearch/data

python download.py --save_path $save_path

 cat $save_path/part_* > e5_Flat.index
