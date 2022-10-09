分句并提取词性
get_split.py

python3 get_split.py --tag XXXX --input XXXX

tag为输出文件的名称，输出为XXXX.splitstc.jsonl
input为输入文件名称，格式：（id）\t 句子

生成人造伪数据
create.py

python3 create.py --tag XXXX --input XXXX

tag为输出文件的名称，输出为XXXX-s{seed}.src和XXXX-s{seed}.trg
input为输入文件名称，是由get_split.py脚本输出得jsonl文件
