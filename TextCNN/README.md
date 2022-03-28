将tx_data中load_CNN_data中文件的地址改为需要检测的文件地址

检测文件的格式需和data中给出的excel文件格式一致

如果需要检测的字符串长度较短，在CNNword.py中修改max_length，与需要检测的字符串长度相匹配

运行CNNword.py可得到结果：python CNNword.py（运行如提示缺少的包直接pip install即可）

环境要求：python>=3.7    tensorflow>=2.3.0
