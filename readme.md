# LLM4DTS
1、用于文本数据+算术编码压缩
2、环境：conda activate transformers
3、运行方式
cd LLM4DTS
python main.py
--model_name 表示模型'Llama'是Llama7-hf,其他分别是Llama13-hf gpt2 gpt2-medium Chatglm gpt2-large xlnet-base xlnet-large
--text_file 表示要压缩的文本路径
--win_size --pred_len --batch_size 表示三个参数
--encode_decode 表示压缩或解压缩，0：压缩，1：解压
--gpu 表示使用哪个gpu

4、结果保存在../Results1/
5、注意
1）这里的win_size指上下文有win_size个token


# LLM4STS
1、基于LLM的整型/浮点型数据编码压缩（使用算术编码）
2、环境：conda activate transformers
3、运行方式
cd LLM4STS
python main.py
--model_name 表示模型'Llama'是Llama7-hf,其他分别是Llama13-hf gpt2 gpt2-medium gpt2-large
--text_file 表示待压缩文件路径，注意要先将数值转换为文本再压缩，示例见../Dataset/Float/text_gps
--win_size --pred_len --batch_size 表示三个参数
--encode_decode 表示压缩或解压缩，0：压缩，1：解压
--gpu 表示使用哪个gpu

4、结果保存在../Results1/
5、注意：
1）gpt系列预测时默认是sp模式，即 单个数字+空格 进行tokenize
2）这里的win_size指上下文有win_size个数字



# Timerxl4NTS
1、基于Timerxl的整型/浮点型数据预测（不含压缩算法，只保存残差结果）
2、环境：conda activate timer
3、运行方式
cd Timerxl4NTS
python main.py
--model_name 表示模型'Timerxl'
--data_file 表示待压缩文件路径，直接使用数值预测，示例见../Dataset/Cyber-Vehicle/aaa.csv
--win_size --pred_len --batch_size 表示三个参数
--encode_decode 表示压缩或解压缩，0：压缩，1：解压
--gpu 表示使用哪个gpu

4、结果保存在../Results1/
5、注意：
1）这里的win_size指上下文有win_size个数值
2）结果里面的residuals_list里保存的是不包含前win_size的残差（因为第一个窗口的残差为0）



# TimesFM4NTS
1、基于Timerxl的整型/浮点型数据预测（不含压缩算法，只保存残差结果）
2、环境：conda activate nts
3、运行方式
cd TimesFM4NTS
python main.py
--model_name 表示模型'TimesFM'
--data_file 表示待压缩文件路径，直接使用数值预测，示例见../Dataset/Cyber-Vehicle/aaa.csv
--win_size --pred_len --batch_size 表示三个参数
--encode_decode 表示压缩或解压缩，0：压缩，1：解压
--gpu 表示使用哪个gpu

4、结果保存在../Results1/
5、注意：
1）这里的win_size指上下文有win_size个数值
2）结果里面的residuals_list里保存的是不包含前win_size的残差（因为第一个窗口的残差为0）



