# DS_SDP_CNN_MIL
基于远程监督，使用SDP特征，CNN模型，多实例学习（选一个实例）



## 注意事项
1、在NLTK包的stanford.py文件中，修改了第356行，添加了cell_separator='\t'

2、从训练数据中删除两条数据，具体删除内容，保存在train_temp.txt中

## 备注
FilterNYT 数据集 label 类别总数为27 0-26
最大长度：49

## 说明
1. original/FilterNYT中
   1. dict_temp.txt用于将数据中SDP中的"a b"转化为"a_b"
   2. dict_new.txt为新的词典文件，除新增一些词之外，还**将" "转化为"_"，将"\/"转化为"/"**

2. train2word_result/FilterNYT中
   1. test0.txt与test1.txt是由test.txt一分为二得来；同理，train0.txt、train1.txt和train.txt
   2. test_temp.txt和train_temp.txt为部分生成SDP时出错的实例数据
   3. train_process.txt为从train.txt中移除train_temp.txt中数据剩余的实例数据

3. sen2sdp_result/FilterNYT中为批量生成的SDP数据（有错误，<u>**label只有一个**</u>），及生成SDP时的错误日志

4. sen2sdp_result_final/FilterNYT中
   1. train_sdp.pickle和test_sdp.pickle（<u>**label只有一个**</u>）为处理错误日志（批量生成SDP）后的结果**（!!!删除了两条实例数据)**
   2. 对于train_sdp_final.pickle和test_sdp_final.pickle（**<u>label只有一个</u>**），由于生成SDP时自动将" "(ascii为160)转化为" "(ascii为32)，因此进一步处理，将" "(ascii为32，且由160转化而来)转化为"_"
   3. test_sdp_final_final.pickle和train_sdp_final_final.pickle，**<u>label由一个变为四个</u>**