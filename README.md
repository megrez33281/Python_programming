# Python_programming
Python程式設計_專題_手寫算式圖形辨識
主要功能為可以用手機拍攝一條簡單的四則運算的算式，傳送給line/telegram聊天機器人。
line/telegram聊天機器人會將圖片傳送到後端。
後端會先將圖片中的算式切割出來，再將所有的運算子以及運算元切割出來。
切割出來的運算子與運算元會傳送到圖形識別的模型中分析。
根據分析出的結果計算出答案後再藉由line/telegram聊天機器人傳回。

## 說明
    
![1](img/1.png)
![2](img/2.png)
![3](img/3.png)
![4](img/4.png)
![5](img/5.png)
![6](img/6.png)
![7](img/7.png)
![8](img/8.png)
![9](img/9.png)
![10](img/10.png)
![11](img/11.png)
詳細內容請參見Group5.pdf