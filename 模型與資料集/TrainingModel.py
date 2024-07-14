import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.utils import np_utils


TrainPath = ["資料集/train","資料集/陳彥呈 dataset"]  #訓練資料路徑
TestPath = ["資料集/test"]   #測試資料路徑

class_names = ['0','1','2','3','4','5','6','7','8','9','+','-','mul','div','(',')']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
CLASSNUMBER = 16
img_row, img_col = 28,28  #定義圖片大小
EPOCH = 10

def data_x_y_preprocess(datapaths):
    #對資料進行預先處理

    data_x = np.zeros((img_row,img_col,1)).reshape(1,img_row,img_col) #讀取黑白圖片
    pictureCount = 0
    data_y = []
    num_class = CLASSNUMBER  #16種符號
    for datapath in datapaths:
        for root, dirs, files in os.walk(datapath):
            #root為當前圖片之路徑
            print(root)
            for f in files:
                folder = (root.split("\\")[-1])
                label = class_names_label[folder]

                data_y.append(label)
                fullpath = os.path.join(root,f)#獲得圖片路徑
                img = Image.open(fullpath)
                img = img.convert('L')
                #img.show()
                img = img.resize((img_row,img_col)) #需取雙括號
                img = (np.array(img)/255).reshape(1,img_row,img_col) #讀取黑白圖片
                data_x = np.vstack((data_x,img))
                pictureCount += 1
    data_x = np.delete(data_x,[0],0)
    data_x=data_x.reshape(pictureCount,img_row,img_col,1)
    data_y = np_utils.to_categorical(data_y,num_class)
    print(pictureCount)

    return data_x,data_y


        
model = Sequential() 
model.add(Conv2D(32, kernel_size=(3,3),input_shape=(img_row,img_col,1),activation='relu'))#第一層卷積層
model.add(MaxPooling2D(pool_size=(2,2)))#第一層池化層
model.add(Conv2D(64, (3,3), activation='relu'))#第二層卷積層
model.add(MaxPooling2D(pool_size=(2,2)))#第二層池化層
model.add(Dropout(0.1))#隨機斷開0.1的輸入神經元
model.add(Flatten())#展開
model.add(Dropout(0.1))#隨機斷開0.1的輸入神經元

model.add(Dense(128, activation='relu'))#全連接層
model.add(Dropout(0.25))
model.add(Dense(CLASSNUMBER, activation='softmax')) #units表示要分類的種類數量

model.summary()

#訓練模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("讀取檔案：")

data_train_X,data_train_Y = data_x_y_preprocess(TrainPath)
train_history = model.fit(data_train_X, data_train_Y,
              batch_size=32, epochs=EPOCH,verbose=1,shuffle=True,
              validation_split=0.1,
          )
	       #batch_size表示一次訓練的張數
               #validation_split表示訓練時多少比例用來當Test       
               #epochs表示訓練次數
               
               
model.save(r'formula.h5')

data_test_X,data_test_Y = data_x_y_preprocess(TestPath)

prediction = model.predict(data_test_X)

# 驗證模型
score = model.evaluate(data_test_X, data_test_Y, verbose=0)
# 輸出結果
print('Test loss:', score[0])
print('Test accuracy:', score[1])

