# -*- coding: utf-8 -*-
import numpy as np
from PIL import Image
from keras.models import load_model
import CutPicture

class_names = ['0','1','2','3','4','5','6','7','8','9','+','-','mul','div','(',')']
class_names_label = {class_name:i for i, class_name in enumerate(class_names)}
CLASSNUMBER = 16
img_row, img_col = 28,28  #定義圖片大小
EPOCH = 10
reduce_retracing=True


def preprocess(path):
  #輸入數學算式照片進行分割以及資料預處理  
  #對資料進行預先處理    
  data_x = np.zeros((img_row,img_col,1)).reshape(1,img_row,img_col) #讀取黑白圖片

    
  img = Image.open(path)
  img = CutPicture.accessBinary(img)
  #行掃描
  hori_vals = np.sum(img, axis=1) #得到橫軸和的陣列用以判斷是否為邊界
  hori_points = CutPicture.extractPeek(hori_vals,5,100) #得到行座標
  #filepath = path.split(".")[0]
  #os.mkdir(filepath)
  #根據每一行來掃描列
  counter = 0
  for hori_point in hori_points:
    extractImg = img.crop((0, hori_point[0], img.width, hori_point[1])) #提取橫切割區域
    vec_vals = np.sum(extractImg,axis=0) #得到縱軸和之陣列用以判斷邊界
    vec_points = CutPicture.extractPeek(vec_vals, min_rect=10)
    #extractImg.save(filepath+ '/' +str(counter)+'.png')
    
    for vec_point in vec_points:
      IndividualImg = extractImg.crop((vec_point[0], 0, vec_point[1], extractImg.height))#依左上角以及右下角座標提取
      hori_valsI = np.sum(IndividualImg, axis=1) #得到橫軸和的陣列用以判斷是否為邊界
      hori_pointI = CutPicture.SignalExtract(hori_valsI,10,20) #得到行座標
      IndividualImgI = IndividualImg.crop((0, hori_pointI[0] , IndividualImg.width, hori_pointI[1]))#依左上角以及右下角座標提取
      if(IndividualImg.width<100 and hori_pointI[1]-hori_pointI[0] < 100):
          continue
      #過濾雜訊
      whiteBlock = np.sum(IndividualImg)/255
      if whiteBlock < 1000:
          continue
      if IndividualImg.width > 270:
          IndividualImgI = IndividualImgI.resize((270,IndividualImgI.height))
      if hori_pointI[1]-hori_pointI[0]>270:
          IndividualImgI = IndividualImgI.resize((IndividualImgI.width,270))
      IndividualImgI = CutPicture.patch(IndividualImgI,300)
      IndividualImgI = IndividualImgI.convert('L') #轉灰階，高度變成1
      IndividualImgI = IndividualImgI.resize((img_row,img_col)) #需取雙括號
      IndividualImgI = (np.array(IndividualImgI)/255).reshape(1,img_row,img_col) #讀取黑白圖片
      data_x = np.vstack((data_x,IndividualImgI))
      #IndividualImgI.save(filepath + '/' + filename+"_"+str(counter)+".png")
      counter+=1
      
  data_x = np.delete(data_x,[0],0)
  data_x=data_x.reshape(counter,img_row,img_col,1)
  return data_x

def getResult(index):
  formula = ""
  for i in index:
    if class_names[i] == 'div':
        formula += '/'
    elif class_names[i] == 'mul':
        formula += '*'
    else:
        formula += class_names[i]
  return formula

def GetFormula(path):

  model = load_model("formula.h5")#呼叫訓練出的.h5檔案
  data_test_X = preprocess(path)
  prediction = model.predict(data_test_X, verbose=0)
  index = np.argmax(prediction, axis=1)
  formula = getResult(index)
  return formula


if __name__ == '__main__':
    path = r"6.jpg"
    print(GetFormula(path))
