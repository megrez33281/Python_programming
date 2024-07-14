#圖片切割=>行列掃描
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter

class_names = ['0','1','2','3','4','5','6','7','8','9','+','-','mul','div','(',')']
def accessPiexl(img):
  height,width   = img.size #得到長與寬 
  img = img.convert('L') 
  for i in range(height):
    for j in range(width):
      img.putpixel((i,j), 255 - img.getpixel((i,j)))
  return img

def accessBinary(img, threshold=127): #將圖片二值化
  img = accessPiexl(img)
  #定義膨脹內核
  kernel_size = 3
  kernel = ImageFilter.MaxFilter(kernel_size)
  #進行膨脹操作
  img = img.filter(kernel)
  #定義閾值
  threshold_value = 127
  #閾值化操作
  img = img.point(lambda p: p > threshold_value and 255)
  return img

def extractPeek(array_vals, min_vals=5, min_rect=20):
  #進行邊界判斷
  #min_vals：每行/列的相加值之邊界判斷
  extrackPoints = []
  startPoint = None
  endPoint = None
  for i,point in enumerate(array_vals):
    if point>min_vals and startPoint == None:
      startPoint = i
    elif point<min_vals and startPoint != None:
      endPoint = i

    if startPoint != None and endPoint != None:
      if endPoint-startPoint >= min_rect:
          extrackPoints.append((startPoint, endPoint)) 
      startPoint = None
      endPoint = None

  return extrackPoints

def SignalExtract(array_vals, min_vals=5, min_rect=20):
  #進行單個圖片的最後一次橫切
  #min_vals：每行/列的相加值之邊界判斷
  startPoint = None
  endPoint = None
  for i,point in enumerate(array_vals):
    if point>min_vals and startPoint == None:
      startPoint = i
    elif point>min_vals and startPoint != None: 
      endPoint = i #找到最後一個大於min_vals之位置
    
  if endPoint == None and startPoint != None:
      endPoint = len(array_vals)-1  #當到達底部且沒找到邊界時將底部視為邊界

  return [startPoint, endPoint]

def findBorderOneLine(path):
  img = Image.open(path)
  #img = accessBinary(img) #注意讀取之圖片若已經為黑底白字，則不需要再呼叫

  basename = os.path.basename(path) # basename - example.py
  filename = os.path.splitext(basename)[0]  # filename - example 
  filepath = path.split(".")[0]
  
  #根據每一行來掃描列
  counter = 0
  vec_vals = np.sum(img,axis=0) #得到縱軸和之陣列用以判斷邊界
  vec_points = extractPeek(vec_vals)
  os.mkdir(filepath)

  for vec_point in vec_points:
    IndividualImg = img.crop((vec_point[0], 0, vec_point[1], img.height))#依左上角以及右下角座標提取
    hori_valsI = np.sum(IndividualImg, axis=1) #得到橫軸和的陣列用以判斷是否為邊界
    hori_pointI = SignalExtract(hori_valsI,10,20) #得到行座標
    IndividualImgI = IndividualImg.crop((0, hori_pointI[0] , IndividualImg.width, hori_pointI[1]))#依左上角以及右下角座標提取
    if(IndividualImg.width<100 and hori_pointI[1]-hori_pointI[0] < 100):
        continue

    IndividualImgI = patch(IndividualImgI,300)
    IndividualImgI.save(filepath + '/' + filename+"_"+str(counter)+".png")
    counter+=1





def patch(image,size):
  #將圖片擴充到對應size
  new_image = Image.new("RGB", (size, size), color="black")

  #將原圖放在新圖片中心
  x_offset = (new_image.width - image.width) // 2
  y_offset = (new_image.height - image.height) // 2
  new_image.paste(image, (x_offset, y_offset))
  return new_image

def findBorderHistogram(path):
  img = Image.open(path)
  img = accessBinary(img)
  
  #行掃描
  hori_vals = np.sum(img, axis=1) #得到橫軸和的陣列用以判斷是否為邊界
  hori_points = extractPeek(hori_vals,5,100) #得到行座標

  basename = os.path.basename(path) # basename - example.py
  filename = os.path.splitext(basename)[0]  # filename - example 
  filepath = path.split(".")[0]
  os.mkdir(filepath)
  #img.save(filepath + "/00.png")

  #根據每一行來掃描列
  counter = 0
  for hori_point in hori_points:
    extractImg = img.crop((0, hori_point[0], img.width, hori_point[1])) #提取橫切割區域
    vec_vals = np.sum(extractImg,axis=0) #得到縱軸和之陣列用以判斷邊界
    vec_points = extractPeek(vec_vals, min_rect=10)
    
    for vec_point in vec_points:
      IndividualImg = extractImg.crop((vec_point[0], 0, vec_point[1], extractImg.height))#依左上角以及右下角座標提取
      hori_valsI = np.sum(IndividualImg, axis=1) #得到橫軸和的陣列用以判斷是否為邊界
      hori_pointI = SignalExtract(hori_valsI,10,20) #得到行座標
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
          
      IndividualImgI = patch(IndividualImgI,300)
      IndividualImgI.save(filepath + '/' + filename+"_I_"+str(counter)+".png")
      counter+=1
      




if __name__ == '__main__':
  path = "1.jpg"
  findBorderHistogram(path)


