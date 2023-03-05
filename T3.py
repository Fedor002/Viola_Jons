import cv2

#Считывание фотографии
img_rgb = cv2.imread('Stock/Stock5.jpg')
#Перекидываю код метода в объект
face_casd = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Обнаружение лиц с помощью детектора Виолы-Джонса
faces = face_casd.detectMultiScale(img_rgb, scaleFactor=1.5,minNeighbors=5, minSize=(20,20))
for (x,y,w,h) in faces:
    #Обводка по координатам, после использования метода
    cv2.rectangle(img_rgb, (x,y),(x+w, y+h),(0, 0, 0),2)

#Вывести результат
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()