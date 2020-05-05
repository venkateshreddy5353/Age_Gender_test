from flask import Flask,send_file,render_template
from flask import request, jsonify
import base64
import os
import numpy as np
import cv2
from io import StringIO
from flask_cors import CORS
from flask_ngrok import run_with_ngrok
import shutil
count=0
main_source=0
app=Flask(__name__)
CORS(app)
dirpath=os.getcwd()
#run_with_ngrok(app)

@app.route("/", methods=['POST','GET'])
def index():

    if request.method == 'POST':
        data=request.stream.read()
        data=str(data).split(',')[1] #.encode()
        global count
        count+=1
        img = base64.b64decode(data)
        npimg = np.frombuffer(img, dtype=np.uint8)
        source = cv2.imdecode(npimg, 1)
        if count <4:
            return ""
        elif count==1:
            os.mkdir('/images')
        elif count==100:
            shutil.rmtree("/images")
            os.mkdir('/images')
        cv2.imwrite('/images/hello1'+str(count)+'.jpg', source)
        face_cascade = cv2.CascadeClassifier(dirpath+'/haarcascade_frontalface_default.xml')
        face_img = source.copy()
        face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2)
        for (x,y,w,h) in face_rects:
            cv2.rectangle(face_img, (x,y), (x+w,y+h), (0,255,0), 2)
            faceProto="opencv_face_detector.pbtxt"
            faceModel="opencv_face_detector_uint8.pb"
            ageProto="age_deploy.prototxt"
            ageModel="age_net.caffemodel"
            genderProto="gender_deploy.prototxt"
            genderModel="gender_net.caffemodel"
            MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
            ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
            genderList=['Male','Female']
            faceNet=cv2.dnn.readNet(faceModel,faceProto)
            ageNet=cv2.dnn.readNet(ageModel,ageProto)
            genderNet=cv2.dnn.readNet(genderModel,genderProto)
            #face_img = frame[y:y+h, h:h+w].copy()#without a copying function your blob object wont work
            blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            #Predict Gender
            genderNet.setInput(blob)
            gender_preds = genderNet.forward()
            gender = genderList[gender_preds[0].argmax()]
            print("Gender : " + gender)
            #Age_prediction
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("AGE Output : {}".format(agePreds))
            print("Age : {}".format(age))
            overlay_text = "%s %s" % (gender, age)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(face_img, overlay_text, (x,y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        global main_source
        cv2.imwrite(dirpath+'/images/hello'+str(count)+'.jpg', face_img)
        try:
            with open(dirpath+'/images/hello'+str(count)+'.jpg', 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read())
            main_source=encoded_string

        except:
            try:
                with open(dirpath+'/images/hello1'+str(count-1)+'.jpg', 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                main_source=encoded_string
            except:
                with open(dirpath+'/images/hello1'+str(count-2)+'.jpg', 'rb') as image_file:
                    encoded_string = base64.b64encode(image_file.read())
                main_source=encoded_string
        return ""
    else:
        if type(main_source)==int:
            return ""
        return main_source
    #return render_template('index.html')	
if __name__ == "__main__":
    app.run()
