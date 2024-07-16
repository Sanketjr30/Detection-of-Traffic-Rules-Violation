from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
import torch
import cv2
import easyocr

model =  torch.hub.load('yolov5-master', 'custom', source ='local', path='best.pt',force_reload=True) 
classes = model.names 

EASY_OCR = easyocr.Reader(['en']) 
OCR_TH = 0.2
 
def detectx (frame, model):
    frame = [frame]
    results = model(frame)
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cordinates

def plot_boxes(results, frame,classes):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55: 
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) 
            text_d = classes[int(labels[i])]
            coords = [x1,y1,x2,y2]
            plate_num = recognize_plate_easyocr(img = frame, coords= coords, reader= EASY_OCR, region_threshold= OCR_TH)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
            cv2.putText(frame, f"{plate_num}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,255), 2)
    return frame, plate_num

def recognize_plate_easyocr(img, coords,reader,region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)] 
    ocr_result = reader.readtext(nplate)
    text = filter_text(region=nplate, ocr_result=ocr_result, region_threshold= region_threshold)
    if len(text) ==1:
        text = text[0].upper()
    return text

def filter_text(region, ocr_result, region_threshold):
    rectangle_size = region.shape[0]*region.shape[1]
    plate = [] 
    print(ocr_result)
    for result in ocr_result:
        length = np.sum(np.subtract(result[0][1], result[0][0]))
        height = np.sum(np.subtract(result[0][2], result[0][1]))
        if length*height / rectangle_size > region_threshold:
            plate.append(result[1])
    return plate

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

@app.route('/') 
def home():
    return render_template("index.html")

@app.route('/use_app') 
def use_app():
    return render_template("index.html")

@app.route("/uploader" , methods=['GET', 'POST'])
def uploader():    
    if request.method=='POST':
        f = request.files['file1']
        f.filename = "input_image.jpg"
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        frame = cv2.imread("static/input_image.jpg") 
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = detectx(frame, model = model)   
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame, plate_num = plot_boxes(results, frame,classes = classes)
        print(plate_num)
        cv2.imwrite("static/output_image.jpg",frame)

        # Load the image
        img = cv2.imread("static/input_image.jpg")

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Save noisy image
        cv2.imwrite('static/grey_scale.jpg', gray_image)

        # Load the image
        img = cv2.imread("static/input_image.jpg")

        # Convert to graycsale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
        edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
        
        # Save Pixel image
        cv2.imwrite('static/canny_edge_image.jpg', edges)
        print("saved")
        return render_template("display_2.html", plate_num=plate_num)

if __name__ == '__main__':
    app.run(debug=True) 