import sys

import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork

model = [
'face-detection-0100',
'face-detection-0102',
'face-detection-0104',
'face-detection-0105',
'face-detection-0106',
'face-detection-adas-0001',
'face-detection-retail-0004',
'face-detection-retail-0005'
][0]
model = 'intel/'+model+'/FP16/'+model

def main():
    _N = 0
    _C = 1
    _H = 2
    _W = 3

    # Load DL model and setup Inference Engine of OpenVINO
    ie = IECore()
    net = ie.read_network(model+'.xml', model+'.bin')
    input_name  = next(iter(net.inputs))
    input_shape = net.inputs[input_name].shape
    out_name    = next(iter(net.outputs))
    out_shape   = net.outputs[out_name].shape            # [ image_id, label, conf, xmin, ymin, xmax, ymax ]
    exec_net    = ie.load_network(net, 'CPU')

    if len(sys.argv)<2:
        print('Use "image.jpg" as the input file name')
        infile = 'image.jpg'
    else:
        infile = sys.argv[1]
    img = cv2.imread(infile)

    objects=[]
    inBlob = cv2.resize(img, (input_shape[_W], input_shape[_H]))
    inBlob = inBlob.transpose((2, 0, 1))
    inBlob = inBlob.reshape(input_shape)
    res = exec_net.infer(inputs={input_name: inBlob})
    for obj in res[out_name][0][0]:                   # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
        conf = obj[2]
        if conf > 0.6:                              # Confidence > 60% 
            ROI_shape = img.shape
            xmin = abs(int(obj[3] * ROI_shape[1]))
            ymin = abs(int(obj[4] * ROI_shape[0]))
            xmax = abs(int(obj[5] * ROI_shape[1]))
            ymax = abs(int(obj[6] * ROI_shape[0]))
            class_id = int(obj[1])
            objects.append([xmin, ymin, xmax, ymax, conf, class_id, True])

    # Draw detection result
    for obj in objects:
        img = cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (0,255,0), 2)  # Found object
    cv2.imshow('result', img)
    print('Displaying detection result for 10 seconds.')
    cv2.waitKey(10 * 1000)

if __name__ == '__main__':
    main()
