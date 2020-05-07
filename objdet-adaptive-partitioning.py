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


# Divide an image into multiple regions for object detection task
#  image_shape = (w,h)
#  dividers_list = A list of integer numbers. The numbers represent how many columns the row will be divided into, from top row to bottom row respectively. 
#  overlap_rate = How much the regions overlap each other
def divideImage(image_shape, dividers_list, overlap_rate=0.1):
    _W=0
    _H=1
    rows = len(dividers_list)

    region_list = []
    baseY = 0
    for row, num_divide in enumerate(dividers_list):
        region_width = image_shape[_W]/num_divide
        overlap = region_width * overlap_rate
        for i in range(num_divide):
            x1 = i * region_width - overlap
            y1 = baseY - overlap
            x2 = (i+1) * region_width + overlap
            y2 = baseY + region_width + overlap
            if x1<0:                x1=0
            if x1>=image_shape[_W]: x1=image_shape[_W]-1
            if y1<0:                y1=0
            if y1>=image_shape[_H]: y1=image_shape[_H]-1
            if x2<0:                x2=0
            if x2>=image_shape[_W]: x2=image_shape[_W]-1
            if y2<0:                y2=0
            if y2>=image_shape[_H]: y2=image_shape[_H]-1
            region_list.append((int(x1),int(y1),int(x2),int(y2)))
        baseY+=region_width
    return region_list


# Prepare data for object detection task
#  - Crop input image based on the region list produced by divideImage()
#  - Create a list of task which consists of coordinate of the ROI in the input image, and the image of the ROI
def createObjectDectionTasks(img, region_list):
    task_id = 0
    task_list = []
    for region in region_list:
        ROI = img[region[1]:region[3], region[0]:region[2]]
        task_list.append([region, ROI])
    return task_list

# Calculate IOU for non-maximum suppression
def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    iou_x1 = np.maximum(a[0], b[0])
    iou_y1 = np.maximum(a[1], b[1])
    iou_x2 = np.minimum(a[2], b[2])
    iou_y2 = np.minimum(a[3], b[3])

    iou_w = iou_x2 - iou_x1
    iou_h = iou_y2 - iou_y1

    if iou_w < 0 or iou_h < 0:
        return 0.0

    area_iou = iou_w * iou_h
    iou = area_iou / (area_a + area_b - area_iou)

    return iou


def draw_regions(img, region_list):
    colors= [
        (   0,   0,   0 ),
        ( 255,   0,   0 ),
        (   0,   0, 255 ),
        ( 255,   0, 255 ),
        (   0, 255,   0 ),
        ( 255, 255,   0 ),
        (   0, 255, 255 )
    ]
    _W=0
    _H=1
    for i, region in enumerate(region_list):
        img = cv2.rectangle(img, (region[0], region[1]), (region[2], region[3]), colors[i%7], 4)
    return img




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
    
    # Divide an image into multiple regions for object detection task
    #  (1920,1080) is the input image size
    #  [9,6,3] means, the image will be divided into 3 rows and each row will be divided into multiple colums (9, 6, and 3 columns from tom to bottom, respectively)
    #  [4,3] or [5,3] could be another options for 1920x1080 pictures. 
    region_list = divideImage((img.shape[1], img.shape[0]), [9,6,3])

    print('Displaying regions boudary boxes for 3 seconds.')
    img_tmp = img.copy()
    img_tmp = draw_regions(img_tmp, region_list)
    cv2.imshow('regions', img_tmp)
    cv2.waitKey(3 * 1000)

    task_list = createObjectDectionTasks(img, region_list)

    objects=[]
    for task in task_list:
        inBlob = cv2.resize(task[1], (input_shape[_W], input_shape[_H]))
        inBlob = inBlob.transpose((2, 0, 1))
        inBlob = inBlob.reshape(input_shape)
        res = exec_net.infer(inputs={input_name: inBlob})
        for obj in res[out_name][0][0]:                   # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            conf = obj[2]
            if conf > 0.6:                              # Confidence > 60% 
                ROI_shape = task[1].shape
                xmin = abs(int(obj[3] * ROI_shape[1])) + task[0][0]
                ymin = abs(int(obj[4] * ROI_shape[0])) + task[0][1]
                xmax = abs(int(obj[5] * ROI_shape[1])) + task[0][0]
                ymax = abs(int(obj[6] * ROI_shape[0])) + task[0][1]
                class_id = int(obj[1])
                objects.append([xmin, ymin, xmax, ymax, conf, class_id, True])

    # Do non-maximum suppression to reject the redundant objects on the overlap region
    for obj_id1, obj1 in enumerate(objects[:-2]):
        for obj_id2, obj2 in enumerate(objects[obj_id1+1:]):
            if obj1[6] == True and obj2[6]==True:
                IOU = iou(obj1[0:3+1], obj2[0:3+1])
                if IOU>0.5:
                    if obj1[4]<obj2[4]:
                        obj1[6] = False
                    else:
                        obj2[6] = False

    img = draw_regions(img, region_list)

    # Draw detection result
    for obj in objects:
        if obj[6]==True:
            img = cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (0,255,0), 2)  # Found object
        else:
            pass
            img = cv2.rectangle(img, (obj[0], obj[1]), (obj[2], obj[3]), (0,0,255), 1)  # Object which is rejected by NMS
    cv2.imshow('result', img)
    print('Displaying detection result for 10 seconds.')
    cv2.waitKey(10 * 1000)

if __name__ == '__main__':
    main()
