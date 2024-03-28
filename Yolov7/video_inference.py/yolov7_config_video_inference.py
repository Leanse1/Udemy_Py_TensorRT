'''
    Python Packages
    1. os
    2. cv2
    3. time

'''

import os
import cv2
import time
'''
    Class Components
    class Name : YoloV7DNN
    class Init components : nms_threshold, confidence_threshold, class_labels, image_path, yolov4 : [path_to_cfg_yolo, path_to_weights
    target: YOLOV4 DNN Inference on Images
'''

class YoloV7DNN:

    # INIT of  Parameters
                        #[]=0.3         #[]=0.38
    def __init__(self, nms_threshold, conf_threshold, class_labels, video_file, path_to_cfg, path_to_weights):

        # non max suppression threshold
        self.nms_threshold = nms_threshold

        # confidence threshold
        self.conf_threshold = conf_threshold

        # class labels 
        self.class_labels = class_labels
        
        # image path
        self.video_file = video_file

        # path to configuration yolov4
        self.path_to_cfg = path_to_cfg

          # path to weights yolov4
        self.path_to_weights = path_to_weights


        # read classes coco file with open()
        with open(class_labels, 'r') as read_class:
            self.class_labels= [ classes.strip() for classes in read_class.readlines()]


        # frame image
        # load images 
       # self.frames = self.load_images(self.image_path)


        # preprocess images and resize it
        '''for self.frame in self.frames:
            
            self.image = cv2.imread(self.frame)
            
            # get height and width of images

            self.original_h, self.original_w = self.image.shape[:2]
          
            dimension = (640, 640)

            # resize images
            self.resize_image = cv2.resize(self.image, dimension, interpolation=cv2.INTER_AREA)

            # get new height and width of resized image

            self.new_h, self.new_w = self.resize_image.shape[:2]

            # Call Function Inference RUN

            self.inference_run(self.resize_image)'''
    
    '''
            Function Target: Load Images
            param[1] : self
            param[2]: image_path

    '''
    '''def load_images(self, image_path):

            # list of images

        img_list = []

        for img_original in os.listdir(image_path):
            if img_original.endswith('.jpg') or img_original.endswith('.jpeg') or img_original.endswith('.png'):

                img_full_path = os.path.join(image_path, img_original)

                img_list.append(img_full_path)
        return img_list'''

    '''
        target: Inference DNN Opencv with ONNX
        param[1]: path to yolov4 confioguration
        param[2]:  path to yolov4 weights
    '''

    def inference_dnn(self, frame, path_to_cfg, path_to_weights):

        # read dnn of yolov4
        # weights and config    
        network = cv2.dnn.readNet(path_to_cfg, path_to_weights)
        # gpu or cpu 
        network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        network.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16) # floating point 16

       #creates net from file with trained weights and config, 
        model = cv2.dnn_DetectionModel(network)

        #set model parameters 

        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        '''
        classIds	Class indexes in result detection.
        [out]	confidences	A set of corresponding confidences.
        [out]	boxes	A set of bounding boxes.
        '''
        classes, scores, boxes = model.detect(frame, self.conf_threshold, self.nms_threshold)

        return classes, scores, boxes
    

    '''
    target: Inference Run and Draw Bounding boxes
    param[1] : image

    '''

    def inference_run (self):
        
        

        
        # video cap ture is reading
        video_cap = cv2.VideoCapture(self.video_file)

        if (video_cap.isOpened()==False):
            print('Video File Read UnSuccessfull!')

        # while loop, till file is open
        while(video_cap.isOpened()):

            grapped, frame = video_cap.read()

            frame = cv2.resize(frame,  (1000, 800))

            # if not grapped , close the video file and exit loop
            if not grapped:
                exit()

            # start 

            start = time.time()

            # get classes, get boxes, get score
            # inference for every frame
            getClasses, getScores, getBoxes = self.inference_dnn(frame, self.path_to_cfg, self.path_to_weights)

             # Update tracker with detection boxes and receive the tracking boxes
            # Here we assume `getBoxes` is in the format [[x,y,w,h], ...]
            
          

            end = time.time()

            # FRAME time

            frame_time = (end-start) * 1000

            # Frame per second

            FPS = 1.0 * (end-start)
    
            '''
            calculate new scale of image which is image formed between original and resized
            '''

            # new image ratio  height
          #  ratio_h = self.new_h / self.original_h

            # new image ratio width
           # ratio_w = self.new_w / self.original_w

            

            
            for (class_id, score, box) in zip(getClasses, getScores, getBoxes):
                #print(f"Class ID: {class_id}, Score : {score},  Box: {box}")

              #  print(f"Box Coordinates: ", box)
                
                # normalize bounding box to detection
              #  box[0] = int(box[0] * ratio_w) # x
               # box[1] = int(box[1] * ratio_h) # y
               # box[2] = int(box[2] * ratio_w) # width
               # box[3] = int(box[3] * ratio_h) # height


                cv2.rectangle(frame, box, (0,255,0),2)
                label = "Frame Time : %.2f ms, FPS : %.2f , ID: %s, Score: %.2f," % (frame_time, FPS ,self.class_labels[class_id], score)

                # calculate fps
            
    
                cv2.putText(frame,label, (box[0]-30, box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,  (252, 0, 0), 2)
            
            
            # show image
            cv2.imshow("Image Detected :", frame)

            if cv2.waitKey(1) & 0xFF==ord('q'):
                break
        
        # release video file
        video_cap.release()
        # destroy all windows
        cv2.destroyAllWindows()

        # while loop is finished!

def main():
    path_to_classes = '/tensorfl_vision/YOLOV5_CPP/data/coco-classes.txt'
   # video_path = '/tensorfl_vision/Onnx_Yolov4/main_demo.mp4'
    #video_path = '/tensorfl_vision/TensorRT_Inference/Yolov8 Detection.mp4'
    video_path = '/tensorfl_vision/Yolov7/cars.mp4'
    path_to_cfg_yolov7 = '/tensorfl_vision/Yolov8/yolov7-tiny.cfg'
    path_to_weights_yolov7 = '/tensorfl_vision/Yolov8/yolov7-tiny.weights'

    ## call class instance

    yolov7_inst = YoloV7DNN(0.3, 0.38, path_to_classes, video_path, path_to_cfg_yolov7, path_to_weights_yolov7)
    yolov7_inst.inference_run()
if __name__ == "__main__":
    main()




