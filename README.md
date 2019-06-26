# yolov3-implementation-from-scratch-pytorch

This is an implementation of YOLOv3 given the discription of the network in the file "yolov3.cfg". 
- The network is constructed in darknet.py
- the resulted detections are processed for thresholding and non-maximum suppression by functions in the utils.py
- images/ contains sample images for testing
- the pre-trained weights of the network can be downloaded from here: https://pjreddie.com/media/files/yolov3.weights
    or by this command: wget https://pjreddie.com/media/files/yolov3.weights
    
- you can run the network through command line using the detectctl.file and passing the location of your images as follows:
    python detctctl --image /path-to-images
    
  or you can run it in a notebook using in detect.py
  
- you can run it on a video stream from your web cam using web_cam.ipynb.

For video streams, it works in real-time when GPU is used
  
  This is a sample output of an image:
  
  ![alt text](https://github.com/Abdelrahman44/yolov3-implementation-from-scratch-pytorch/blob/master/detections/det_person.jpg)
