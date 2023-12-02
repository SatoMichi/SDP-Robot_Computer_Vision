# SDP-Robot_Computer_Vision
This repository is copy of Computer Vision part of our SDP Robot Project.  
For more information about our SDP Robot Project pelase see *https://github.com/orgesskura/SDP-Robot-Control*

# Description
This project is part of University of Edinburgh Informatics SDP coursework project. Our team developped the robot which clean the lake automatically.  
The robot is disigned to collect the garbage floating on the lake automatically. Therefoer there are several function needed to be implemanted. 
1. Object detection by computer vision
2. Robot controll program which uses the information from object detection
3. Robot route planning algorithm to decide robot traveling path for effective garbage collection process  

This repository is copy of Computer Vision part of code for our robot.

# Computer Vision

## Object detection
First out robot need to recognize and detect the garbage floating on the water surface. This was first done by using object segmantation algorithm.  
One things need to be noted is that our robot need to detect the object on the water. Water surface which will reflect light and confuses our computer vision system.  
Our object segmentation algorithm suffer from this light reflection problem. therefore we need to find out some additional step to remove these noise.  
For solving this problem, I decided to apply some filtering before segmentation process, so that the computer vision system will be more robust to light reflection.  
The idea is simple. 
1. First get the picture of water surface without garbage. 
2. Next get the picture of water surface with garbage.
3. From image with garbage, subtract the image without garbage and get less-noisy infornation.
4. Apply object segmantation algorithm to this obtained image.

This pcocess is possible to remove some noise caused by the water surface light reflection.  
If the there is strong sunshine, the first and second image will both include strong reflection.
If it is cloudy, both images will have less reflection.  
Subtraction of these iamges will remove these light reflection independently from weather of the day.

## Object center detection
Once our robot recognized the garbage, the robot need to know which direction to go for collecting it.  
Our robot will go stright forward towards garbage, therefore obtaining relative position of garbage is important.  
I passed the x-y coordinates of the garbage in the robot vision camera, so that robot will be able to calculate whichdirection to forward.

## Object detection from video
Since our robot will move, the object detection algorithm need to be used on not only images but also videos.  
The algorithm is improved to deal with not only images but also videos which is sequence of images.  

## Object detection using CNN
One of the biggest problem we concerned with is the existence of water birds.  
Our robot need to detect these birds and should avoid them.  
We first tried to construct bird(garbage) detection model with Convolution Neural Network(CNN), however due to lack of data, we could not success to build the decent model.

## Optical flow
For the alternative approach, we decided to use optical flow to distinguish birds from garbage.  
We assumed the bird will move while garbage will not move.  
We thought if we could find out the speed of object mevement, it will be possible to distinguish birds from other objects.  
Of course water surface will move therefore the garbege will move too.   
Therefore we computed the relative speed of the detected object compared to the water surface movement.

## Thermography
Finally we came up with simple solution for detecting animal.  
We decided to use thermography to distinguish animal from other objects.  
If there is red or orange in the image, we can assume there is animal (even fish will have higher tempararure compared to other objects or water).