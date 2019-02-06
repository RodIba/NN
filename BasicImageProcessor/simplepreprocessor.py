import cv2  # imports required packages 

class SimplePreprocessor:
#constructor requires two arguments, folowed by a third optional one.
#width: The target width after resizing
#height: The target height after resizing
#inter: an optional parameter used to control which interpolation algorithm is used when resizing 

  def __init__(self, width, height, inter=cv2.INTER_AREA):
    # store thte target image width, height, and interpolation
    # method used when resizing 
    self.width = width 
    self.height = height
    self.inter = inter 
  

  def preprocesses(self, image):
    # resize the image to a fixed size, ignoring the aspect
    # ratio  
    return cv2.resize(image, (self.width, self.height),
         interpolation = self.inter)			
    #takes an input image, resizes it and returns it 
