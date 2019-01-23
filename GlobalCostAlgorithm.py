import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
#import random # for uniform random
import math # for pi
from skimage.measure import compare_ssim # Note: install with: pip instasll scikit-image
import imutils # Note: install with: pip install imutils
import os
pi=np.pi

###################################################
##############  START - PARAMETERS   ##############
###################################################
imgPath = './input_pics/che.png' #'./input_pics/depp.jpg' #'./input_pics/JC_middle.jpg" #"./input_pics/cyllinder_side.jpg" ##"input_pics/pic_obama.png" #./obama.png" #"./triangle/triangle.jpg" #"./obama.png" ##./cmyk_spectr.jpg"  #"./obama.png" #"./cmyk_spectr.jpg" #"./jolie.jpg"##"./cmyk_spectr.jpg"#"./jolie.jpg" #"./obama.png"# ##"./cmyk_spectr.jpg" ## #"./obama.png" #"./cmyk_spectr.jpg"
imgPath2 ="./input_pics/cyllinder_side.jpg" #"input_pics/pic_obama.png"
imgPath3 = "./input_pics/cyllinder_top.jpg"
PinsCoords_file_2d = "./2dpins_coord.txt"
PinsCoords_file_3d = "./3dpins_coord.txt"
input_directory = "./input_pics/img_SA/" #directory with input images (can be several)  CAN BE CHANGED
output_directory = "./res_SA/" #outFoput directory. created automatically  CAN BE CHANGED
PinCoordsType = "2D"

ImgRadius = 1050#200     # Number of pixels that the image radius is resized to
InitPinIndex = 0         # Initial pin to start threading from 
NbPins = 200#200         # fNumber of pins on the circular loom
NbLines = 5000 # 3000 #500        # Maximal number of lines

Params_SSIM_WindowSize = 45#11
Params_blur_KernelSize = 3 # the blur kernel size (kernel width=kernel height)
Params_k_steps = 0

minLoop = 2#1# in java, none. 3   if-1 then not used      # Disallow loops of less than minLoop lines (if = NbLines then never twice the same pin. maybe a bad idea actually ;-) )
#not used but could: LineWidth = 1#3       # The number of pixels that represents the width of a thread


MinDistConsecPins = 15#25  minimal distance between two consecutive pins (in number of pins)

Params_w_tone = 1.0#0.5
Params_w_ssim = 0#0.5

FlagLineType = 'straight'#'bezier'#'straight'
Dimension = "2D" #"3D"
Params_num_pictures=1 #for different planes in 3D

variation=3#3 # for bezier? also for straight?
FlagMethod_PixelsOnLine='BLA'#'linsampling'#'BLA' # 'linsampling' # 
# Brensenham's Line Algorithm.
#https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

RigShape= "circle" #"square" # # 
###################################################
##############   END - PARAMETERS    ##############
###################################################

LUP_LinePixels={}
inputImgMasked=[]; inputImgMaskedBlurred=[]
imgResultCurrent=[]

####### END - Parameters #######

# Compute coordinates of loom pins
def Fn_CreatePinCoords(radius, nPins=200, offset=0, x0=None, y0=None):

  if RigShape=="square":
    FlagSamplingMethod="Square_AngleSampling" # "Square_SideSampling"
        
  coords = []

  if RigShape=="circle" or (RigShape=="square" and FlagSamplingMethod=="Square_AngleSampling"):

    alpha = np.linspace(0 + offset, 2*np.pi + offset, nPins + 1)

    if (x0 == None) or (y0 == None): # the center
      x0 = radius# + 1
      y0 = radius# + 1

    
    for angle in alpha[0:-1]:

      if RigShape=="circle":

        x = int(x0 + radius*np.cos(angle))
        y = int(y0 + radius*np.sin(angle))

      elif RigShape=="square" and FlagSamplingMethod=="Square_AngleSampling":
        
        # right side
        if (0<= angle and angle <=pi/4) or (7*pi/4<= angle and angle <=2*pi):
          x = int(x0 + radius)
          y = int(y0 - radius*np.tan(angle)) # minus for going up when angle increases
        # left side
        elif (3*pi/4<= angle and angle <=5*pi/4):
          x = int(x0 - radius)
          y = int(y0 + radius*np.tan(angle))
        # top side
        elif (pi/4<= angle and angle <=3*pi/4):
          x = int(x0 + radius/np.tan(angle))
          y = int(y0 - radius)
        # bottom side
        elif (5*pi/4<= angle and angle <=7*pi/4):
          x = int(x0 + radius/np.tan(angle))
          y = int(y0 + radius)
        else:
          print "ERROR Wrong case: ", angle
          sys.exit()
        #print angle, x, y
      else:
            print "ERROR Wrong case: ", RigShape
            sys.exit()

      coords.append((x, y))


  elif RigShape=="square" and FlagSamplingMethod=="Square_SideSampling":

    D=4*(2*radius) #the length of the square
    d=D/nPins

    x=0
    y=0


  else:
    print "ERROR Wrong case: ", RigShape
    sys.exit() 
      
  #print coords
  #print y0, " ", x0
  #cv2.waitKey(0)
      
  return coords

def Fn_CheckMinDistConsecPins(PrevPinIndex,NextPinIndex,myMinDistConsecPins):
    # Prevent to select two consecutive pins with less than minimal distance
    # returns 0 if failure (too close) or 1 if sucess (sufficiently far away)
    diff = abs(PrevPinIndex - NextPinIndex)
    diff = min(diff,NbPins-diff)
    thresh = myMinDistConsecPins # np.random.uniform(myMinDistConsecPins * 2/3, myMinDistConsecPins * 4/3)
    #if (diff < dist or diff > NbPins - dist):
    if diff < thresh:
       return 0
    else:
       return 1


def Fn_GetLinePixels_ForDrawing(coord0, coord1):
  # Inputs:
  #   - coord0, coord1: the 2D coordinates 

    # global FlagLineType #load
#     FlagLineType_backup=FlagLineType
#     FlagLineType='bezier'
#     xLine, yLine = Fn_GetLinePixels(pin0, pin1)
#     
#     FlagLineType=FlagLineType_backup
#     return (xLine,yLine)

     #if FlagLineType=='bezier': # cubic or quadratic?
        # Generate third point to introduce line variation (bezier control point)
        coordmiddle_x = np.random.uniform(-variation, variation) + (coord0[0] + coord1[0]) / 2
        coordmiddle_y = np.random.uniform(-variation, variation) + (coord0[1] + coord1[1]) / 2
        coordmiddle=(coordmiddle_x,coordmiddle_y)
        
        # Draw string as bezier curve
        return Fn_GetBezierCoords(coord0,coordmiddle,coordmiddle,coord1)
    

    
# Compute a line mask
def Fn_GetLinePixels(ind1, ind2):
  # Inputs:
  #   - ind1, ind2: the indices of the two end points of the line (see code below)
    if FlagLineType=='straight':
        # string and hash table
        PairID=Fn_PinPairID(ind1,ind2) 

        return LUP_LinePixels[PairID]
    else:
       print "ERROR: Wrong case"
    sys.exit()
    

# http://incolumitas.com/2013/10/06/plotting-bezier-curves/
def Fn_GetBezierCoords(p1,p2,p3,p4):
    #t = 0
    #coords_x = []
    #coords_y = []
    #N=10000
    N = int(np.hypot(p1[0] - p4[0], p1[1] - p4[1]))*1.5
    #print N
    list_t=np.linspace(0,1,N)
    list_x=np.linspace(0,1,N)
    list_y=np.linspace(0,1,N)
    list_z = np.linspace(0,1,N)
    #while (t < 1):
    i=-1
    for t in list_t:
        x = cubic_bezier_sum(t, (p1[0], p2[0], p3[0], p4[0]))
        y = cubic_bezier_sum(t, (p1[1], p2[1], p3[1], p4[1]))
        x=round(x)
        y=round(y)
     
        i=i+1
        list_x[i]=x
        list_y[i]=y

    return np.column_stack((list_x.astype(np.int)-1, list_y.astype(np.int)-1))
    
# http://incolumitas.com/2013/10/06/plotting-bezier-curves/
# Calculates the cubic Bezier polynomial for 
# the n+1=4 coordinates.
def cubic_bezier_sum(t, w):
        t2 = t * t
        t3 = t2 * t
        mt = 1-t
        mt2 = mt * mt
        mt3 = mt2 * mt
        return w[0]*mt3 + 3*w[1]*mt2*t + 3*w[2]*mt*t2 + w[3]*t3

#// Returns values a and b sorted in a string (e.g. a = 5 and b = 2 becomes 
#// "2-5"). This can be used as key in a map storing the lines between all
#// pins in a non-redundant manner.
def Fn_PinPairID(PinIndex1,PinIndex2):
    if PinIndex1<PinIndex2:
       ID= str(PinIndex1) + "-" + str(PinIndex2)
    else:
       ID= str(PinIndex2) + "-" + str(PinIndex1)

    return ID
    

    
def Fn_ReadPinCoordsFile():
    coords = []      
    for line in open("pincoords.txt", "r"):
       values = line.split(" ")
       x=int(values[0])
       y=int(values[1])
       coords.append((x, y))
    return coords


def Fn_PreComputeLinePixels2D(PinCoords):
    #LUP_LinePixels={}
    for ind1 in range(NbPins):
        coord1=PinCoords[ind1]
        for ind2 in range(ind1,NbPins):
            coord2=PinCoords[ind2]
        
            
            if FlagLineType=='straight':
        
        
                  if FlagMethod_PixelsOnLine=='linsampling':
                      length = int(np.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1]))
                      
                      x = np.linspace(coord1[0], coord2[0], length)
                      y = np.linspace(coord1[1], coord2[1], length)
                      
                      x=x.astype(np.int)-1
                      y=y.astype(np.int)-1
                      
                      
                  elif FlagMethod_PixelsOnLine=='BLA': # Brensenham's Line Algorithm.
                  # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
                      dx=abs(coord2[0] - coord1[0])
                      dy=-abs(coord2[1] - coord1[1])
                      if coord1[0] < coord2[0]:  # a.x < b.x ? 1 : -1;
                          sx=1
                      else:
                          sx=-1    
                      if coord1[1] < coord2[1]: # sy = a.y < b.y ? 1 : -1;
                          sy=1
                      else:
                          sy=-1
                      e=dx+dy
                      x = []
                      y = []
                      p=[coord1[0],coord1[1]] # starting point
                      while (1):
   
                          x.append(p[0])
                          y.append(p[1])
                          if p[0] == coord2[0] and p[1] == coord2[1]: # until we reach the end point
                             break;
                          e2 = 2 * e;
                          if e2 > dy:
                             e += dy;
                             p[0] += sx;
                         
                          if e2 < dx:
                             e += dx;
                             p[1] += sy;
                      x=np.asarray(x)
                      y=np.asarray(y)
                  else:
                      print "ERROR: Wrong case"
                      sys.exit()
                      
                      
                  
            # string and hash table
            PairID=Fn_PinPairID(ind1,ind2) 
            LUP_LinePixels[PairID]=np.column_stack((x, y)) # (xLine,yLine)
            

        
    return 

def Fn_PreComputeLinePixels(PinCoords):
    #LUP_LinePixels={}
    for ind1 in range(NbPins):
        coord1=PinCoords[ind1]
        for ind2 in range(ind1,NbPins):
            coord2=PinCoords[ind2]
            if Dimension=="3D":
        # Brensenham's Line Algorithm.
            # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
                dx=abs(coord2[0] - coord1[0])
                dy=abs(coord2[1] - coord1[1])
                dz=abs(coord2[2] - coord1[2])
                dm = max(dx,dy,dz)
                x1 = y1 = z1 = dm/2
                if coord1[0] < coord2[0]:  # a.x < b.x ? 1 : -1;
                    sx=1
                else:
                    sx=-1    
                if coord1[1] < coord2[1]: # sy = a.y < b.y ? 1 : -1;
                    sy=1
                else:
                    sy=-1
                if coord1[2] < coord2[2]: # sy = a.y < b.y ? 1 : -1;
                    sz=1
                else:
                    sz = -1
                x = []
                y = []
                z = []
                i = dm
                x1=y1=z1=dm/2
                p = [coord1[0],coord1[1],coord1[2]]

                while (1):

                    x.append(p[0])
                    y.append(p[1])
                    z.append(p[2])
                    i-=1
                    #print i
                    if i<0: # until we reach the end point
                       break;
                    x1 -= dx
                    if x1<0:
                      x1+=dm
                      p[0]+=sx
                    y1 -= dy
                    if y1<0:
                      y1+=dm
                      p[1]+=sy
                    z1 -= dz
                    if z1<0:
                      z1+=dm
                      p[2]+=sz

                y=np.asarray(y)
                z=np.asarray(z)
                
                          

            elif Dimension=="2D":
              if FlagLineType=='straight':
          
          
                    if FlagMethod_PixelsOnLine=='linsampling':
                        length = int(np.hypot(coord2[0] - coord1[0], coord2[1] - coord1[1]))
                        
                        x = np.linspace(coord1[0], coord2[0], length)
                        y = np.linspace(coord1[1], coord2[1], length)
                        
                        x=x.astype(np.int)-1
                        y=y.astype(np.int)-1
                        
                        
                    elif FlagMethod_PixelsOnLine=='BLA': # Brensenham's Line Algorithm.
                    # https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
                        dx=abs(coord2[0] - coord1[0])
                        dy=-abs(coord2[1] - coord1[1])
                        if coord1[0] < coord2[0]:  # a.x < b.x ? 1 : -1;
                            sx=1
                        else:
                            sx=-1    
                        if coord1[1] < coord2[1]: # sy = a.y < b.y ? 1 : -1;
                            sy=1
                        else:
                            sy=-1
                        e=dx+dy
                        x = []
                        y = []
                        p=[coord1[0],coord1[1]] # starting point
                        while (1):
     
                            x.append(p[0])
                            y.append(p[1])
                            if p[0] == coord2[0] and p[1] == coord2[1]: # until we reach the end point
                               break;
                            e2 = 2 * e;
                            if e2 > dy:
                               e += dy;
                               p[0] += sx;
                           
                            if e2 < dx:
                               e += dx;
                               p[1] += sy;
                        x=np.asarray(x)
                        y=np.asarray(y)
                    else:
                        print "ERROR: Wrong case"
                        sys.exit()
                        
            if Dimension == "3D":         
              PairID=Fn_PinPairID(ind1,ind2) 
              LUP_LinePixels[PairID]=np.column_stack((x, y, z)) # (xLine,yLine)  
            # string and hash table
            else: 
              PairID=Fn_PinPairID(ind1,ind2) 
              LUP_LinePixels[PairID]=np.column_stack((x, y)) # (xLine,yLine)
            

        
    return # used a global variables


def Fn_SaveResult(path, imgResult, ImgRadius, lines, blur_KernelSize):
  imgResult_blured = cv2.blur(imgResult,(blur_KernelSize,blur_KernelSize))

  cv2.destroyAllWindows()
  threadedPath = path+'threaded_z_projection.png'
  if Params_num_pictures>1:
   threadedPath_XPlane = path + "threaded_x_projection.png"
   threadedPath_YPlane = path + "threaded_y_projection.png"
   cv2.imwrite(threadedPath_XPlane,imgResult2)
   cv2.imwrite(threadedPath_YPlane,imgResult3)
  threadedMaskedPath = path +'threadedMasked.png' 
  blurredPath = path + 'threaded_blured.png'
  threadeBlurredMaskedPath = path + 'threaded_blured_masked.png'
  csvPath = path+'threaded.csv'
  csvCostPath = path + "cost.csv"
  ErrEvolutionPath = path+'ErrEvolution.txt'

  cv2.imwrite(threadedPath, imgResult)
  
  

  cv2.imwrite(blurredPath, imgResult_blured)
  if RigShape=="circle" and Dimension=="2D":
    threadeBlurredMasked = Fn_CreateMaskImage(imgResult_blured, ImgRadius)
    cv2.imwrite(threadeBlurredMaskedPath, imgResult_blured)


  csv_output = open(csvPath,'wb')
  csv_cost_output = open(csvCostPath,'wb')


  if Dimension=="3D":
      csv_output.write("x1,y1,z1,x2,y2,z2,index1,index2\n")  
      csver = lambda c1,c2,i1,i2 : "%i,%i,%i" % c1 + "," + "%i,%i,%i" % c2 + "," + "%i" % i1 + "," + "%i" % i2 + "\n"
  elif Dimension=="2D":
      csv_output.write("x1,y1,x2,y2,index1,index2\n")  
      csver = lambda c1,c2,i1,i2 : "%i,%i" % c1 + "," + "%i,%i" % c2 + "," + "%i" % i1 + "," + "%i" % i2 + "\n"
  for l in lines:
      csv_output.write(csver(PinCoords[l[0]],PinCoords[l[1]],l[0],l[1]))
  csv_output.close()
  for cost in line_costs:
    csv_cost_output.write(str(cost)+"\n")
  csv_cost_output.close()

def Fn_ComputeCostParams(inputImgMasked,imgResultCurrent,inputImgMaskedBlurred):
      ##    = Fn_ComputeCost(imgGray, OrigImage, imgResultCurrent)

  # Inputs:
  #   - inputImage: 
  #   - inputImageMasked
  #   - currentResult: our image with lines drawn so far
  # Outputs:
  #   - cost value (float)

 
  # Compute SSIM score
  if Params_w_ssim!=0:
    ssim_cost = compare_ssim(inputImgMasked, imgResultCurrent, multichannel=True, win_size=Params_SSIM_WindowSize)  #blur the grayscale input image and mask it, so that the background is same
  else:
    ssim_cost=0

  # Compute the tone score
  if Params_w_tone!=0:
    # # Step1: 
    #blurInputImage = cv2.GaussianBlur(inputImage,  blur_kernel, 0)
    # # mask it. optional I think
    # if 0:
    #   blurInputImage = Fn_CreateMaskImage(blurInputImage,ImgRadius)
    # Blur current result
    blurCurrentResult = cv2.GaussianBlur(imgResultCurrent,  (Params_blur_KernelSize,Params_blur_KernelSize), 0)
    m,n = blurCurrentResult.shape
    Diff=abs(inputImgMaskedBlurred.astype(int) - blurCurrentResult)/255.0
    #print Diff
    exp_power=2#2 the parts different will be more selected
    tone_cost=np.sum(Diff**exp_power)/float(m*n)
    # Compute tone and normalize

  else:
    tone_cost=0

  # Compute final score
  cost = Params_w_tone*tone_cost + Params_w_ssim*(1.0-ssim_cost)
  # Return
  #print cost
  return cost


#def Fn_ComputeCost(inputImage,inputImageMasked, currentResult):
#def Fn_ComputeCost(inputImageMasked, inputImageMaskedBlurred,currentResult):
#def Fn_ComputeCost(currentResult):
def Fn_ComputeCost():
      ##    = Fn_ComputeCost(imgGray, OrigImage, imgResultCurrent)

  # Inputs:
  #   - inputImage: 
  #   - inputImageMasked
  #   - currentResult: our image with lines drawn so far
  # Outputs:
  #   - cost value (float)

 
  # Compute SSIM score
  if Params_w_ssim!=0:
    ssim_cost = compare_ssim(inputImgMasked, imgResultCurrent, multichannel=True, win_size=Params_SSIM_WindowSize)  #blur the grayscale input image and mask it, so that the background is same
  else:
    ssim_cost=0

  # Compute the tone score
  if Params_w_tone!=0:
    # # Step1: 
    #blurInputImage = cv2.GaussianBlur(inputImage,  blur_kernel, 0)
    # # mask it. optional I think
    # if 0:
    #   blurInputImage = Fn_CreateMaskImage(blurInputImage,ImgRadius)
    # Blur current result
    blurCurrentResult = cv2.GaussianBlur(imgResultCurrent,  (Params_blur_KernelSize,Params_blur_KernelSize), 0)
    m,n = blurCurrentResult.shape
    Diff=abs(inputImgMaskedBlurred.astype(int) - blurCurrentResult)/255.0
    #print Diff
    exp_power=2#2 the parts different will be more selected
    tone_cost=np.sum(Diff**exp_power)/float(m*n)
    # Compute tone and normalize

  else:
    tone_cost=0

  # Compute final score
  cost = Params_w_tone*tone_cost + Params_w_ssim*(1.0-ssim_cost)
  # Return
  #print cost
  return cost


# Load image
def Fn_LoadImage(ImgPath):
  OrigImage = cv2.imread(ImgPath)
  if OrigImage is None:
     print "Error loading image: " + ImgPath 
     sys.exit()
  print "Image loaded: " + ImgPath 
  return OrigImage

def Fn_ImageCrop(OrigImage):
  # Crop at the center with a square
  height, width = OrigImage.shape[0:2]
  minEdge= min(height, width)
  topEdge = int((height - minEdge)/2)
  leftEdge = int((width - minEdge)/2)
  imgCropped = OrigImage[topEdge:topEdge+minEdge, leftEdge:leftEdge+minEdge]
  croppedPath = './cropped.png'
  cv2.imwrite(croppedPath, imgCropped)
  return imgCropped


def Fn_ImageCropAndResize(OrigImage,ImgRadius):
  imgCropped=Fn_ImageCrop(OrigImage)
  imgResized = cv2.resize(imgCropped, (2*ImgRadius + 1, 2*ImgRadius + 1)) 
  croppedPath = output_directory + 'croppedresized.png'
  cv2.imwrite(croppedPath, imgResized)
  return imgResized


# Apply circular mask to image
# i.e. pixels outside the circle are set to 255
def Fn_CreateMaskImage(image, radius):
  y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
  mask = x**2 + y**2 > radius**2
  image[mask] = 255
  return image

# Convert to grayscale
def Fn_rgb2gray(MyImage):
    imgGray = cv2.cvtColor(MyImage, cv2.COLOR_BGR2GRAY)
    return imgGray

def Fn_ImagePreProcessing(OrigImage,ImgRadius):

  imgGray=Fn_rgb2gray(OrigImage)
  # Crop
  imgCropped=Fn_ImageCrop(imgGray)
  # Resize
  imgCroppedResized = cv2.resize(imgCropped, (2*ImgRadius + 1, 2*ImgRadius + 1)) 
  cv2.imwrite(output_directory + './03_croppedresized.png', imgCroppedResized)
  # Mask the image
  if RigShape=="circle" and Dimension=="2D":
    imgMasked = Fn_CreateMaskImage(imgCroppedResized, ImgRadius)
    cv2.imwrite(output_directory + './04_masked.png', imgMasked)
    imgProcessed=imgMasked
  # The returned result
  # Mask to get circle for 2D, don't mask for 3D (cube, square plane)
    
  elif Dimension=="3D" or (Dimension=="2D" and RigShape=="square"):
    imgProcessed=imgCroppedResized

  # Saving for debug
  cv2.imwrite(output_directory + './01_gray.png', imgGray)
  cv2.imwrite(output_directory + './02_cropped.png', imgCropped)
  return imgProcessed
  
  
# def Fn_Check_k_Steps(NextPinIndex):
#   #NextPinIndex - from the last iteration

#   global LeastLineCost, BestPinIndex
#   PrevPinIndex2 = NextPinIndex

#   for index in range(NbPins):
#     imgResult_k_steps = np.copy(imgResultCurrent)
#     NextPinIndex2 = index
#     xyLine = Fn_GetLinePixels(PrevPinIndex2, NextPinIndex2)
#     # Step2: draw the line

#     imgResult_k_steps[xyLine[:,1], xyLine[:,0]]=0 
#     LineCost = Fn_ComputeCost(imgResult_k_steps)
#     PairID_next=Fn_PinPairID(PrevPinIndex2,NextPinIndex2)
#     if LineCost<LeastLineCost and not (PairID_next in ListPairs):

#       BestPinIndex = PrevPinIndex2
#       LeastLineCost = LineCost
      
#   return BestPinIndex

def Fn_Read3DCoords(file_path):
  coords = []
  f = open(file_path, "r")
  for line in f:
    coord = line.split()
    coords+=[(int(coord[0]),int(coord[1]),int(coord[2]))]
  return coords

def Fn_Read2DCoords(file_path):
  coords = []
  f = open(file_path, "r")
  for line in f:
    coord = line.split()
    coords+=[(int(coord[0]),int(coord[1]))]
  return coords

def initialize(input_directory, output_directory):
  onlyfiles = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
  #print onlyfiles
  output_dirs = []
  output_files = []
  if not os.path.exists(output_directory):
    os.makedirs(output_directory)
  for i in range(len(onlyfiles)) :
    file = onlyfiles[i]
    directory = output_directory+"_blur_"+str(Params_blur_KernelSize)+"/"+file.split('.')[0]+"/"
    onlyfiles[i] = input_directory+file
    
    #print directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    #if direcotory is empty, the image was not processed before
    if len(os.listdir(directory)) == 0:
      output_files.append(onlyfiles[i])
      output_dirs.append(directory)
  return output_files,output_dirs


#def XXX(): #
if __name__=="__main__":

  imgs, outpit_dirs = initialize(input_directory, output_directory)
  print len(imgs), " imgs"
  for i in range(len(imgs)):
    imgPath = imgs[i]
    output_directory = outpit_dirs[i]
    print "Welcome to main"

    print "IMG: ", imgPath
    print "OUTPUT: ",  output_directory
    print "Dimension: ", Dimension
    print Params_k_steps, "steps ahead"
    print "Number of pins: ", NbPins
    print "ImgRadius: ", ImgRadius

    inputImg = Fn_LoadImage(imgPath)
    
    # Image pre-processing (crop, gray, masked, etc)
    inputImgMasked = Fn_ImagePreProcessing(inputImg,ImgRadius)
    # one more pre-processing
    inputImgMaskedBlurred = cv2.GaussianBlur(inputImgMasked,(Params_blur_KernelSize,Params_blur_KernelSize), 0)

    #compute and store pin coordinates
    if Dimension=="3D":
      PinCoords = Fn_Read3DCoords(PinsCoords_file_3d)
      NbPins = len(PinCoords)
      if PinCoordsType == "2D":
        PinCoordsProjected = Fn_Read2DCoords(PinsCoords_file_2d)

      if Params_num_pictures>1:
        inputImg2 = Fn_LoadImage(imgPath2)
        inputImgMasked2 = Fn_ImagePreProcessing(inputImg2,ImgRadius)
        inputImgMaskedBlurred2 = cv2.GaussianBlur(inputImgMasked2,  (Params_blur_KernelSize,Params_blur_KernelSize), 0)
        inputImg3 = Fn_LoadImage(imgPath3)
        inputImgMasked3 = Fn_ImagePreProcessing(inputImg3,ImgRadius)
        inputImgMaskedBlurred3 = cv2.GaussianBlur(inputImgMasked3,  (Params_blur_KernelSize,Params_blur_KernelSize), 0)

    elif Dimension=="2D" and RigShape== "circle" :
      PinCoords = Fn_CreatePinCoords(ImgRadius, NbPins)
   
    elif Dimension=="2D" and RigShape== "square" :
      PinCoords = Fn_Read2DCoords(PinsCoords_file_2d)
      NbPins = len(PinCoords)

    height, width = inputImgMasked.shape[0:2]
    print "height:", height, " width:", width
          

    Fn_PreComputeLinePixels2D(PinCoords)

    # Initialize variables
    lines = [] 
    ListPairs=[] # the list of the selected pin pairs
    previousPins = []
    PrevPinIndex = InitPinIndex
    lineMask = np.zeros((height, width))
    line_costs = [] # to plot the evolution of the cost along the iterations
    
    #for json file


    # image result is rendered to
    imgResult = 255 * np.ones((height, width), np.uint8) # at each iteration, we will draw one more line

    if Params_num_pictures>1:
     imgResult2= 255 * np.ones((height, width), np.uint8)
     imgResult3= 255 * np.ones((height, width), np.uint8)
    #imgResultCurrent = 255 * np.ones((height, width), np.uint8) # initialize to white
    # Note: imwrite always expects [0,255], whereas imshow expects [0,1] for floating point and [0,255] for unsigned chars.https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite

    #######################################################################
    ############ START - Generate the line drawing (main loop) ############
    #######################################################################

    for line in range(NbLines): # for the target/max number of lines to draw


      LeastLineCost = float('inf')
      LineCost = LeastLineCost
      BestPinIndex = -1
      PrevPinCoord = PinCoords[PrevPinIndex]



      ###################################################################
      ############ START - Loop over all possible lines/pins ############
      ###################################################################
      for index in range(NbPins):


          imgResultCurrent = np.copy(imgResult) # i.e. the result line imagae up to the previous iteration
          if Params_num_pictures>1:  
            imgResultCurrent2 = np.copy(imgResult2)
            imgResultCurrent3 = np.copy(imgResult3)

          NextPinIndex=index

          NextPinCoord = PinCoords[NextPinIndex]

          # Draw the current line
          # Notes: we could create a function but we should avoid copying the whole image (by address would be good)
          # Step1: get the points on the line
          if Dimension=="3D" and PinCoordsType=="2D":
            xyPrev=PinCoords[PrevPinIndex][:2]
            xyNext=PinCoords[NextPinIndex][:2]
            PrevPinProjectedIndex = PinCoordsProjected.index(xyPrev)
            NextPinProjectedIndex = PinCoordsProjected.index(xyNext)
            xyLine = Fn_GetLinePixels(PrevPinProjectedIndex, NextPinProjectedIndex)
          else:
            xyLine = Fn_GetLinePixels(PrevPinIndex, NextPinIndex)
          
          #Step2: draw the line

          imgResultCurrent[xyLine[:,1], xyLine[:,0]]=0 
          if Params_num_pictures>1:  
           imgResultCurrent2[xyLine[:,1], xyLine[:,2]]=0   
           imgResultCurrent3[xyLine[:,0], xyLine[:,2]]=0   


          PairID=Fn_PinPairID(PrevPinIndex,NextPinIndex)

          if Params_k_steps>0:
            if not (PairID in ListPairs):
              BestPinIndex = Fn_Check_k_Steps(NextPinIndex)
          else:
            if Params_num_pictures>1:
              LineCost1 = Fn_ComputeCostParams(inputImgMasked,imgResultCurrent,inputImgMaskedBlurred)
              LineCost2 = Fn_ComputeCostParams(inputImgMasked2,imgResultCurrent2,inputImgMaskedBlurred2)
              LineCost3 = Fn_ComputeCostParams(inputImgMasked3,imgResultCurrent3,inputImgMaskedBlurred3)

              LineCost = LineCost1**2+LineCost2**2+LineCost3**2
            else:
              LineCost = Fn_ComputeCost()

          # Check if the cost is less than the previous leat
            if (LineCost < LeastLineCost) and not(NextPinIndex in previousPins) and not(PairID in ListPairs):
              LeastLineCost = LineCost
              BestPinIndex = NextPinIndex

      ###################################################################
      ############  END - Loop over all possible lines/pins  ############
      ###################################################################

      if BestPinIndex == -1:
        print "break: no best pin"
        break
     
      # For debug: append the cost of the best line of the current iteration (to plot the score evolution)
      if Params_k_steps==0:
        line_costs.append(LeastLineCost)


      # Update previous pins
      if minLoop !=-1:
        if len(previousPins) >= minLoop:
            previousPins.pop(0)
        previousPins.append(BestPinIndex)

      BestPinCoord=PinCoords[BestPinIndex]




      lines.append((PrevPinIndex, BestPinIndex))

      PairID=Fn_PinPairID(PrevPinIndex,BestPinIndex)
      ListPairs.append(PairID)
      # plot results
      #add best line to the current result image
      #xyLine = Fn_GetLinePixels_ForDrawing(PrevPinCoord, BestPinCoord)
      xyLine = Fn_GetLinePixels(PrevPinIndex, BestPinIndex)
      if Dimension=="3D" and PinCoordsType=="2D":
        xyPrev=PinCoords[PrevPinIndex][:2]
        xyBest=PinCoords[BestPinIndex][:2]
        PrevPinProjectedIndex = PinCoordsProjected.index(xyPrev)
        BestPinProjectedIndex = PinCoordsProjected.index(xyBest)
        xyLine = Fn_GetLinePixels(PrevPinProjectedIndex, BestPinProjectedIndex)
      else:
        xyLine = Fn_GetLinePixels(PrevPinIndex, BestPinIndex)
        #xyLine = Fn_GetLinePixels_ForDrawing(PrevPinCoord, BestPinCoord)


      imgResult[xyLine[:,1], xyLine[:,0]]=0
      if Params_num_pictures>1:
        imgResult2[xyLine[:,1], xyLine[:,2]]=0
        imgResult3[xyLine[:,0], xyLine[:,2]]=0


      # if Params_k_steps>0:
      #   LeastLineCost = Fn_ComputeCost(imgResult)
      #   line_costs.append(LeastLineCost)
      PrevPinIndex = BestPinIndex

      # Print progress
      if line%20==0:
        sys.stdout.write("\b\b")
        sys.stdout.write("\r")
        sys.stdout.write("[+] Computing line " + str(line + 1) + " of " + str(NbLines) + " total\n")
        sys.stdout.flush()
              

    #######################################################################
    ############ START - Generate the line drawing (main loop) ############
    #######################################################################

        print "saved to ", output_directory+str(NbPins)+"/"+str(line)
        Fn_SaveResult(output_directory+str(line), imgResult, ImgRadius, lines,Params_blur_KernelSize)

    Fn_SaveResult(output_directory, imgResult, ImgRadius, lines,Params_blur_KernelSize)
