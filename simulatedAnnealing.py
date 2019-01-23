import sys
import cv2
import numpy as np
import os
#import random # for uniform random
import math # for pi
import random
randomInd = 10
#import matplotlib.pyplot as plt # for plotting the error evolution
pi=np.pi
import time
start_time = time.time()

####### START - Parameters #######
#ImgPath = "./input_pics/T.jpg" #"./input_pics/kate_anf_contrast.jpg" #'./obama.png' #'starfish_01.jpg'#'./trump.jpg' #./obama.png'#'alish3.jpg' #'starfish_01.jpg' #'./obama_smile.jpg' # kitten.jpg" JC_middle.jpg ./couple_01.jpeg" #"./JC_middle.jpg"#""./yeoja_01.jpg" #./JC_middle.jpg" # "Marion_01.jpg"#
#path = "./blur_alg/T/"




#[(switch,remove,add,move_n)]
#combs = [(False,True,False,False), (False,False,True,False),(False,False,False, True)] #for choosing switch or remove
#combs = [(False, False, False,True)]
#output_directory = "./res/rmv_add_move_n/log_a7_c0.01_10000de_probability"
input_directory = "./test_img/"
# connections_path = "./connections/portrait_threaded.csv"
ImgRadius = 1050 #350     # Number of pixels that the image radius is resized to
blurInd = 10;
InitPinIndex = 0         # Initial pin to start threading from 
NbPins = 200 #200         # Number of pins on the circular loom
NbLines =  3000 #500        # Maximal number of lines

minLoop = 2#1# in java, none. 3   if-1 then not used      # Disallow loops of less than minLoop lines (if = NbLines then never twice the same pin. maybe a bad idea actually ;-) )
LineWidth = 1#3       # The number of pixels that represents the width of a thread
LineFade = 25#15       # The weight a single thread has in terms of "darkness"

Params_blur_kernel = (3,3)
MinDistConsecPins = 25#25 # // minimal distance between two consecutive pins (in number of pins)
myThreshCenterAngle=10./360.*2.*pi # if high, then no effect

LineScoreDef='sum_darkness_normalized' # 'sum_darkness' # 'sum_darkness_normalized'

FlagLineType = 'straight'#'bezier'#'straight'
variation=3#3 # for bezier? also for straight?
FlagMethod_PixelsOnLine='BLA'#'linsampling'#'BLA' # 'linsampling' # 
# Brensenham's Line Algorithm.
#https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

RigShape="circle" # "circle" # "square"


LUP_LinePixels={}

Params_SSIM_WindowSize = 11
Params_blur_KernelSize = 3 # the blur kernel size (kernel width=kernel height)


MinDistConsecPins = 15#25  minimal distance between two consecutive pins (in number of pins)

Params_w_tone = 1.0#0.5
Params_w_ssim = 0#0.5
NbMultiCircles=1 #2
####### END - Parameters #######

def Fn_moveNConnections(lines_list): #line0 line1 line2
  #print "before: ", lines_list
  check = True
  new_lines = []

  while (len(new_lines)<(len(lines_list)-1)):
    NewPin = random.randrange(NbPins)
    ind = len(new_lines) #index of processed line
    #print "new lines: ", new_lines
    #print "old: ", lines_list
    #print "ind: ", ind
    if ind>0:
      while check:
        check = False
        NewPin = random.randrange(NbPins)
        if (new_lines[ind-1][1]==NewPin): 
          check = True
        if (Fn_CheckMinDistConsecPins(new_lines[ind-1][1], NewPin,MinDistConsecPins)==0):
          check = True
    if ind == 0:
      NewLine = (lines_list[ind][0], NewPin)
    else:
      NewLine = (new_lines[ind-1][1], NewPin)
    new_lines.append(NewLine)
  LastLine = (new_lines[-1][1],lines_list[-1][1])
  new_lines.append(LastLine)
  #print "after: ", new_lines
  return new_lines


#(pin1, pin2) (pin2, pin3), (pin3, pin4)
#(pin1, new_pin2) (pin2, pin3), (pin3, pin4) #ind=0  (pin1, new_pin2)
#(pin1, new_pin2) (new_pin2, pin3), (pin3, pin4) #ind=1 (pin1, new_pin2) (new_pin2, new_pin3)
#

def Fn_switch_connection (line1, line2, line_ind1, line_ind2):
  check = True
  while check:
    NewPin = random.randrange(NbPins)
    NewLine1 = (line1[0], NewPin)
    NewLine2 = (NewPin, line2[1])
    check = False
    if (line1[0]==NewPin) or (NewPin==line2[1]):
      check = True
    if (Fn_CheckMinDistConsecPins(line1[0], NewPin,MinDistConsecPins)==0) or (Fn_CheckMinDistConsecPins(NewPin, line2[1],MinDistConsecPins)==0):
      check = True
    if line_ind1 % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
       #good. nothing
       blatemp=0
    else: # i.e. odd number-th of connection
        if Fn_CheckPassingOutCenter(line1[0], NewPin,myThreshCenterAngle)==0:
          check = True
    if line_ind2 % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
       #good. nothing
       blatemp=0
    else: # i.e. odd number-th of connection
        if Fn_CheckPassingOutCenter(NewPin, line2[1],myThreshCenterAngle)==0:
          check = True
 
  return NewLine1, NewLine2


def Fn_remove_connection (line1, line2, line_ind1, line_ind2):
  check = True

  NewLine = (line1[0], line2[1])
  #print line1, line2, NewLine
  check = False
  if (line1[0]==line2[1]):
    check = True
  if (Fn_CheckMinDistConsecPins(line1[0], line2[1],MinDistConsecPins)==0):
    check = True
  if line_ind1 % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
     #good. nothing
     blatemp=0
  else: # i.e. odd number-th of connection
      if Fn_CheckPassingOutCenter(line1[0], line2[1],myThreshCenterAngle)==0:
        check = True

  return check, NewLine

def Fn_add_connection (line1,line_ind1):
  check = True
  while check:
    NewPin = random.randrange(NbPins)
    NewLine1 = (line1[0], NewPin)
    NewLine2 = (NewPin, line1[1])
    check = False
    if (line1[0]==NewPin) or (NewPin==line1[1]):
      check = True
    if (Fn_CheckMinDistConsecPins(line1[0], NewPin,MinDistConsecPins)==0) or (Fn_CheckMinDistConsecPins(NewPin, line1[1],MinDistConsecPins)==0):
      check = True
    if line_ind1 % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
       #good. nothing
       blatemp=0
    else: # i.e. odd number-th of connection
        if Fn_CheckPassingOutCenter(line1[0], NewPin,myThreshCenterAngle)==0:
          check = True
    if (line_ind1+1) % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
       #good. nothing
       blatemp=0
    else: # i.e. odd number-th of connection
        if Fn_CheckPassingOutCenter(NewPin, line1[1],myThreshCenterAngle)==0:
          check = True
 
  return NewLine1, NewLine2

def Fn_choose_line(lines):
  line_ind = random.randrange(1, (NbPins-1))
  return lines[line_ind-1], lines[line_ind], line_ind

def Fn_choose_lines(lines, n_lines):
  list_lines = []
  line_ind = random.randrange(NbPins-n_lines)
  for ind in range(line_ind, line_ind+n_lines):
    list_lines.append(lines[ind])
  #print n_lines, " lines: ", list_lines
  return list_lines,line_ind

def Fn_acc_probability(new_energy, energy, temperature,a,factor):
  if (new_energy<energy):
    return 1.0
  else:
    delta_energy = new_energy - energy
    return math.exp(-factor*delta_energy/(a*temperature))

def Fn_ComputeCost(imgResultCurrent):
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
    Diff=abs(inputImgBlurred.astype(int) - blurCurrentResult)/255.0
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
def FnLoadImage(ImgPath):
    OrigImage = cv2.imread(ImgPath)
    if OrigImage is None:
       print "Error loading image: " + ImgPath 
       sys.exit()
    print "Image loaded: " + ImgPath 
       
    return OrigImage

# Apply circular mask to image
def maskImage(image, radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 > radius**2
    image[mask] = 255

    return image

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
      
      
  return coords

def Fn_GetLinePixels_ForDrawing(coord0, coord1):


    coordmiddle_x = np.random.uniform(-variation, variation) + (coord0[0] + coord1[0]) / 2
    coordmiddle_y = np.random.uniform(-variation, variation) + (coord0[1] + coord1[1]) / 2
    coordmiddle=(coordmiddle_x,coordmiddle_y)
    
    # Draw string as bezier curve
    return Fn_GetBezierCoords(coord0,coordmiddle,coordmiddle,coord1)
    

    
# Compute a line mask
def Fn_GetLinePixels(ind1, ind2):
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


# Image Pre-processing
def Fn_ImagePreProcessing(OrigImage,ImgRadius, path):

    # Crop image
    height, width = OrigImage.shape[0:2]
    minEdge= min(height, width)
    topEdge = int((height - minEdge)/2)
    leftEdge = int((width - minEdge)/2)
    imgCropped = OrigImage[topEdge:topEdge+minEdge, leftEdge:leftEdge+minEdge]

    croppedPath = path + 'cropped.png'
    grayPath = path + 'gray.png'
    maskedPath = path + 'masked.png'

    cv2.imwrite(croppedPath, imgCropped)

    # Convert to grayscale
    imgGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(grayPath, imgGray)

    # Resize image
    imgSized = cv2.resize(imgGray, (2*ImgRadius + 1, 2*ImgRadius + 1)) 


    # Mask image
    imgMasked = maskImage(imgSized, ImgRadius)
    cv2.imwrite(maskedPath, imgMasked)
    print("written")
    
    #ProcessedImage=imgMasked
    ProcessedImage = imgMasked
    return ProcessedImage

#// Returns values a and b sorted in a string (e.g. a = 5 and b = 2 becomes 
#// "2-5"). This can be used as key in a map storing the lines between all
#// pins in a non-redundant manner.
def Fn_PinPairID(PinIndex1,PinIndex2):
    if PinIndex1<PinIndex2:
       ID= str(PinIndex1) + "-" + str(PinIndex2)
    else:
       ID= str(PinIndex2) + "-" + str(PinIndex1)

    return ID

    
def Fn_CheckMinDistConsecPins(PrevPinIndex,NextPinIndex,myMinDistConsecPins):
    #// Prevent two consecutive pins with less than minimal distance
    diff = abs(PrevPinIndex - NextPinIndex)
    diff = min(diff,NbPins-diff)
    thresh = myMinDistConsecPins # np.random.uniform(myMinDistConsecPins * 2/3, myMinDistConsecPins * 4/3)
    #if (diff < dist or diff > NbPins - dist):
    if diff < thresh:
       return 0
    else:
       return 1

def Fn_CheckPassingOutCenter(PrevPinIndex,NextPinIndex,myThreshCenterAngle):
    #// Prevent two consecutive pins with more than maximal distance: it is used to avoid the thread to pass near the center
    # returns 1 if not passing (success), returns 0 if passing near the center (failure)
    DeltaTheta=2*pi/NbPins*abs(PrevPinIndex-NextPinIndex)#the angle between the two pins
    diff=abs(pi-DeltaTheta)#between the opposite pin and the next pin

    if diff < myThreshCenterAngle:  
       return 0
    else:
       return 1
    
                 
def Fn_DrawPins(MyImage,PinCoords):
    for k in range(len(PinCoords)):
       cv2.circle(MyImage, (PinCoords[k][0],PinCoords[k][1]), 1, 0, -1) #x,y
    return MyImage


def Fn_PreComputeLinePixels(PinCoords):
    #LUP_LinePixels={}
    for ind1 in range(NbPins):
        coord1=PinCoords[ind1]
        for ind2 in range(ind1,NbPins):
            coord2=PinCoords[ind2]
    
            #xLine, yLine = Fn_GetLinePixels(ind1, ind2)
     
            
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

def saveResult(path, imgResult, ImgRadius, lines, lines_num, lines_stats,lines_moved, times,t):

  cv2.destroyAllWindows()
  threadedPath = path+str(t)+'threaded.png'
  threadedMaskedPath = path +'threadedMasked.png' 
  blurredPath = path + 'threaded_blured.png'
  csvPath = path+'threaded.csv'
  csvCostPath = path + "cost.csv"
  csvProbPath = path + "probability.csv"
  csvLineNumEvolutionPath = path + "line_evolution.csv"
  csvLineStatsPath = path + "lines_stats.csv"
  csvLinesMovedPath = path + "lines_moved.csv"
  txtTime = path+"time.txt"
  cv2.imwrite(threadedPath, imgResult)

  csv_output = open(csvPath,'wb')
  csv_output.write("x1,y1,x2,y2,index1,index2\n")
  csver = lambda c1,c2,i1,i2 : "%i,%i" % c1 + "," + "%i,%i" % c2 + "," + "%i" % i1 + "," + "%i" % i2 + "\n"
  for l in lines:
      csv_output.write(csver(PinCoords[l[0]],PinCoords[l[1]],l[0],l[1]))
  csv_output.close()

  csv_cost_output = open(csvCostPath,'wb')
  for cost in line_costs:
      line = "%s,%i\n"%(str(cost[0]), cost[1])
      csv_cost_output.write(line)
  csv_cost_output.close()

  csv_prob_output = open(csvProbPath,'wb')
  for prob in probs:
      csv_prob_output.write(str(prob)+"\n")
  csv_prob_output.close()

  csv_line_output = open(csvLineNumEvolutionPath,'wb')
  for num in lines_num:
      line = "%i,%i\n"%(num[0], num[1])
      #csv_line_output.write(str(num)+"\n")
      csv_line_output.write(line)
  csv_line_output.close()

  csv_lines_moved_output = open(csvLinesMovedPath,'wb')
  for num in lines_moved:
      line = "%i,%i\n"%(num[0], num[1])
      csv_lines_moved_output.write(line)
  csv_lines_moved_output.close()

  txt_time_output = open(txtTime,'wb')
  for i in times:
      line = "iteration: " + str(i[1])+ " time: " + i[0] + "\n"
      txt_time_output.write(line)
  txt_time_output.close()

  csv_line_stats_output = open(csvLineStatsPath,'wb')
  csv_line_stats_output.write("switched,removed\n")
  line = "%i,%i,%i,%i,%i"%(lines_stats[0], lines_stats[1], lines_stats[2], lines_stats[3], lines_stats[4])
  csv_line_stats_output.write(line)
  csv_line_stats_output.close()


#Creating directories for saving images and loading all images from the input dir
def initialize(input_directory, output_directory):
    onlyfiles = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))] # take just the files, no dir

    output_dirs = []
    output_files = []
    connections_paths = []
    # create directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for i, file in enumerate(onlyfiles):
        directory = output_directory + file.split('.')[0] + "/"
        onlyfiles[i] = input_directory + file
        connections = "./connections/"+  file.split('.')[0] + "/threaded.csv"

        # create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        # if output direcotory of image is empty, the image was not processed before; if it is not empty, skip that image
        if len(os.listdir(directory)) == 0:
            output_files.append(onlyfiles[i])
            output_dirs.append(directory)
            connections_paths.append(connections)
    connections_paths 
    return output_files, output_dirs,connections_paths 

def ReadConnections(file_path):
    lines = []   
    f = open(file_path, "r")
    f.readline()
    for line in f:
       l = line.split(",")
       ind1=int(l[4])
       ind2=int(l[5])
       lines.append((ind1, ind2))
    return lines

#def XXX(): #
if __name__=="__main__":

  #[(switch,remove,add,move_n)]
  combs = [(False,True,False,False),(False,False,True,False),(False,False,False,True)] #for choosing switch or remove
  for (a, C) in [(5, 0.001), (7,0.001), (5, 0.005), (7,0.005)]:
      init_temperature = 0.01
      limit = 100000 #max number of iterations
      annealing_factor = 0.6
      #C = 0.01

      t = 0
      #a = 7
      factor = 10000
      remove_connection = True
      switch_connection = False
      add_connection = False
      RandomInput = False
      output_directory = "./res/for_paper/add_rmv_move_n/log_a"+str(a)+"_c"+str(C)+"_10000de_probability/"
      imgs, outpit_dirs, connections_paths = initialize(input_directory, output_directory)
      print len(imgs), " imgs"
      for i in range(len(imgs)): #traverse all image in folder
        ImgPath = imgs[i]
        path = outpit_dirs[i]
        connections_path = connections_paths[i]
        print "IMG: ", ImgPath
        print "OUTPUT: ", path
        print "Number of pins: ", NbPins
        print "ImgRadius: ", ImgRadius
        # Load input image
        InputImage = FnLoadImage(ImgPath)
        
        # Preprocess the image (crop, resize, grayscale, invert colors, etc)
        imgMasked = Fn_ImagePreProcessing(InputImage,ImgRadius, path)
        print "Image preprocessed"

        OrigImage=imgMasked;
        cv2.imshow('image', imgMasked/255.0)

        
        # Define pin coordinates
        PinCoords = Fn_CreatePinCoords(ImgRadius, NbPins)

        height, width = imgMasked.shape[0:2]
        print "height:", height, " width:", width
        
        #precompute pixels of all the possible connections
        Fn_PreComputeLinePixels(PinCoords)
        
        # Initialize variables
        lines = [] 
        lines_num = [] #to plot line evolution
        lines_moved = []
        times = []
        lines_stats = [0,0,0,0,0] #to track how many [switched, removed, added]
        probs = []
        ListPairs=[] # the list of selected pin pairs
        previousPins = []
        PrevPinIndex = InitPinIndex
        lineMask = np.zeros((height, width))
        line_costs = [] # to plot the evolution of the cost along the iterations
        K = NbPins

        # image result is rendered to
        imgResult = 255 * np.ones((height, width), np.uint8)
        # Note: imwrite always expects [0,255], whereas imshow expects [0,1] for floating point and [0,255] for unsigned chars.https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite
        cv2.namedWindow('image', cv2.WINDOW_NORMAL) # resizable window
        
        #imgResult=Fn_DrawPins(imgResult,PinCoords)
        imgValues = 255 * np.ones((height, width), np.int64)
        if RandomInput:
          for line in range(NbLines): 

              #BestLineScore = -9999999999.0 #0
              
              BestPinIndex = -1
              PrevPinCoord = PinCoords[PrevPinIndex]

             
              #Break if the best connection was not found (therefore BestPinIndex didn't change)
              PrevPinIndex = random.randrange(NbPins)
              NextPinIndex = random.randrange(NbPins)
              PairID=Fn_PinPairID(PrevPinIndex,NextPinIndex)

              while (PrevPinIndex==NextPinIndex) or (NextPinIndex in previousPins) or (PairID in ListPairs):
                PrevPinIndex = random.randrange(NbPins)
                NextPinIndex = random.randrange(NbPins)
                PairID=Fn_PinPairID(PrevPinIndex,NextPinIndex)

              BestPinIndex = NextPinIndex


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
              #xyLine = Fn_GetLinePixels_ForDrawing(PrevPinCoord, BestPinCoord)
              xyLine = Fn_GetLinePixels(PrevPinIndex, BestPinIndex)
              # for k in range(xyLine.shape[0]):
              #   #draw the line in black
              #     imgResult[xyLine[k][1], xyLine[k][0]]=0
              #     imgValues[xyLine[k][1], xyLine[k][0]] -= 255 #to track intersections
              imgResult[xyLine[:,1], xyLine[:,0]]=0
              imgValues[xyLine[:,1], xyLine[:,0]] -= 255 #to track intersections
              #update previoud pin index to the new one
              PrevPinIndex = BestPinIndex

              # Print progress
              if line%20==1:
                sys.stdout.write("\b\b")
                sys.stdout.write("\r")
                sys.stdout.write("[+] Computing line " + str(line + 1) + " of " + str(NbLines) + " total\n")
                sys.stdout.flush()

        else:
          lines = ReadConnections(connections_path)
          for line in range(len(lines)):
            PrevPinIndex = lines[line][0]
            NextPinIndex = lines[line][1]
            xyLine = Fn_GetLinePixels(PrevPinIndex, NextPinIndex)
            for k in range(xyLine.shape[0]):
              #draw the line in black
                imgResult[xyLine[k][1], xyLine[k][0]]=0
                imgValues[xyLine[k][1], xyLine[k][0]] -= 255 #to track intersections
            if line%20==1:
              sys.stdout.write("\b\b")
              sys.stdout.write("\r")
              sys.stdout.write("[+] Drawing line " + str(line + 1) + " of " + str(NbLines) + " total\n")
              sys.stdout.flush()


              #saveResult(path+str(line), imgResult, ImgRadius, lines)
        inputImgBlurred = cv2.GaussianBlur(OrigImage, Params_blur_kernel,0)
        LineCost = Fn_ComputeCost(imgResult)
        line_costs.append((LineCost,t))
        lines_num.append((len(lines),t))
        print LineCost
        saveResult(path+"init", imgResult, ImgRadius, lines,lines_num, lines_stats,lines_moved, times,t)
        temperature = init_temperature
        energy = LineCost
        best_energy = energy
        #while energy>limit:
        while t<limit:
            
            ind = random.randrange(len(combs))
            switch_connection, remove_connection, add_connection, move_n_connections = combs[ind]
            RandInd1, RandInd2, line_ind = Fn_choose_line(lines) #connection (PrevPinIndex,NextPinIndex)
            n_lines = 0
            if switch_connection:
            #choose 2 connected lines randomly
            #change the connection
             
              NewInd1, NewInd2 = Fn_switch_connection(RandInd1, RandInd2, line_ind-1,line_ind)
              NewLine1 = Fn_GetLinePixels(NewInd1[0], NewInd1[1])
              NewLine2 = Fn_GetLinePixels(NewInd2[0], NewInd2[1])
            elif remove_connection:
              check, NewInd = Fn_remove_connection(RandInd1, RandInd2, line_ind-1,line_ind)
              while check:
                check, NewInd = Fn_remove_connection(RandInd1, RandInd2, line_ind-1,line_ind)
                RandInd1, RandInd2, line_ind = Fn_choose_line(lines) #connection (PrevPinIndex,NextPinIndex)
            elif add_connection:

              NewInd1, NewInd2 = Fn_add_connection(RandInd1, line_ind-1)
              NewLine1 = Fn_GetLinePixels(NewInd1[0], NewInd1[1])
              NewLine2 = Fn_GetLinePixels(NewInd2[0], NewInd2[1])
            elif move_n_connections:
              n_lines = random.randrange(2,10)
              list_lines, line_ind = Fn_choose_lines(lines, n_lines)


            if not move_n_connections:
              RandLine1 = Fn_GetLinePixels(RandInd1[0],  RandInd1[1])
              RandLine2 = Fn_GetLinePixels(RandInd2[0], RandInd2[1])
              for k in range(RandLine1.shape[0]):
                  #draw the line in white
                if imgValues[RandLine1[k][1], RandLine1[k][0]] == 0:
                  imgValues[RandLine1[k][1], RandLine1[k][0]] += 255
                  imgResult[RandLine1[k][1], RandLine1[k][0]] = 255
                
                #not draw on intersections
                elif imgValues[RandLine1[k][1], RandLine1[k][0]] < 0:
                  imgValues[RandLine1[k][1], RandLine1[k][0]] += 255
                  imgResult[RandLine1[k][1], RandLine1[k][0]] = 0
              if remove_connection or switch_connection:
                for k in range(RandLine2.shape[0]):
                    #draw the line in white
                  if imgValues[RandLine2[k][1], RandLine2[k][0]] == 0:
                    imgValues[RandLine2[k][1], RandLine2[k][0]] += 255
                    imgResult[RandLine2[k][1], RandLine2[k][0]] = 255
                  #not draw on intersections
                  elif imgValues[RandLine2[k][1], RandLine2[k][0]] < 0:
                    imgValues[RandLine2[k][1], RandLine2[k][0]] += 255
                    imgResult[RandLine2[k][1], RandLine2[k][0]] = 0

            else:
              for Line in list_lines:
                RandLine = Fn_GetLinePixels(Line[0],  Line[1])
                for k in range(RandLine.shape[0]):
                #draw the line in white
                  if imgValues[RandLine[k][1], RandLine[k][0]] == 0:
                    imgValues[RandLine[k][1], RandLine[k][0]] += 255
                    imgResult[RandLine[k][1], RandLine[k][0]] = 255
                  elif imgValues[RandLine[k][1], RandLine[k][0]] < 0:
                    imgValues[RandLine[k][1], RandLine[k][0]] += 255
                    imgResult[RandLine[k][1], RandLine[k][0]] = 0

            if switch_connection or add_connection:
              imgResult[NewLine1[:,1], NewLine1[:,0]] = 0
              imgValues[NewLine1[:,1], NewLine1[:,0]] -= 255

              imgResult[NewLine2[:,1], NewLine2[:,0]] = 0
              imgValues[NewLine2[:,1], NewLine2[:,0]] -= 255

            elif remove_connection:
              NewLine = Fn_GetLinePixels(NewInd[0], NewInd[1])
              imgResult[NewLine[:,1], NewLine[:,0]] = 0
              imgValues[NewLine[:,1], NewLine[:,0]] -= 255

            elif move_n_connections:
              new_lines = Fn_moveNConnections(list_lines)
              for Line in new_lines:
                NewLine = Fn_GetLinePixels(Line[0], Line[1])
                imgResult[NewLine[:,1], NewLine[:,0]] = 0
                imgValues[NewLine[:,1], NewLine[:,0]] -= 255

            new_energy = Fn_ComputeCost(imgResult)
            
            
            probability = Fn_acc_probability(new_energy, energy, temperature,a, factor)

            probs.append(probability)
         
            #which kind of distribution (look in the paper)
            p = random.random()

            #if random.random()<probability:
            #print "random: ", p, "probability: ", probability, "t: ", temperature
            if (p<probability):
              if switch_connection:
                lines_stats[0]+=1
                print "switched"
                lines[line_ind-1] = NewInd1
                lines[line_ind] = NewInd2
              elif remove_connection:
                lines_stats[1]+=1
                lines[line_ind-1] = NewInd
                lines.pop(line_ind)
                print "removed"
                print "# of lines: ", len(lines)
              elif add_connection:
                lines_stats[2]+=1
                lines[line_ind-1] = NewInd1
                lines.insert(line_ind, NewInd2)
                print "added"
                print "# of lines: ", len(lines)
              elif move_n_connections:
                lines_stats[3]+=1
                lines_stats[4]+=n_lines
                
                for i in range(len(new_lines)):
                  lines[line_ind+i] = new_lines[i]
                print "moved ", n_lines, " lines"


              print "NEW ENERGY: ", new_energy, "iteration: ", t
              lines_num.append((len(lines),t))
              lines_moved.append((n_lines, t))
              line_costs.append((new_energy,t))
              energy = new_energy
              e = int(time.time() - start_time)
              line = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
              print (line)
              times.append((line,t))
              if new_energy<best_energy:
                best_energy = new_energy
                saveResult(path+"best", imgResult, ImgRadius, lines,lines_num,lines_stats,lines_moved, times,t)
              saveResult(path, imgResult, ImgRadius, lines,lines_num,lines_stats,lines_moved, times,t)

            else:
              if not move_n_connections:
                imgResult[RandLine1[:,1], RandLine1[:,0]] = 0
                imgValues[RandLine1[:,1], RandLine1[:,0]] -= 255

                if switch_connection or remove_connection:
                  imgResult[RandLine2[:,1], RandLine2[:,0]] = 0
                  imgValues[RandLine2[:,1], RandLine2[:,0]] -= 255

                if switch_connection or add_connection:
                  for k in range(NewLine1.shape[0]):
                      #draw the line in white
                    if imgValues[NewLine1[k][1],NewLine1[k][0]] == 0:
                      imgValues[NewLine1[k][1], NewLine1[k][0]] += 255
                      imgResult[NewLine1[k][1], NewLine1[k][0]] = 255
                    
                    elif imgValues[NewLine1[k][1], NewLine1[k][0]] < 0:
                      imgValues[NewLine1[k][1], NewLine1[k][0]] += 255
                      imgResult[NewLine1[k][1], NewLine1[k][0]] = 0

                      
                  for k in range(NewLine2.shape[0]):
                      #draw the line in white
                    if imgValues[NewLine2[k][1],NewLine2[k][0]] == 0:
                      imgValues[NewLine2[k][1], NewLine2[k][0]] += 255
                      imgResult[NewLine2[k][1], NewLine2[k][0]] = 255
                    #not draw on intersections
                    elif imgValues[NewLine2[k][1], NewLine2[k][0]] < 0:
                      imgValues[NewLine2[k][1], NewLine2[k][0]] += 255
                      imgResult[NewLine2[k][1], NewLine2[k][0]] = 0

                elif remove_connection:
                  for k in range(NewLine.shape[0]):
                      #draw the line in white
                    if imgValues[NewLine[k][1],NewLine[k][0]] == 0:
                      imgValues[NewLine[k][1], NewLine[k][0]] += 255
                      imgResult[NewLine[k][1], NewLine[k][0]] = 255
                    
                    elif imgValues[NewLine[k][1], NewLine[k][0]] < 0:
                      imgValues[NewLine[k][1], NewLine[k][0]] += 255
                      imgResult[NewLine[k][1], NewLine[k][0]] = 0
              elif move_n_connections:
                for Line1 in list_lines:
                  RandLine1 = Fn_GetLinePixels(Line1[0],  Line1[1])
                  imgResult[RandLine1[:,1], RandLine1[:,0]] = 0
                  imgValues[RandLine1[:,1], RandLine1[:,0]] -= 255

                for Line2 in new_lines:
                  NewLine = Fn_GetLinePixels(Line2[0], Line2[1])

                  for k in range(NewLine.shape[0]):
                    if imgValues[NewLine[k][1],NewLine[k][0]] == 0:
                      imgValues[NewLine[k][1], NewLine[k][0]] += 255
                      imgResult[NewLine[k][1], NewLine[k][0]] = 255
                    
                    elif imgValues[NewLine[k][1], NewLine[k][0]] < 0:
                      imgValues[NewLine[k][1], NewLine[k][0]] += 255
                      imgResult[NewLine[k][1], NewLine[k][0]] = 0


            t = t + 1

            #exp2:
            #temperature = init_temperature*(annealing_factor**t)
            #exp:
            #temperature = init_temperature*math.exp(-annealing_factor*t)
            #log:
            temperature = C/math.log(1+t)
            #temperature = annealing_factor*temperature
        
        e = int(time.time() - start_time)
        line = '{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60)
        print (line)
        times.append((line,t))
        saveResult(path, imgResult, ImgRadius, lines,lines_num,lines_stats,lines_moved, times,t)
        print "\n[+] Image threaded"


