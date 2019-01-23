import sys
import cv2
import numpy as np

#import random # for uniform random
import math # for pi
import random
randomInd = 10
#import matplotlib.pyplot as plt # for plotting the error evolution
pi=np.pi

####### START - Parameters #######
ImgPath = "./input_pics/jc_wife.jpg" #"./input_pics/kate_anf_contrast.jpg" #'./obama.png' #'starfish_01.jpg'#'./trump.jpg' #./obama.png'#'alish3.jpg' #'starfish_01.jpg' #'./obama_smile.jpg' # kitten.jpg" JC_middle.jpg ./couple_01.jpeg" #"./JC_middle.jpg"#""./yeoja_01.jpg" #./JC_middle.jpg" # "Marion_01.jpg"#
ImgRadius = 350     # Number of pixels that the image radius is resized to
blurInd = 10;
InitPinIndex = 0         # Initial pin to start threading from 
NbPins = 200 #200         # Number of pins on the circular loom
NbLines = 10 # 3000 #500        # Maximal number of lines

minLoop = 2#1# in java, none. 3   if-1 then not used      # Disallow loops of less than minLoop lines (if = NbLines then never twice the same pin. maybe a bad idea actually ;-) )
LineWidth = 1#3       # The number of pixels that represents the width of a thread
LineFade = 25#15       # The weight a single thread has in terms of "darkness"


MinDistConsecPins = 15#25 # // minimal distance between two consecutive pins (in number of pins)
myThreshCenterAngle=10./360.*2.*pi # if high, then no effect

LineScoreDef='sum_darkness_normalized' # 'sum_darkness' # 'sum_darkness_normalized'

FlagLineType = 'straight'#'bezier'#'straight'
variation=0#3 # for bezier? also for straight?
FlagMethod_PixelsOnLine='BLA'#'linsampling'#'BLA' # 'linsampling' # 
# Brensenham's Line Algorithm.
#https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

RigShape="circle" # "circle" # "square"

LineOpacity=10.0/100.0#20.0/255.0 # if 1 then fully opaque
LineColor=0.0#100.0 # 0~255

LUP_LinePixels={}



NbMultiCircles=1 #2
####### END - Parameters #######

# Load image
def FnLoadImage(ImgPath):
    OrigImage = cv2.imread(ImgPath)
    if OrigImage is None:
       print "Error loading image: " + ImgPath 
       sys.exit()
    print "Image loaded: " + ImgPath 
       
    return OrigImage

# Invert grayscale image
def invertImage(image):
    return (255-image)

# Apply circular mask to image
def maskImage(image, radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = x**2 + y**2 > radius**2
    image[mask] = 0

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
      
  #print coords
  #print y0, " ", x0
  #cv2.waitKey(0)
      
  return coords

def Fn_GetLinePixels_ForDrawing(coord0, coord1):

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
    

def Fn_CompareLinePixels():
    for ind1 in range(NbPins):
       for ind2 in range(ind1+1,NbPins):
    
        # string and hash table
        PairID=Fn_PinPairID(ind1,ind2) 
        coords = LUP_LinePixels[PairID]
        
        nb=-1
        for line in open(str(ind1) + "_" + str(ind2) + ".txt", "r"):
            print line
            nb=nb+1
            if nb>=1:
                values = line.split(" ")
                x=int(values[0])
                y=int(values[1])
                
                x2=coords[nb-1][0]
                y2=coords[nb-1][1]
                
                
                print "x:", x, " y:", y, " x2:", x2, " y2:", y2
                
                if not (x==x2 and y==y2):
                   print "ERROR"
                   sys.exit()
        
        if not (nb==coords.shape[0]):
            print "ERROR SIZE"
            print nb, ' ', coords.shape[0]
            sys.exit()
        
        #sys.exit()

    #sys.exit()
    
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

    croppedPath = './cropped.png'
    grayPath = './gray.png'
    maskedPath = './res_contrast2/obama/masked.png'
    contrastedPath = './res_contrast2/obama/contasted.png'

    cv2.imwrite(croppedPath, imgCropped)

    # Convert to grayscale
    imgGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(grayPath, imgGray)

    # Resize image
    imgSized = cv2.resize(imgGray, (2*ImgRadius + 1, 2*ImgRadius + 1)) 

    # Invert image
    #imgInverted = invertImage(imgSized)
    #cv2.imwrite('./inverted.png', imgInverted)
    imgInverted=imgSized
    contrastedImage = cv2.equalizeHist(imgSized)
    #https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html

    
    contrastedImageMasked = maskImage(contrastedImage, ImgRadius)
    cv2.imwrite(contrastedPath, contrastedImageMasked)
    # Mask image
    imgMasked = maskImage(imgInverted, ImgRadius)
    cv2.imwrite(maskedPath, imgMasked)
    print("written")
    
    #ProcessedImage=imgMasked
    ProcessedImage = contrastedImageMasked
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

def Fn_PlotFullConnectivity():

    print "Welcome to full connectivity"

    height= 2*ImgRadius + 1
    width = height
    
    # Define pin coordinates
    #PinCoords = Fn_CreatePinCoords(ImgRadius, NbPins)
    PinCoords = Fn_CreatePinCoords(ImgRadius, NbPins,0) # -pi/2 like Christian
    
    # image result is rendered to
    imgResult = 255 * np.ones((height, width))

    # Loop over all pins
    for PinIndex1 in range(1, NbPins):
        PinCoord1 = PinCoords[PinIndex1]
        print "PinCoord1: " , "x=" , PinCoord1[0] , "y=" , PinCoord1[1]

        #Loop over all pins
        for PinIndex2 in range(1, NbPins):
           if PinIndex1==PinIndex2:
              continue
        for PinIndex2 in range(PinIndex1+1, NbPins):
                              
            PinCoord2 = PinCoords[PinIndex2]

            xyLine = Fn_GetLinePixels_ForDrawing(PinCoord1, PinCoord2)

            # Plot the line
            imgResult[xyLine[:,1], xyLine[:,0]] = 0
            cv2.imshow('image', imgResult)
    
    cv2.imshow('image', imgResult)
    cv2.imwrite('./fullconnect.png', imgResult)
    cv2.waitKey(1)
    cv2.waitKey(0)
    
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
    
def Fn_ReadPinCoordsFile():
    coords = []      
    for line in open("pincoords.txt", "r"):
       values = line.split(" ")
       x=int(values[0])
       y=int(values[1])
       coords.append((x, y))
    return coords

def Fn_ReadResultFile():
    ListPinIndex=[]
    for line in open("instruction.txt", "r"):
       values = line.split(":")
       ListPinIndex.append(int(values[1]))
    return ListPinIndex

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

def Fn_DrawSelectedStrings(SelectedLines,height,width,NbCircles):
  # Info: SelectedLines contains (prevind,nextind)

  N=len(SelectedLines)
  print "nlines:", N
  #NbCircles=1
  kth_circle=0


  FlagNewImage=1 # initialize

  for i in range(N):

    if FlagNewImage==1:
      kth_circle+=1
      imgResult = 255 * np.ones((height, width), np.uint8)
      FlagNewImage=0


    PrevPinIndex=SelectedLines[i][0]
    NextPinIndex=SelectedLines[i][1]

    PrevPinCoord = PinCoords[PrevPinIndex]
    NextPinCoord = PinCoords[NextPinIndex]
    
    # plot results
    xyLine = Fn_GetLinePixels_ForDrawing(PrevPinCoord, NextPinCoord)
 
    for k in range(xyLine.shape[0]):
        val = round((1-LineOpacity)*imgResult[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
        if val>255:
           val=255
        imgResult[xyLine[k][1], xyLine[k][0]]=val
    
    #print round(N/NbCircles) 
    if i==N-1 or (NbCircles>1 and i>0 and i%round(N/NbCircles)==0): # i==round(N/NbCircles)
      print "in:", i
      FlagNewImage=1
      cv2.imwrite('./threaded_part-' + str(kth_circle) + '_outof-' + str(NbCircles) + '.png', imgResult)

def saveResult(path, imgResult, ImgRadius, lines, blurInd):
  imgResult_blured = cv2.blur(imgResult,(blurInd,blurInd))

  cv2.destroyAllWindows()
  threadedPath = path+'threaded.png'
  threadedMaskedPath = path +'threadedMasked.png' 
  blurredPath = path + 'threaded_blured.png'
  threadeBlurredMaskedPath = path + 'threaded_blured_masked.png'
  csvPath = path+'threaded.csv'
  ErrEvolutionPath = path+'ErrEvolution.txt'

  cv2.imwrite(threadedPath, imgResult)

  #Mask threaded image to compute the difference:
  threaded = FnLoadImage(threadedPath)
  threadedMasked = maskImage(threaded, ImgRadius)
  cv2.imwrite(threadedMaskedPath, threadedMasked)

  cv2.imwrite(blurredPath, imgResult_blured)
  threadeBlurredMasked = maskImage(imgResult_blured, ImgRadius)
  cv2.imwrite(threadeBlurredMaskedPath, imgResult_blured)

  csv_output = open(csvPath,'wb')
  csv_output.write("x1,y1,x2,y2,index1,index2\n")
  csver = lambda c1,c2,i1,i2 : "%i,%i" % c1 + "," + "%i,%i" % c2 + "," + "%i" % i1 + "," + "%i" % i2 + "\n"
  for l in lines:
      csv_output.write(csver(PinCoords[l[0]],PinCoords[l[1]],l[0],l[1]))
  csv_output.close()

  f = open( ErrEvolutionPath, 'w' )
  for e in ListErrEvolution:
    f.write("%f\n" % e)
  f.close()


#def XXX(): #
if __name__=="__main__":
  all_lines = []
  all_result = []
  img_pathes = ['./input_pics/siggraph_logo_blue.jpg', './input_pics/siggraph_logo_red.jpg']
  for i in range(len(img_pathes)):

      path = "./siggraph_logo/"+str(i)
      ImgPath = img_pathes[i]
      print(ImgPath)

      #print "Welcome to main"

      
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
      
      Fn_PreComputeLinePixels(PinCoords)
      
      #Fn_CompareLinePixels()
      ListPinIndex_GT=Fn_ReadResultFile()

      
      # Initialize variables
      lines = [] 
      #lines_MultiCircle = np.empty(NbMultiCircles, dtype=np.object) #create array
      #lines_MultiCircle[:]=[[] for _ in xrange(NbMultiCircles)] # [],[],[] # initialize. mandatory
      ListPairs=[] # the list of selected pin pairs
      previousPins = []
      PrevPinIndex = InitPinIndex
      lineMask = np.zeros((height, width))
      ListErrEvolution=[]
      
      #ListPrevIndex_MultiCircles=[InitPinIndex] * NbMultiCircles # i.e. repeat: InitPinIndex*np.ones((height, width))
      #kth_MultiCircle=-1

      # image result is rendered to
      imgResult = 255 * np.ones((height, width), np.uint8)
      imgResult_strings=imgResult
      # Note: imwrite always expects [0,255], whereas imshow expects [0,1] for floating point and [0,255] for unsigned chars.https://stackoverflow.com/questions/22488872/cv2-imshow-and-cv2-imwrite
      cv2.namedWindow('image', cv2.WINDOW_NORMAL) # resizable window
      
      imgResult=Fn_DrawPins(imgResult,PinCoords)
      cv2.imshow('image', imgResult/255.0)

      #imgResult_MultiCircle=np.tile(imgResult[:, :, None],(1,1,NbMultiCircles)) # https://stackoverflow.com/questions/26166471/3d-tiling-of-a-numpy-array
      #print imgResult_MultiCircle.shape

      for line in range(NbLines):
          bestLines = []

          # kth_MultiCircle=kth_MultiCircle+1
          # if kth_MultiCircle>=NbMultiCircles:
          #   kth_MultiCircle=0
          # PrevPinIndex=ListPrevIndex_MultiCircles[kth_MultiCircle]

          BestLineScore = -9999999999.0 #0
          BestPinIndex = -1
          PrevPinCoord = PinCoords[PrevPinIndex]

          #Compute the best connection:
          #if (i==0 or (line%randomInd)!=0):
          # if (line%randomInd)!=0:
          if True:
            #DiffMat=imgMasked.astype(int)-imgResult_strings.astype(int) # i.e. signe diff of original image - the result/drawn image 
            DiffMat=(255.0-OrigImage.astype(int))-(255.0-imgResult_strings.astype(int)) # i.e. signe diff of original image - the result/drawn image 
            #print type(DiffMat)
            myerr=np.sum(np.absolute(DiffMat))
            #print myerr
            ListErrEvolution.append(myerr) #signed error

            # Loop over possible lines/pins
            for index in range(NbPins):
                
                #pin = (oldPin + index) % nPins
                NextPinIndex=index
                if NextPinIndex==PrevPinIndex:
                   continue
                if Fn_CheckMinDistConsecPins(PrevPinIndex,NextPinIndex,MinDistConsecPins)==0:
                   #print "PrevPinIndex:", PrevPinIndex, 'NextPinIndex:', NextPinIndex
                   continue
                
                if line % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
                   #good. nothing
                   blatemp=0
                else: # i.e. odd number-th of connection
                    if Fn_CheckPassingOutCenter(PrevPinIndex,NextPinIndex,myThreshCenterAngle)==0:
                      continue;


                NextPinCoord = PinCoords[NextPinIndex]

                xyLine = Fn_GetLinePixels(PrevPinIndex, NextPinIndex)
                
                # Fitness function
                if LineScoreDef=='sum_darkness' or LineScoreDef=='sum_darkness_normalized':
                    #LineScore = np.sum(imgMasked[yLine, xLine])
                    #print "LineScore:", LineScore
                    
                    LineScore=0.0
                    for t in range(xyLine.shape[0]):
                    
                      LineScore+=DiffMat[xyLine[t][1], xyLine[t][0]]

                else:
                    print "ERROR: Wrong case"
                    sys.exit()
                    
                if LineScoreDef=='sum_darkness_normalized': 
      
                    LineScore = float(LineScore) / xyLine.shape[0]
                    #print("score:",LineScore)
          
                
                PairID=Fn_PinPairID(PrevPinIndex,NextPinIndex)
                

                if (LineScore > BestLineScore) and not(NextPinIndex in previousPins) and not(PairID in ListPairs):
                    
                    BestLineScore = LineScore
                    #print("better line score:", BestLineScore)
                    BestPinIndex = NextPinIndex

            #sort bestLines by score (ascending):

           # print(BestPinIndex)

            if BestPinIndex == -1:
              print "break: no best pin"
              break

          #Compute random connection:
          else:
#DiffMat=imgMasked.astype(int)-imgResult_strings.astype(int) # i.e. signe diff of original image - the result/drawn image 
            DiffMat=(255.0-OrigImage.astype(int))-(255.0-imgResult_strings.astype(int)) # i.e. signe diff of original image - the result/drawn image 
            #print type(DiffMat)
            myerr=np.sum(np.absolute(DiffMat))
            #print myerr
            ListErrEvolution.append(myerr) #signed error

            # Loop over possible lines/pins
            for index in range(NbPins):
                
                #pin = (oldPin + index) % nPins
                NextPinIndex=index
                if NextPinIndex==PrevPinIndex:
                   continue
                if Fn_CheckMinDistConsecPins(PrevPinIndex,NextPinIndex,MinDistConsecPins)==0:
                   #print "PrevPinIndex:", PrevPinIndex, 'NextPinIndex:', NextPinIndex
                   continue
                
                if line % 2 == 0: # i.e. # starting at 0 (like range()), even number is ok
                   #good. nothing
                   blatemp=0
                else: # i.e. odd number-th of connection
                    if Fn_CheckPassingOutCenter(PrevPinIndex,NextPinIndex,myThreshCenterAngle)==0:
                      continue;


                NextPinCoord = PinCoords[NextPinIndex]

                xyLine = Fn_GetLinePixels(PrevPinIndex, NextPinIndex)
                
                # Fitness function
                if LineScoreDef=='sum_darkness' or LineScoreDef=='sum_darkness_normalized':
                    #LineScore = np.sum(imgMasked[yLine, xLine])
                    #print "LineScore:", LineScore
                    
                    LineScore=0.0
                    for t in range(xyLine.shape[0]):
                    
                      LineScore+=DiffMat[xyLine[t][1], xyLine[t][0]]

                else:
                    print "ERROR: Wrong case"
                    sys.exit()
                    
                if LineScoreDef=='sum_darkness_normalized': 
      
                    LineScore = float(LineScore) / xyLine.shape[0]
                    #print("score:",LineScore)
          
                
                PairID=Fn_PinPairID(PrevPinIndex,NextPinIndex)
                
                if not(NextPinIndex in previousPins) and not(PairID in ListPairs):
                    bestLines.append([NextPinIndex, LineScore])


                # if (LineScore > BestLineScore) and not(NextPinIndex in previousPins) and not(PairID in ListPairs):
                    
                #     BestLineScore = LineScore
                #     #print("better line score:", BestLineScore)
                #     BestPinIndex = NextPinIndex

            #sort bestLines by score (ascending):
            bestLines.sort(key = lambda x: x[1])
            #reverse to get descending:
            bestLines = bestLines[::-1]
            #print(line)
            #print(bestLines[:6])

            ind = random.randrange(0,6)
            BestPinIndex = bestLines[ind][0]
           # print(BestPinIndex)

            if BestPinIndex == -1:
              print "break: no best pin"
              break

          # Update previous pins
          if minLoop !=-1:
            if len(previousPins) >= minLoop:
                previousPins.pop(0)
            previousPins.append(BestPinIndex)
          #print '\npreviousPins', previousPins

          BestPinCoord=PinCoords[BestPinIndex]
    
          xyLine = Fn_GetLinePixels(PrevPinIndex, BestPinIndex)

          for k in range(xyLine.shape[0]):
               val=imgMasked[xyLine[k][1], xyLine[k][0]]
               oldval=val
            
               val=val+LineFade
               #print val
               if val>255:
                  val=255
                  #print val
               imgMasked[xyLine[k][1], xyLine[k][0]] = val
          #
          imgMasked[imgMasked > 255] = 255 # truncate just in case

          # Save line to results
          lines.append((PrevPinIndex, BestPinIndex))
   
         # lines_MultiCircle[kth_MultiCircle].append((PrevPinIndex, BestPinIndex))
          
          PairID=Fn_PinPairID(PrevPinIndex,BestPinIndex)
          ListPairs.append(PairID)

          # plot results
          xyLine = Fn_GetLinePixels_ForDrawing(PrevPinCoord, BestPinCoord)
          
          for k in range(xyLine.shape[0]):
   
              val = round((1-LineOpacity)*imgResult[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
              if val>255:
                 val=255
              imgResult[xyLine[k][1], xyLine[k][0]]=val
              imgResult_strings[xyLine[k][1], xyLine[k][0]]=val


              # val = round((1-LineOpacity)*imgResult_MultiCircle[xyLine[k][1], xyLine[k][0],kth_MultiCircle] + LineOpacity*LineColor) #/255.0
              # if val>255:
              #    val=255
              # imgResult_MultiCircle[xyLine[k][1], xyLine[k][0],kth_MultiCircle] =val
              
          
          PrevPinIndex = BestPinIndex
          # ListPrevIndex_MultiCircles[kth_MultiCircle]=PrevPinIndex

          # Print progress
          if line%20==1:
            sys.stdout.write("\b\b")
            sys.stdout.write("\r")
            sys.stdout.write("[+] Computing line " + str(line + 1) + " of " + str(NbLines) + " total\n")
            sys.stdout.flush()
            saveResult(path+str(line), imgResult, ImgRadius, lines, blurInd)
      print "\n[+] Image threaded"
      all_result.append(imgResult)

      Fn_DrawSelectedStrings(lines,height,width,NbCircles=1)
      all_lines.append(lines)
  ImgResultCombined = 255 * np.ones((height, width), np.uint8)
  ImgResultBlue = 255 * np.ones((height, width), np.uint8)
  ImgResultRed = 255 * np.ones((height, width), np.uint8)
 

  lines_blue = all_lines[0]
  lines_red = all_lines[1]

  for line in lines_red:

    xyLine = Fn_GetLinePixels_ForDrawing(PinCoords[line[0]], PinCoords[line[1]])
    val = round((1-LineOpacity)*ImgResultBlue[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
    if val>255:
       val=255
    ImgResultCombined[xyLine[k][1], xyLine[k][0]]=[val,0,0]

    val = round((1-LineOpacity)*ImgResultBlue[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
    if val>255:
       val=255
    ImgResultBlue[xyLine[k][1], xyLine[k][0]]=val
    

  for line in lines_red:
    xyLine = Fn_GetLinePixels_ForDrawing(PinCoords[line[0]], PinCoords[line[1]])
    val = round((1-LineOpacity)*ImgResultRed[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
    if val>255:
       val=255
    ImgResultCombined[xyLine[k][1], xyLine[k][0]]=[0,0,val]
    val = round((1-LineOpacity)*ImgResultRed[xyLine[k][1], xyLine[k][0]] + LineOpacity*LineColor) #/255.0
    if val>255:
       val=255
    ImgResultRed[xyLine[k][1], xyLine[k][0]]=val
  cv2.imwrite('./siggraph_logo/blue_part.png', ImgResultBlue)
  cv2.imwrite('./siggraph_logo/red_part.png', ImgResultRed)
  cv2.imwrite('./siggraph_logo/combined.png',ImgResultCombined)