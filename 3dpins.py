##The algorithm computes 3D coordinates of the pins and saves it to pins_coord.txt##
import matplotlib
import sys
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#import random # for uniform random
import math # for pi
import random
import itertools
pi=np.pi

height = 720
width = 720
imgRadius = 700
egde = 700



RigShape = "square"

# Compute coordinates of loom pins
def Fn_CreatePinCoords(radius, nPins=100, offset=0, x0=None, y0=None):

  if RigShape=="square":
    FlagSamplingMethod="Square_AngleSampling" # "Square_SideSampling"
  edge = radius
  coords = []
  points = np.linspace(0, edge, nPins + 1)
  for point in points:


  	xyz1 = (int(point), 0, 0)
  	xyz2 = (0, int(point), 0)
  	xyz3 = (0, 0, int(point))
  	xyz4 = (int(point),edge,0)
  	xyz5 = (int(point),0,edge)
  	xyz6 = (0,int(point),edge)
  	xyz7 = (edge,int(point),0)
  	xyz8 = (0,edge,int(point))
  	xyz9 = (edge,0,int(point))
  	xyz10 = (edge,edge,int(point))
  	xyz11 = (edge,int(point),edge)
  	xyz12 = (int(point),edge,edge)






  	coords.append(xyz1)
  	coords.append(xyz2)
  	coords.append(xyz3)
  	coords.append(xyz4)
  	coords.append(xyz5)
  	coords.append(xyz6)
  	coords.append(xyz7)
  	coords.append(xyz8)
  	coords.append(xyz9)
  	coords.append(xyz10)
  	coords.append(xyz11)
  	coords.append(xyz12)
  #remove dublicates
  coords = list(set(coords))


  return coords

  # if RigShape=="circle" or (RigShape=="square" and FlagSamplingMethod=="Square_AngleSampling"):

  #   alpha = np.linspace(0 + offset, 2*np.pi + offset, nPins + 1)
  #   print alpha

  #   if (x0 == None) or (y0 == None): # the center
  #     x0 = radius# + 1
  #     y0 = radius# + 1

    
  #   for angle in alpha[0:-1]:
  #     if RigShape=="circle":

  #       x = int(x0 + radius*np.cos(angle))
  #       y = int(y0 + radius*np.sin(angle))

  #     elif RigShape=="square" and FlagSamplingMethod=="Square_AngleSampling":
        
  #       # right side
  #       if (0<= angle and angle <=pi/4) or (7*pi/4<= angle and angle <=2*pi):
  #         x = int(x0 + radius)
  #         y = int(y0 - radius*np.tan(angle)) # minus for going up when angle increases
  #       # left side
  #       elif (3*pi/4<= angle and angle <=5*pi/4):
  #         x = int(x0 - radius)
  #         y = int(y0 + radius*np.tan(angle))
  #       # top side
  #       elif (pi/4<= angle and angle <=3*pi/4):
  #         x = int(x0 + radius/np.tan(angle))
  #         y = int(y0 - radius)
  #       # bottom side
  #       elif (5*pi/4<= angle and angle <=7*pi/4):
  #         x = int(x0 + radius/np.tan(angle))
  #         y = int(y0 + radius)
  #       else:
  #         print "ERROR Wrong case: ", angle
  #         sys.exit()
  #       #print angle, x, y
  #     else:
  #           print "ERROR Wrong case: ", RigShape
  #           sys.exit()

  #     coords.append((x, y))


  # elif RigShape=="square" and FlagSamplingMethod=="Square_SideSampling":

  #   D=4*(2*radius) #the length of the square
  #   d=D/nPins

  #   x=0
  #   y=0


  # else:
  #   print "ERROR Wrong case: ", RigShape
  #   sys.exit() 

      
  # return coords


def Fn_DrawPins(MyImage,PinCoords):
    for k in range(len(PinCoords)):
       cv2.circle(MyImage, (PinCoords[k][0],PinCoords[k][1]), 1, 0, -1) #x,y
    return MyImage

PinCoords = Fn_CreatePinCoords(imgRadius, nPins=100, offset=0, x0=None, y0=None)
fig = plt.figure(figsize=(50,50))
ax = fig.add_subplot(111, projection='3d')
f = open('./pins_coord.txt','w')

for xyz in PinCoords:
	line = "%d %d %d\n" %(xyz[0], xyz[1], xyz[2])
	f.write(line)
	ax.scatter(xyz[0], xyz[1], xyz[2], c="r", marker="o")

f.close()


print PinCoords
print len(PinCoords)


ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
plt.show()
# PinCoords = Fn_CreatePinCoords(imgRadius, nPins=100, offset=0, x0=None, y0=None)
# print PinCoords
# imgResult = 255 * np.ones((height, width), np.uint8)
# imgResultimgResult=Fn_DrawPins(imgResult,PinCoords)
# cv2.imshow('image', imgResult/255.0)
# cv2.waitKey(0)


