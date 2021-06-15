import math
import numpy as np
from numpy import linalg as LA


###
#DataMemory' definition
#list of vehicle's that update during the time of video
#[car1, car2, truck3, motorbike4]

#STRUCT
# [ id, [class] , [frame] , [center] , [velocity] , [boxes] , [iteration] ]
# id: is object in list
# [class] -> [class] 
# contains only id's class of object
# 
# [frame] -> [id_frame, id_previous_frame] 
# contains number of actual frame and number of previous frame in which it was detected

# [center] -> [center, prev_center]
# contains center of actual prediction and previous. 
#  
# [velocity] -> [velocity, previous_velocity, average_velocity, deltaD]
# contains actual velocity, load like actual position-previous_position / Time_actual - Time_previous ,
# previous velocity, mean velocity load like average of all velocity, deltaD load like actual position - previous position 
#
#  [boxes] -> [boxes]
# contains boxe 
# [iteration] -> [num_of_iteration]
# contains number of iteration of that object, how many frame detect it


#add object to dataMemory
def AddData(DataMemory, classes, counter, center, vel, boxe, num):
    DataMemory.append([
                    len(DataMemory),
                    [classes],
                    [counter, counter],
                    [center, center],
                    [vel,vel,vel,vel],
                    [boxe],
                    [num]])
    return DataMemory

def UpdateData(DataLevel,counter,center,boxe):
    
    DataLevel[2][1]=DataLevel[2][0]   #assign actual_IdData to previous
    DataLevel[3][1]=DataLevel[3][0]   #assign actual_centerData to previous
    DataLevel[4][1]=DataLevel[4][0]   #assign actual_BoxeData to previous

    DataLevel[2][0]=counter   #assign new id_frame
    DataLevel[3][0]=center    #assign new center
    DataLevel[5][0]=boxe      #assign new boxe
    
    #Calculate the new speed as the distance between the new center and the old one divided by the number of frames passed from the previous upgrade to the current one
    dist=math.sqrt( ( DataLevel[3][1][0] -  DataLevel[3][0][0])**2 + ( DataLevel[3][1][1] - DataLevel[3][0][1])**2 )
    time=DataLevel[2][0] - DataLevel[2][1]
    if time==0: time=1

    DataLevel[4][0] = dist/time

    #save actual deltaD
    Ax=DataLevel[3][0][0] -  DataLevel[3][1][0]
    Ay=DataLevel[3][0][1] -  DataLevel[3][1][1]
    DataLevel[4][3]=(Ax,Ay)

    #increase the counter of iterations made, I need it to calculate the weighted average of the speed
    DataLevel[6][0]= DataLevel[6][0]+1          
    
    #calculate the average speed as a average of past speeds
    DataLevel[4][2]= ( ( ( DataLevel[4][2] * ( DataLevel[6][0] - 1) ) + DataLevel[4][0] )
                         / DataLevel[6][0] )
    DataLevel[4][2]=round(DataLevel[4][2], 1)

    return DataLevel
    


def TrackingMemory(
    DataMemory,
    Class,
    Center,
    Boxe,
    counter,
    distance):

    IdArray=[]
    exReturn=[]
    #control if empty list 
    if len(DataMemory)!=0:  
        find=False
        dist=0
        #find object with right condition of distance, class and frame
        for j in range(0,len(DataMemory)):  
            #load distance from center in memory and object's center
            dist = math.sqrt((DataMemory[j][3][0][0] - Center[0])**2 + (DataMemory[j][3][0][1] -Center[1])**2 )
            tim= counter-DataMemory[j][2][0]
            if tim==0:
                tim=1

            vel= dist / tim 
            #object need right class, small distance, last update not not older than 15 frame and velocity control
            #velocity control comparison that actual velocity is not greater than 10*mean velocity of that,
            # ==0 need for the first upgrade.
            if (DataMemory[j][1][0]==Class and dist < distance and (counter-DataMemory[j][2][0]) < 15
                    and (vel<(10*DataMemory[j][4][2]) or DataMemory[j][4][2]==0) ):
               #if i find save id
               find=True
               IdArray.append(DataMemory[j][0])

        #if i did not find anything change search, check if by chance my network did not recognize the object for the previous frames.
        if find == False:
    
            #not found any acceptable value for the distance, I try to see along the hypothetical straight trajectory I have confirmation
            findProb=False
            for k in range(0,len(DataMemory)):
                #exclude object with velocity too small
                if DataMemory[k][4][2] > 2:

                    # I calculate the straight line passing through the previous and the current center, this is useful for me assuming one
                    # straight path, I calculate the minimum distance of the center I am looking for from the straight line, I see if it is close.                               
                    p1=np.asarray(DataMemory[k][3][1])  #previous center Data
                    p2=np.asarray(DataMemory[k][3][0])  #actual center Data
                    p3=np.asarray(Center)
                    distLine = LA.norm(np.cross(p2-p1, p1-p3))/LA.norm(p2-p1)

                    #load possible point on that line
                    if p2[0] - p1[0] !=0: m = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    else: m=0
                    b = p1[1] - m * p1[0]
                    pLine=(Center[0], int((Center[0]*m)+b))

                    #check that the signs of the past acceleration correspond to those between the center and the calculated center, if the same trajectory is possible
                    # Also check that the latest update is no more than 15 frames. to exclude vehicles that have left the scene
                    cond1=DataMemory[k][4][3][0] * (pLine[0]-DataMemory[k][3][0][0])
                    cond2=DataMemory[k][4][3][1] * (pLine[1]-DataMemory[k][3][0][1])
                    cond3=counter-DataMemory[k][2][0]

                    #cond1>0 cond2>0 direction with same sign
                    if distLine<distance and cond1>0 and cond2>0 and cond3<15:
                        findProb=True
                        IdArray.append(DataMemory[k][0])
                        #if find someone of useful, i return on different list for drawing line 
                        exReturn.append([Center,pLine,DataMemory[k][3][0]])
            
            #if i find probabilityPoint, update value
            if findProb==True:
                if len(IdArray)==1:    
                    DataMemory[IdArray[-1]]=UpdateData( DataMemory[IdArray[-1]],counter,Center,Boxe)       
                else:
                    Smallest=10000
                    idSmall=0
                    #search who is the distance's smallest
                    for i in range(0,len(IdArray)):
                        SmallDist = math.sqrt((DataMemory[IdArray[i]][3][0][0] - Center[0])**2 + (DataMemory[IdArray[i]][3][0][1] -Center[1])**2 )
                        if SmallDist<Smallest:
                            Smallest=SmallDist
                            idSmall=i
                    DataMemory[IdArray[idSmall]]=UpdateData( DataMemory[IdArray[idSmall]],counter,Center,Boxe) 
            #if i don't find, create new object in list
            else:
                DataMemory=AddData(DataMemory,Class,counter,Center,0,Boxe,0)

        else:

            #find object in data, control that only one casn be the right
            if len(IdArray)==1:    
                DataMemory[IdArray[-1]]=UpdateData( DataMemory[IdArray[-1]],counter,Center,Boxe)       
   
            else:
                Smallest=10000
                idSmall=0
                #find the best value
                for i in range(0,len(IdArray)):
                    SmallDist = math.sqrt((DataMemory[IdArray[i]][3][0][0] - Center[0])**2 + (DataMemory[IdArray[i]][3][0][1] -Center[1])**2 )
                    if SmallDist<Smallest:
                        Smallest=SmallDist
                        idSmall=i
                DataMemory[IdArray[idSmall]]=UpdateData( DataMemory[IdArray[idSmall]],counter,Center,Boxe) 
                       
                
    else:
        DataMemory=AddData(DataMemory,Class,counter,Center,0,Boxe,0)

    return DataMemory,exReturn


