from new_env import CarlaEnv
import eye_detection as ed
import numpy as np
import cv2 as cv

env=CarlaEnv()
ed.init(cv.imread('w.png'))

def main():
    frame,_=env.reset()
    amount=0.3
    frame,_,_,_,_=env.step([[1,1,1]])
    while True:
        canvas,cord,distance,direction=ed.run(frame)
        env.step([[.4,.5,0]])
        
#        if distance==1:
#            ed.upArrow(canvas,cord)
#        elif distance==-1:
#            ed.downArrow(canvas,cord)
#        
#        if direction==-1:
#            ed.rightArrow(canvas,cord)
#        elif direction==1:
#            ed.leftArrow(canvas,cord)
#        ed.draw(canvas)

try:
    main()
except:
    env.destroy()        
