import cv2 as cv
import numpy as np

class Segmentator:
    def __init__(self):
        cv.setUseOptimized(True);
        cv.setNumThreads(4);
        
        self.net=cv.dnn.readNet('road-segmentation-adas-0001.xml',
                                'road-segmentation-adas-0001.bin')
        self.colors={         
                1: [255,0,0],
                2: [0,255,0],   
                3: [255,0,0],  
                }
    
    def forward(self,img):
        blob=cv.dnn.blobFromImage(img,size=(896,512))

        self.net.setInput(blob)
        out=self.net.forward()[0]
        out=np.argmax(out,axis=0).astype(np.float32)
        out=self.prepare(out,img)
        return out
        

    def prepare(self,segMap,original):
        res=np.ones(original.shape)
    
        for key,val in self.colors.items():
            res[np.where(segMap==key)]=val
        return res
    
    
    def fill(self,img):
#        im_in = cv.imread("road.jpg", cv.IMREAD_GRAYSCALE);
        th, im_th = cv.threshold(img, 1, 255, cv.THRESH_BINARY_INV);
        # Copy the thresholded image.
        im_floodfill = im_th.copy()
         
        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = im_th.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
         
        # Floodfill from point (0, 0)
        cv.floodFill(im_floodfill, mask, (0,0), 255);
         
        # Invert floodfilled image
        im_floodfill_inv = cv.bitwise_not(im_floodfill)
         
        # Combine the two images to get the foreground.
        im_out = im_th | im_floodfill_inv
         
        # Display images.
        cv.imshow("Thresholded Image", im_th)
        cv.imshow("Floodfilled Image", im_floodfill)
        cv.imshow("Inverted Floodfilled Image", im_floodfill_inv)
        cv.imshow("Foreground", im_out)
        cv.waitKey(0)
    
    
    def morph(self,img):
        img=self.forward(img).astype(np.float32)
#        print(img.shape)
        gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY).astype(np.uint8)
        
        _,res=cv.threshold(gray,1,255,cv.THRESH_BINARY)
#        self.fill(gray)
        kernal=np.ones((19,19),np.uint8)
#        opn=cv.morphologyEx(res,cv.MORPH_GRADIENT,kernal)
#        close=cv.morphologyEx(res,cv.MORPH_CLOSE,kernal)
        opn=cv.dilate(res,kernal,iterations=1)
        close=cv.erode(res,kernal,iterations=1)
        return opn,close