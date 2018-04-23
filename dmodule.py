import cv2
def work(a,b,h1,h2):
    cx1=(a[0]/2)+(a[2]/2)
    cy1=(a[1]/2)+(a[3]/2)
    cx2=(b[0]/2)+(b[2]/2)
    cy2=(b[1]/2)+(b[3]/2)
    dx=cx1-cx2
    dy=cy1-cy2
    z=h1/h2
##    if z>0.97:
##        z=1
    return dx,dy,z
