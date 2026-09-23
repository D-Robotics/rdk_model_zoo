"""Owned track overlay rendering, separate from tracking state and inference."""
import cv2


def draw_tracks(image,tracks):
    result=image.copy()
    for t in tracks:
        x1,y1,x2,y2=(int(x) for x in t.tlbr);tid=t.track_id
        color=((37*tid)%255,(17*tid)%255,(29*tid)%255)
        cv2.rectangle(result,(x1,y1),(x2,y2),color,2)
        cv2.putText(result,f'ID:{tid}',(x1,max(15,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,2)
    return result
