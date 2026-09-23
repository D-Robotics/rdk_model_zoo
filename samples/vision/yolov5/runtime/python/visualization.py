"""Detection rendering kept outside the inference stages."""
import cv2


def draw_detections(image,result,labels):
    """Return an owned BGR overlay; do not modify the source image or result."""
    output=image.copy()
    for box,score,class_id in zip(result.boxes,result.scores,result.class_ids):
        x1,y1,x2,y2=(int(v) for v in box)
        label=labels[int(class_id)] if 0<=class_id<len(labels) else str(class_id)
        cv2.rectangle(output,(x1,y1),(x2,y2),(0,200,0),2)
        cv2.putText(output,f'{label}: {score:.3f}',(x1,max(15,y1)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,0),1)
    return output
