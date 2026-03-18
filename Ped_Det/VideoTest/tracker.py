import numpy as np
from kalman import KalmanFilterSimple
from utils import cosine_similarity,iou
from scipy.optimize import linear_sum_assignment


class Track:

    def __init__(self,track_id,bbox,feature):

        self.id=track_id
        self.bbox=bbox
        self.feature=feature
        self.kf=KalmanFilterSimple()
        self.lost=0
        self.trace=[]

        x1,y1,x2,y2=bbox

        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=x2-x1
        h=y2-y1

        self.kf.x[:4]=np.array([[cx],[cy],[w],[h]])


    def predict(self):

        self.kf.predict()

        cx,cy,w,h=self.kf.x[:4].flatten()

        x1=int(cx-w/2)
        y1=int(cy-h/2)
        x2=int(cx+w/2)
        y2=int(cy+h/2)

        self.bbox=[x1,y1,x2,y2]


    def update(self,bbox,feature):

        x1,y1,x2,y2=bbox

        cx=(x1+x2)/2
        cy=(y1+y2)/2
        w=x2-x1
        h=y2-y1

        self.kf.update(np.array([cx,cy,w,h]))

        self.feature=0.8*self.feature+0.2*feature
        self.bbox=bbox
        self.lost=0

        center=(int(cx),int(cy))
        self.trace.append(center)

        if len(self.trace)>30:
            self.trace.pop(0)

    def deduplicate_tracks(tracks, cos_th=0.9, iou_th=0.5):
        keep = []
        removed = set()
        for i in range(len(tracks)):
            if i in removed:
                continue
            for j in range(i + 1, len(tracks)):
                if j in removed:
                    continue
                sim = np.dot(tracks[i].feature, tracks[j].feature)
                iou_val = iou(tracks[i].bbox, tracks[j].bbox)
                if sim > cos_th and iou_val > iou_th:
                    if tracks[i].lost <= tracks[j].lost:
                        removed.add(j)
                    else:
                        removed.add(i)
            keep.append(i)
        return [tracks[i] for i in keep if i not in removed]