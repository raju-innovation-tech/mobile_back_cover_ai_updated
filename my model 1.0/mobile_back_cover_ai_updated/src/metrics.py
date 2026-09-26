import numpy as np

def confusion(pred,tgt,k):
    p=pred.ravel(); t=tgt.ravel(); m=(t>=0)&(t<k); return np.bincount(k*t[m]+p[m],minlength=k*k).reshape(k,k)

def scores(cm):
    out=[]
    for i in range(len(cm)):
        tp=cm[i,i]; fp=cm[:,i].sum()-tp; fn=cm[i,:].sum()-tp
        iou=tp/(tp+fp+fn+1e-9); f1=2*tp/(2*tp+fp+fn+1e-9); out.append((iou,f1))
    return out

def boundary_f1(pred,tgt):
    import cv2
    pb=cv2.Canny((pred>0).astype(np.uint8)*255,50,150)>0; tb=cv2.Canny((tgt>0).astype(np.uint8)*255,50,150)>0
    d=cv2.dilate(tb.astype(np.uint8),np.ones((3,3),np.uint8))>0; dp=cv2.dilate(pb.astype(np.uint8),np.ones((3,3),np.uint8))>0
    prec=(pb&d).sum()/(pb.sum()+1e-9); rec=(tb&dp).sum()/(tb.sum()+1e-9); return 2*prec*rec/(prec+rec+1e-9)

def critical_boundary_f1(pred,tgt,editable=(1,),protected=(2,3),tol_px=2):
    """
    Boundary F1 restricted to the editable<->protected transition only
    (cover/logo vs camera/frame). This is the boundary that matters for
    this task -- the generic boundary_f1() above treats every object edge
    the same, which can look good while the one edge that must never leak
    (camera housing / frame line) is actually soft or offset.
    """
    import cv2
    ed_p=np.isin(pred,editable); ed_t=np.isin(tgt,editable)
    pb=cv2.Canny((ed_p).astype(np.uint8)*255,50,150)>0
    tb=cv2.Canny((ed_t).astype(np.uint8)*255,50,150)>0
    k=np.ones((2*tol_px+1,2*tol_px+1),np.uint8)
    d=cv2.dilate(tb.astype(np.uint8),k)>0; dp=cv2.dilate(pb.astype(np.uint8),k)>0
    prec=(pb&d).sum()/(pb.sum()+1e-9); rec=(tb&dp).sum()/(tb.sum()+1e-9)
    return 2*prec*rec/(prec+rec+1e-9)
