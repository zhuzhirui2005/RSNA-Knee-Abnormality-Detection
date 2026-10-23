# Adapted from Heliosli's RSNA Knee pipeline. See NOTICE and THIRD_PARTY.md.
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class ViewGeometry:
    native_shape: tuple
    spacing: tuple
    size: int
    crop_xyxy: tuple
    valid_xyxy: tuple

    @property
    def crop_shape(self):
        x0,y0,x1,y1=self.crop_xyxy
        return y1-y0,x1-x0

    @property
    def resized_shape(self):
        x0,y0,x1,y1=self.valid_xyxy
        return y1-y0,x1-x0

    @property
    def extent(self):
        ch,cw=self.crop_shape;sy,sx=self.spacing
        return max(ch*sy,cw*sx)

    def receipt(self):
        x,y,_,_=self.crop_xyxy;px,py,_,_=self.valid_xyxy
        ch,cw=self.crop_shape;oh,ow=self.resized_shape;sy,sx=self.spacing
        pull=np.array([[cw/ow,0.,x-.5+(.5-px)*cw/ow],
                       [0.,ch/oh,y-.5+(.5-py)*ch/oh],[0.,0.,1.]],np.float64)
        valid=np.zeros((1,self.size,self.size),bool)
        valid[:,py:py+oh,px:px+ow]=True
        return {
            'output_to_native':pull,
            'native_spacing_mm':np.array([sy,sx],np.float64),
            'native_crop_xyxy':np.array(self.crop_xyxy,np.int64),
            'rendered_valid_xyxy':np.array(self.valid_xyxy,np.int64),
            'valid_image_xyxy':np.array(self.valid_xyxy,np.int64),
            'source_crop_fov_mm':np.array([ch*sy,cw*sx],np.float64),
            'output_spacing_mm':np.array([ch*sy/oh,cw*sx/ow],np.float64),
            'input_valid_mask':valid,
        }


def view_geometry(native_shape,spacing,output_size,*,crop_mm=None,center=(.5,.5)):
    """One crop/rounding/letterbox calculation shared by pixels and masks."""
    h,w=native_shape;sy,sx=map(float,spacing);size=int(output_size)
    if min(h,w,size)<1 or not np.isfinite([sy,sx]).all() or min(sy,sx)<=0:
        raise ValueError('invalid image geometry')
    ch,cw=h,w;y=x=0
    if crop_mm is not None:
        if not np.isfinite(crop_mm) or crop_mm<=0:raise ValueError('invalid crop extent')
        extent=min(float(crop_mm),h*sy,w*sx)
        ch=min(h,max(1,int(round(extent/sy))));cw=min(w,max(1,int(round(extent/sx))))
        y=int(np.clip(round(center[0]*(h-1)-(ch-1)/2),0,h-ch))
        x=int(np.clip(round(center[1]*(w-1)-(cw-1)/2),0,w-cw))
    field_y,field_x=ch*sy,cw*sx;extent=max(field_y,field_x)
    oh=max(1,round(size*field_y/extent));ow=max(1,round(size*field_x/extent))
    py=(size-oh)//2;px=(size-ow)//2
    return ViewGeometry((h,w),(sy,sx),size,(x,y,x+cw,y+ch),(px,py,px+ow,py+oh))
