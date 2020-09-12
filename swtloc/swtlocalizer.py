# Author : Achintya Gupta

from .utils import prog_bar, auto_canny, prepCC, print_valcnts, imgshow, imgshowN, imgsave
from .swt import SWT
from .bubble_bbox import BubbleBBOX


import os
from cv2 import cv2
import numpy as np
import warnings
import math
import time

warnings.filterwarnings('ignore')

COMPONENT_PROPS = {'pixels': None,
                   'bbm_h': None, 'bbm_w': None, 'bbm_cx': None, 'bbm_cy': None, 'bbm_ar': None,
                   'bbm_bbox': None, 'bbm_outline': None, 'bbm_ang': None,
                   'img_color_mean': None, 'img_color_median': None,
                   'sw_countdict': None, 'sw_var': None, 'sw_median': None, 'sw_mean': None}

class SWTLocalizer:
    
    def __init__(self, show_report=True, multiprocessing=False):
        
        # Variable Initialisation
        self.show_report = show_report

        # Sanity Check for the object so created
        self.obj_sanity_check()

        self.has_custom_edge_func = False
        self.components_props = {}

    def obj_sanity_check(self):
        if not isinstance(self.show_report, bool):
            raise ValueError("Invalid 'show_report' type, should be of type 'bool'")
    


    def swttransform(self, imgpaths, save_results = False, save_rootpath = '../SWTlocResults/', *args, **kwargs):
        self.imgpaths = imgpaths
        self.save_rootpath = os.path.abspath(save_rootpath)
        self.sanity_check_transform(kwargs)

        # Progress bar to report the total done
        self.probar = prog_bar(self.imgpaths, len(self.imgpaths))
        for each_imgpath in self.probar:
            self.transform(imgpath=each_imgpath, **kwargs)
            if save_results:
                imgname = each_imgpath.split('/')[-1].split('.')[0]
                savepath = self.save_rootpath+f'/{imgname}'
                self.save(savepath=savepath)

    def save(self, savepath):
        
        os.makedirs(savepath, exist_ok=True)

        # Save Original Image
        imgsave(self.orig_img, title='Orignal', savepath=savepath+'/orig_img.jpg')
        
        # Save Original Image
        imgsave(self.grayedge_img, title='EdgeImage', savepath=savepath+'/edge_img.jpg')
        
        # Save Original Image
        imgsave(self.img_gradient, title='Gradient', savepath=savepath+'/grad_img.jpg')
        
        # Save Original Image
        imgsave(self.swt_labelled3C, title='SWT Transform', savepath=savepath+'/swt3C_img.jpg')
        
        # Save Original Image
        imgsave(self.swtlabelled_pruned13C, title='SWT Pruned', savepath=savepath+'/swtpruned3C_img.jpg')
        
    def sanity_check_transform(self, kwargs):
        
        # Check for imgpaths
        if isinstance(self.imgpaths, str) or isinstance(self.imgpaths, list):
            if isinstance(self.imgpaths, list):
                self.progress_ind = False
                for eachPath in self.imgpaths:
                    if not os.path.isfile(eachPath):
                        raise FileNotFoundError(f"No image present at {eachPath}")
                        
            if isinstance(self.imgpaths, str):
                if not os.path.isfile(self.imgpaths):
                    raise FileNotFoundError(f"No image present at {self.imgpaths}")
                self.imgpaths = [self.imgpaths] # Convert a single image path to a list
        else:
            raise ValueError("'imgpaths' argument needs to be of type 'str' or 'list'")

        # Save the savepath directory, if not there then make one
        if not os.path.isdir(self.save_rootpath):
            os.makedirs(self.save_rootpath, exist_ok=True)

        # Check for the kwargs for the transform function
        if 'text_mode' in kwargs:
            if not (kwargs['text_mode'] in ['db_lf', 'lb_df']):
                raise ValueError("'text_mode' should be one of ['db_lf', 'lb_df']")
        if 'gs_blurr' in kwargs:
            if not isinstance(kwargs['gs_blurr'], bool):
                raise ValueError("'gs_blurr' should be of type bool")
        if 'blurr_kernel' in kwargs:
            if not (isinstance(kwargs['blurr_kernel'], tuple) and all(isinstance(k, int) and (k%2!=0) and (k>=3) for k in kwargs['blurr_kernel']) and (kwargs['blurr_kernel'][0] == kwargs['blurr_kernel'][1])):
                raise ValueError("'blurr_kernel' should be of type tuple, and must contain integer odd values")
        if 'edge_func' in kwargs:
            if isinstance(kwargs['edge_func'], str) or callable(kwargs['edge_func']):
                if isinstance(kwargs['edge_func'], str):
                    if not (kwargs['edge_func'] in ['ac']):
                        raise ValueError("'edge_func' should be one of ['ac']")
                elif not callable(kwargs['edge_func']):
                    raise ValueError("'edge_func' custom function which returns an edged image.")
            else:
                raise ValueError("'edge_func' should be either 'ac' or callable")
        if 'ac_sigma' in kwargs:
            if not (isinstance(kwargs['ac_sigma'], float) and (0.0<=kwargs['ac_sigma']<=1.0)):
                raise ValueError("'ac_sigma' should be of type float and value between 0 and 1")
        if 'minrsw' in kwargs:
            if not (isinstance(kwargs['minrsw'], int) and kwargs['minrsw']>=3):
                raise ValueError("'minrsw' should be of type int and be more than 3")
        if 'maxrsw' in kwargs:
            if not( isinstance(kwargs['maxrsw'], int) and kwargs['maxrsw']>=3):
                raise ValueError("'maxrsw' should be of type int and be more than 3")
        if ('minrsw' in kwargs) and ('maxrsw' in kwargs):
            if kwargs['maxrsw'] <= kwargs['minrsw']:
                raise ValueError("'minrsw' should be smaller than 'maxrsw'")
        if 'max_angledev' in kwargs:
            if not (isinstance(kwargs['max_angledev'], float) and (-np.pi/2<=kwargs['max_angledev']<=np.pi/2)):
                raise ValueError("'max_angledev' should be a float and inbetween -90° <-> 90° (in radians)")
        if 'check_anglediff' in kwargs:
            if not (isinstance(kwargs['check_anglediff'], bool)):
                raise ValueError("'isinstance' should be type bool")
        if 'minCC_comppx' in kwargs:
            if not (isinstance(kwargs['minCC_comppx'], int) and kwargs['minCC_comppx']>0):
                raise ValueError("'minCC_comppx' should be type int")
        if 'maxCC_comppx' in kwargs:
            if not (isinstance(kwargs['maxCC_comppx'], int) and kwargs['maxCC_comppx']>0):
                raise ValueError("'maxCC_comppx' should be of type int")
        if 'acceptCC_aspectratio' in kwargs:
            if not (isinstance(kwargs['acceptCC_aspectratio'], int) and kwargs['acceptCC_aspectratio']>0):
                raise ValueError("'acceptCC_aspectratio' should be of type int and positive")


    def transform(self, imgpath, text_mode = 'lb_df',
                  gs_blurr = True, blurr_kernel = (5,5),
                  edge_func = 'ac', ac_sigma = 0.33,
                  minrsw=3, maxrsw=200, max_angledev=np.pi/6, check_anglediff=True,
                  minCC_comppx = 50, maxCC_comppx = 10000, acceptCC_aspectratio=5):
        """
        Entry Point for the Stroke Width Transform - Single Image
        """
        ts = time.perf_counter_ns()
        # Read the image..
        self.orig_img, origgray_img = self.image_read(imgpath=imgpath, gs_blurr=gs_blurr, blurr_kernel=blurr_kernel)

        # Find the image edge
        origgray_img, self.grayedge_img = self.image_edge(gray_image = origgray_img, edge_func = edge_func, ac_sigma = ac_sigma)

        # Find the image gradient
        self.img_gradient = self.image_gradient(orignal_img = origgray_img, edged_img=self.grayedge_img)
        hstep_mat  = np.round(np.cos(self.img_gradient), 5)
        vstep_mat  = np.round(np.sin(self.img_gradient), 5)
        if text_mode == 'db_lf':
            hstep_mat *= -1
            vstep_mat *= -1

        # Find the Stroke Widths in the Image
        self.swtObj = SWT(edgegray_img = self.grayedge_img, hstepmat = hstep_mat, vstepmat = vstep_mat, imggradient = self.img_gradient,
                     minrsw=minrsw, maxrsw=maxrsw, max_angledev=max_angledev, check_anglediff=check_anglediff)
        self.swt_mat = self.swtObj.find_strokes()

        # Find the connected Components in the image
        numlabels, self.swt_labelled = self.image_swtcc(swt_mat=self.swt_mat)
        self.swt_labelled3C = prepCC(self.swt_labelled)

        # Prune and Extract LabelComponet Properties
        self.swtlabelled_pruned1 = self.image_prune_getprops(orig_img = self.orig_img, swtlabelled=self.swt_labelled, minCC_comppx=minCC_comppx,
                                                        maxCC_comppx=maxCC_comppx, acceptCC_aspectratio = acceptCC_aspectratio)
        self.swtlabelled_pruned13C = prepCC(self.swtlabelled_pruned1)

        self.transform_time = str(np.round((time.perf_counter_ns() - ts)/1e9, 3))+' sec'



    def image_read(self, imgpath, gs_blurr = True, blurr_kernel = (5,5)):
        
        orig_img = cv2.imread(imgpath) # Read Image
        origgray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY) # Convert to Grayscale
        if gs_blurr:
            origgray_img = cv2.GaussianBlur(origgray_img, (5,5), 0)

        return orig_img, origgray_img

    def image_edge(self, gray_image, edge_func, ac_sigma):

        if edge_func == 'ac':
            image_edge = auto_canny(image=gray_image, sigma=ac_sigma)
        elif callable(edge_func):
            image_edge = edge_func(gray_image)

        return gray_image, image_edge

    def image_gradient(self, orignal_img, edged_img):
        
        rows,columns = orignal_img.shape[:2]
        dx = cv2.Sobel(orignal_img, cv2.CV_32F, 1, 0, ksize = 5, scale = -1, delta = 1, borderType = cv2.BORDER_DEFAULT)
        dy = cv2.Sobel(orignal_img, cv2.CV_32F, 0, 1, ksize = 5, scale = -1, delta = 1, borderType = cv2.BORDER_DEFAULT)

        theta_mat = np.arctan2(dy, dx)
        edgesbool = (edged_img != 0).astype(int)
        theta_mat = theta_mat*edgesbool
        
        return theta_mat

    def image_swtcc(self, swt_mat):
        
        threshmask = swt_mat.copy().astype(np.int16)
        threshmask[threshmask==np.max(threshmask)] = 0 # Set the maximum value(Diagonal of the Image :: Maximum Stroke Width) to 0
        threshmask[threshmask>0]=1
        threshmask = threshmask.astype(np.int8)

        num_labels, labelmask = cv2.connectedComponents(threshmask, connectivity=8)

        return num_labels, labelmask

    def image_prune_getprops(self, orig_img, swtlabelled, minCC_comppx, maxCC_comppx, acceptCC_aspectratio):

        swtlabelled_pruned = swtlabelled.copy()
        lc_count=print_valcnts(swtlabelled_pruned, _print=False)
        # Pruning based on min and max number of pixels in a connected component
        for label,count in lc_count.items():
            if count < minCC_comppx or count > maxCC_comppx:
                swtlabelled_pruned[swtlabelled_pruned==label] = 0
        lc_count=print_valcnts(swtlabelled_pruned, _print=False)

        # Pruning based on a Aspect Ratio
        for label, pixel_count in lc_count.items():

            lmask = (swtlabelled_pruned==label).astype(np.uint8).copy()

            cntrs = cv2.findContours(lmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
            
            rotrect = cv2.minAreaRect(cntrs[0])
            label_height = np.round(max(rotrect[1]), 2)
            label_width = np.round(min(rotrect[1]), 2)
            label_aspectratio = label_width/label_height

            if not ((1/acceptCC_aspectratio) < label_aspectratio < acceptCC_aspectratio):
                swtlabelled_pruned[swtlabelled_pruned==label] = 0
            else:
                bbm_cx, bbm_cy = np.round(rotrect[0],2)
                bbm_bbox = cv2.boxPoints(rotrect)
                
                anchor_point = bbm_bbox[np.argmax((bbm_bbox==np.min(bbm_bbox[:,0])).sum(axis=1))]
                remain_point = np.array([k for k in bbm_bbox if (k != anchor_point).any()])
                all_lengths = [np.linalg.norm(k-anchor_point) for k in remain_point]
                anchor_armlength_point = remain_point[all_lengths == np.sort(all_lengths)[1]][0]
                
                bbox_ang = np.arctan(-(anchor_armlength_point[1] - anchor_point[1])/(anchor_armlength_point[0] - anchor_point[0]))
                bbox_ang = np.rad2deg(bbox_ang)
                if bbox_ang < 0:
                    bbox_ang = 180+bbox_ang

                self.components_props[label] = COMPONENT_PROPS.copy()
                self.components_props[label]['pixels'] = pixel_count
                self.components_props[label]['bbm_h'] = label_height
                self.components_props[label]['bbm_w'] = label_width
                self.components_props[label]['bbm_cx'] = bbm_cx
                self.components_props[label]['bbm_cy'] = bbm_cy
                self.components_props[label]['bbm_ar'] = label_aspectratio
                self.components_props[label]['bbm_bbox'] = bbm_bbox
                self.components_props[label]['bbm_anchor'] = anchor_point
                self.components_props[label]['bbm_outline'] = cntrs
                self.components_props[label]['bbm_ang'] = bbox_ang

                _iy, _ix = lmask.nonzero()
                mean_rgbcolor = self.orig_img[_iy, _ix].mean(axis=0)
                median_rgbcolor = np.median(self.orig_img[_iy, _ix], axis=0)
                self.components_props[label]['img_color_mean'] = str(list(np.floor(mean_rgbcolor)))
                self.components_props[label]['img_color_median'] = str(list(np.floor(median_rgbcolor)))

                sw_xyvals = self.swt_mat[_iy, _ix].copy()
                sw_countdict=print_valcnts(sw_xyvals, _print=False, remove_0=False)

                self.components_props[label]['sw_countdict'] = str(sw_countdict)
                self.components_props[label]['sw_var'] = np.var(sw_xyvals)
                self.components_props[label]['sw_median'] = np.median(sw_xyvals)
                self.components_props[label]['sw_mean'] = np.mean(sw_xyvals)

        return swtlabelled_pruned


    # Get the Grouping and Bounding BBoxes
    def get_min_bbox(self, show = False, padding = 5):
        
        min_bboxes = []
        temp = self.swtlabelled_pruned13C.copy()
        for label, labelprops in self.components_props.items():
            bbm_bbox = np.int32(labelprops['bbm_bbox'])
            
            # Calculate centre coordinates
            _tr,_br,_bl,_tl = bbm_bbox.copy()
            _d1_vec = _tr-_bl
            _d2_vec = _tl-_br
            _d1_ang = -math.atan2(_d1_vec[1], _d1_vec[0])
            _d2_ang = -math.atan2(_d2_vec[1], _d2_vec[0])
            
            _tr = _tr+padding*np.array([np.cos(_d1_ang),-np.sin(_d1_ang)])
            _br = _br-padding*np.array([-np.cos(np.pi-_d2_ang),-np.sin(np.pi-_d2_ang)])
            _bl = _bl-padding*np.array([-np.cos(np.pi-_d1_ang),-np.sin(np.pi-_d1_ang)])
            _tl = _tl+padding*np.array([np.cos(_d2_ang),-np.sin(_d2_ang)])
            bbm_bbox = np.c_[_tr,_br,_bl,_tl].T.astype(int)
            
            min_bboxes.append(bbm_bbox)
            temp = cv2.polylines(temp, [bbm_bbox], True, (0,0,255), 1)
        
        if show:
            imgshow(temp, 'Minimum Bounding Box')

        return min_bboxes, temp

    def get_extreme_bbox(self, show = False, padding = 5):
        
        ext_bboxes = []
        temp1 = self.swtlabelled_pruned1.copy()
        temp2 = self.swtlabelled_pruned13C.copy()

        for label, labelprops in self.components_props.items():
            lmask = (temp1 == label).astype(np.uint16)
            if np.sum(lmask)>0:
                _iy, _ix = lmask.nonzero()
                _tr = [max(_ix)+padding, min(_iy)-padding]
                _br = [max(_ix)+padding, max(_iy)+padding]
                _bl = [min(_ix)-padding, max(_iy)+padding]
                _tl = [min(_ix)-padding, min(_iy)-padding]
                bbe_bbox = np.c_[_tr,_br,_bl,_tl].T.astype(int)
                ext_bboxes.append(bbe_bbox)
        
                temp2 = cv2.polylines(temp2, [bbe_bbox], True, (0,0,255), 1)
        
        if show:
            imgshow(temp2, 'Extreme Bounding Box')

        return ext_bboxes, temp2

    def get_comp_outline(self, show = False, padding = 5):
        
        outlines = []
        temp = self.swtlabelled_pruned13C.copy()
        for label, labelprops in self.components_props.items():
            loutline = labelprops['bbm_outline']
            outlines.append(loutline)
    
            temp = cv2.polylines(temp, loutline, True, (0,0,255), 1, 4)
        
        if show:
            imgshow(temp, 'Component Outlines')

        return outlines, temp

    def get_grouped(self, lookup_radii_multiplier=0.8, sw_ratio=2,
                    cl_deviat=[13,13,13], ht_ratio=2, ar_ratio=3, ang_deviat=30):
        
        bubbleBbox = BubbleBBOX(labelmask = self.swtlabelled_pruned1, comp_props = self.components_props, lookup_radii_multiplier=lookup_radii_multiplier, 
                                sw_ratio=sw_ratio, cl_deviat=cl_deviat, ht_ratio=ht_ratio, ar_ratio=ar_ratio,
                                ang_deviat=ang_deviat)
        grouped_labels, grouped_bubblebbox, grouped_annot_bubble, grouped_annot, maskviz, maskcomb = bubbleBbox.run_grouping()

        return grouped_labels, grouped_bubblebbox, grouped_annot_bubble, grouped_annot, maskviz, maskcomb

