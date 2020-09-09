import numpy as np

class SWT:
    def __init__(self, edgegray_img, hstepmat, vstepmat, imggradient, minrsw = 3, maxrsw = 200,
                 max_angledev = np.pi/6, check_anglediff = True):

        self.edgegray_img = edgegray_img
        self.hstep_mat = hstepmat
        self.vstep_mat= vstepmat
        self.grad_theta = imggradient

        self.edgey, self.edgex = self.edgegray_img.nonzero()
        self.edge_indexes = set(zip(self.edgey, self.edgex))

        self.max_h = self.edgegray_img.shape[0]
        self.max_w = self.edgegray_img.shape[1]
        self.max_sw = int(np.sqrt(self.max_h**2+self.max_w**2))+5 # Plus 5 for safety sake

        # Initialise with the MAXIMUM Possible Width there can possibly be..
        self.swt_mat = np.ones(self.edgegray_img.shape, dtype=np.uint16)*self.max_sw 
        self.max_rsw = maxrsw  # Max ray stroke width. Needs to be smaller than the diagonal
        self.min_rsw = minrsw  # Min ray stroke width. Needs to be bigger than the 1
        self.max_angle_dev = max_angledev

        self.check_angle_diff = check_anglediff

    def get_raylength(self, ray):
        p1 = ray[0]
        p2 = ray[-1]
        return int(np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2))

    def next_point_gen(self, _oyi, _oxi, _delx, _dely):
        for i in range(self.max_sw):
            yield int(np.floor(_oyi+i*_dely)), int(np.floor(_oxi+i*_delx))
        
    def check_filtering(self, _oyi, _oxi, _nyi, _nxi):
        """
        Check three conditions:
            1.) If the raylength is smaller than or equal to the 'max_stroke_width'
            2.) if the last point in the ray is within the image bounds
            3.) If the given point is in the edge indexes
            4.) if the angle difference is between 180°-30° and 180°+30°
        """

        ray_length = self.get_raylength([(_oyi, _oxi), (_nyi, _nxi)])
        
        check1 =  ray_length <= self.max_rsw
        check2 = (0 <= _nyi < self.max_h) and (0 <= _nxi < self.max_w)
        check3 = (_nyi, _nxi) not in self.edge_indexes

        if check1 and check2 and check3:
            return True, -1
        else:
            if not check3:
                if self.check_angle_diff:
                    theta1 = self.grad_theta[_oyi, _oxi]
                    theta2 = self.grad_theta[_nyi, _nxi]
                    theta_diff = abs(theta1-theta2)
                    check4 = np.pi-self.max_angle_dev <= theta_diff <= np.pi+self.max_angle_dev

                    if check4:
                        return False, ray_length
                    else:
                        return False, -1
                else:
                    return False, ray_length
            return False, -1

    def new_edge(self):
        for edge in self.edge_indexes:
            yield edge

    def find_strokes(self):

        for _yi, _xi in  zip(self.edgey, self.edgex):
            rayidx = []
            _delx = self.hstep_mat[_yi, _xi]
            _dely = self.vstep_mat[_yi, _xi]

            nextp_gen = self.next_point_gen(_oyi = _yi, _oxi = _xi,
                                            _delx = _delx, _dely = _dely)
            rayidx.append(next(nextp_gen))
            stroke_len = -1
            for _nextp in nextp_gen:
                if _nextp not in rayidx:
                    _proceed, stroke_len = self.check_filtering(_yi, _xi, _nextp[0], _nextp[1])
                    if not _proceed:
                        break
                    rayidx.append(_nextp)
            
            rayidx = np.array(rayidx)
            if stroke_len == -1:
                rayidx = np.array([rayidx[0]])
                sw = 1
            else:
                prevals = list(self.swt_mat[rayidx[:,0], rayidx[:,1]])
                sw = min(prevals+[stroke_len])
            self.swt_mat[rayidx[:,0], rayidx[:,1]] = sw

        return self.swt_mat


























