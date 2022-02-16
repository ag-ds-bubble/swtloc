# Author : Achintya Gupta

import numpy as np
from cv2 import cv2


class BubbleBBOX:

    def __init__(self, labelmask, comp_props, lookup_radii_multiplier=0.8,
                 sw_ratio=2, cl_deviat=[13, 13, 13], ht_ratio=2, ar_ratio=3, ang_deviat=30,
                 bubble_width=1):

        self.labelmask = labelmask.copy()
        self.lookup_radii_multiplier = lookup_radii_multiplier
        self.h, self.w = self.labelmask.shape[:2]
        self.maskviz = np.zeros(self.labelmask.shape)
        self.maskcomb = np.zeros(self.labelmask.shape)
        self.comp_props = comp_props.copy()
        self.comp_dstack = []

        self.sw_ratio = sw_ratio
        self.cl_deviat = np.linalg.norm(cl_deviat)
        self.ht_ratio = ht_ratio
        self.ar_ratio = ar_ratio
        self.ang_deviat = ang_deviat
        self.grouped_labels = []
        self.ungrouped_labels = set(list(self.comp_props.keys()))

        # Asthetics
        self.bubble_width = bubble_width

        self.sanity_checks()

    def sanity_checks(self):
        # Check for the Bubble Widths
        if not isinstance(self.bubble_width, int):
            raise ValueError("'bubble_width' parameter should be of type in 'int'")

    def create_circular_mask(self, center, radius):

        if center is None:  # use the middle of the image
            center = (int(self.w / 2), int(self.h / 2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], self.w - center[0], self.h - center[1])
        Y, X = np.ogrid[:self.h, :self.w]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        mask = dist_from_center <= radius
        return mask

    def generate_comp_bubble(self):

        for label, props in self.comp_props.items():
            label_ct = np.array([props['bbm_cy'], props['bbm_cx']]).astype(np.uint16)
            label_bx = props['bbm_bbox']
            label_an = props['bbm_anchor']
            radii = max([np.linalg.norm(epnt[::-1] - label_ct) for epnt in label_bx]) * self.lookup_radii_multiplier
            cv2.putText(self.maskviz, str(label), tuple(label_an.astype(int)),
                        cv2.FONT_HERSHEY_PLAIN, 5,
                        2, 1, cv2.LINE_AA)
            cv2.polylines(self.maskviz, np.int32([label_bx]), True, 1, 1)
            mask = self.create_circular_mask(label_ct[::-1], radii)
            self.comp_dstack.append(mask * label)
            self.maskcomb += mask

        self.comp_dstack = np.dstack(tuple(self.comp_dstack))

    def get_attr(self, label, mode='component'):

        props = self.comp_props[label]

        if mode == 'component':
            sw = props['sw_median']
            ar = props['bbm_ar']
            co = np.array(eval(props['img_color_median']))
            ht = props['bbm_h']
            wt = props['bbm_w']
            ang = props['bbm_ang']
            return sw, ar, co, ht, wt, ang

        elif mode == 'proximity':
            ct = np.array([props['bbm_cy'], props['bbm_cx']]).astype(np.uint16)
            bx = props['bbm_bbox']
            return ct, bx

    def get_proximity_labels(self, label):

        _properties = self.get_attr(label, mode='proximity')
        label_ct, label_bx = _properties
        radii = max([np.linalg.norm(epnt[::-1] - label_ct) for epnt in label_bx]) * self.lookup_radii_multiplier
        mask = self.create_circular_mask(label_ct[::-1], radii)
        maxk_y, mask_x = mask.nonzero()
        proximty_labels = np.setdiff1d(np.unique(self.comp_dstack[maxk_y, mask_x, :]), [0, label])
        return proximty_labels

    def grouping_check(self, label1, label2):

        label1_props = self.get_attr(label1, mode='component')
        label_sw, label_ar, label_co, label_ht, label_wt, label_ang = label1_props

        label2_props = self.get_attr(label2, mode='component')
        comp_sw, comp_ar, comp_co, comp_ht, comp_wt, comp_ang = label2_props

        # Check for the Stroke Width Ratio
        check1 = (1 / self.sw_ratio) <= (comp_sw / label_sw) <= self.sw_ratio
        # Check for the Color Deviation
        check2 = np.linalg.norm(label_co - comp_co, axis=0) <= self.cl_deviat
        # Check for the Height Ratio
        check3 = (1 / self.ht_ratio) <= (comp_ht / label_ht) <= self.ht_ratio
        # Check for the Angle Deviation
        diff1 = np.abs(comp_ang - label_ang)
        diff2 = np.abs(90 - comp_ang - label_ang)
        check4 = any(k <= self.ang_deviat for k in [diff1, diff2])
        # Check for the Aspect Ratio
        check5 = (1 / self.ar_ratio) <= (comp_ar / label_ar) <= self.ar_ratio
        return check1 and check2 and check3 and check4 and check5

    def grouplabel(self, label, bucket):
        proxim_labels = self.get_proximity_labels(label=label)

        proxim_labels = [k for k in proxim_labels if k not in bucket]
        if proxim_labels == []:
            return bucket

        for each_pl in proxim_labels:
            if self.grouping_check(label1=label, label2=each_pl):
                bucket.append(each_pl)
                bucket = self.grouplabel(label=each_pl, bucket=bucket)
        return bucket

    def run_grouping(self):
        self.generate_comp_bubble()
        while len(self.ungrouped_labels) > 0:
            curr_label = list(self.ungrouped_labels)[0]
            curr_bucket = self.grouplabel(label=curr_label, bucket=[curr_label])
            self.grouped_labels.append(curr_bucket)
            self.ungrouped_labels = self.ungrouped_labels.difference(set(curr_bucket))

        self.grouped_bubblebbox = []

        self.grouped_annot_bubble = np.zeros(self.labelmask.shape, dtype=np.uint8)
        self.grouped_annot_bubble = cv2.cvtColor(self.grouped_annot_bubble, cv2.COLOR_GRAY2BGR)

        self.grouped_annot = np.zeros(self.labelmask.shape, dtype=np.uint8)
        self.grouped_annot = cv2.cvtColor(self.grouped_annot, cv2.COLOR_GRAY2BGR)

        for each_group in self.grouped_labels:
            mask = np.zeros(self.labelmask.shape, dtype=np.uint8)
            for each_label in each_group:
                label_ct, label_bx = self.get_attr(each_label, mode='proximity')
                radii = max([np.linalg.norm(epnt[::-1] - label_ct) for epnt in label_bx])
                mask += self.create_circular_mask(label_ct[::-1], radii)

            contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            self.grouped_bubblebbox.append(contours)

            mask = np.zeros(self.labelmask.shape, dtype=np.uint8)
            for each_label in each_group:
                mask += self.labelmask == each_label
            mask *= 255

            self.grouped_annot_bubble += cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            self.grouped_annot += cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)
            cv2.drawContours(self.grouped_annot_bubble, contours, -1, (0, 0, 255), self.bubble_width)

            rotrect = cv2.minAreaRect(contours[0])
            combbbox = cv2.boxPoints(rotrect)
            self.grouped_annot += cv2.polylines(self.grouped_annot, np.int32([combbbox]), True, (0, 0, 255), 2)

        return self.grouped_labels, self.grouped_bubblebbox, self.grouped_annot_bubble, self.grouped_annot, self.maskviz, self.maskcomb
