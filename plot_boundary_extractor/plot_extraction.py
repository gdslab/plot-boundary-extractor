import os
import math
import itertools
import numpy as np
import cv2
import rasterio
from rasterio.warp import transform_geom
from rasterio.features import rasterize

from PIL import Image
from geopandas import gpd
import pandas as pd

from skimage.transform import probabilistic_hough_line
from skimage.measure import label
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN

from shapely.geometry import Polygon as sPolygon
from shapely.geometry import Point, box, mapping
from shapely.affinity import rotate

from rs_tools import LightImage,pixel2world,polygonize, norm_min_max

from sympy import plot
import torch
from torchvision.ops import nms
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from affine import Affine

import matplotlib.pyplot as plt


class PlotExtraction(LightImage):
    
    def __init__(self, **args):
        
        # load data product
        self.data_product = args['base_layer']
        if args.get('api_key') is not None:
            self.data_product_url = f"{self.data_product.url}?API_KEY={args.get('api_key')}"
        else:
            self.data_product_url = self.data_product.url
        
        # load image
        if args.get('clipped_filename') is not None:
            self.filename = args.get('clipped_filename')

        self.boundary = args.get('clip_boundary', None)
        
        # load sam model
        self.sam_checkpoint = args['sam_checkpoint']
        self.points_per_side = args.get('points_per_side', 32)
        if args.get('resize') is not None:
            self.resize = args.get('resize')
        else:
            self.resize = None
        
        # plot parameters
        self.plot_width = args['plot_width']
        self.plot_height = args['plot_height']
        self.delta_area = args.get('delta_area', 1)
        
        # grid parameters
        self.n_rows = args['n_rows']
        self.n_cols = args['n_cols']
        
        # thresholds
        self.iou_threshold = args.get('iou_threshold', 0.1)
        self.dist_thr = args.get('dist_thr', 1)
        self.dist_thr2 = args.get('dist_thr2', 1)
        self.cc_coverage_thr = args.get('cc_coverage_thr', 0.0)
        
        # output
        if args.get('out_filename') is not None:
            self.out_filename = args.get('out_filename')
        else:
            self.out_filename = None
            
        
    def load_image(self):
        
        if self.boundary is not None:
            if os.path.exists(self.filename):
               pass
            else:
                self.data_product.clip(geojson_feature=self.boundary, 
                                   out_raster=self.filename)
            
        super().__init__(self.filename)
        self.img_array = self.get_img_array()[:,:,:3]
        if self.img_array.dtype != np.uint8:
            self.img_array = norm_min_max(self.img_array, 0, 255).astype(np.uint8)
        self.img_width = self.img_array.shape[1]
        self.img_height = self.img_array.shape[0]
        self.cc_map = self.canopeo(self.img_array)
        self.epsg = self.projection.GetAttrValue('AUTHORITY', 1)
        self.geometry = box(self.ext_left, self.ext_down, self.ext_right, self.ext_up)
        
        
    def canopeo(self, arr, th1=0.95, th2=0.95, th3=20):
        
        # Read bands
        red = arr[:,:,0].astype(np.float32)
        green = arr[:,:,1].astype(np.float32)
        blue = arr[:,:,2].astype(np.float32)

        # Find canopy cover
        i1 = red / green
        i2 = blue / green
        i3 = (2*green - blue - red)
        # i4 = (red + green + blue)
        # i5 = (green - blue)
        cond1 = i1 < th1
        cond2 = i2 < th2
        cond3 = i3 > th3
        # cond4 = i4 < th4
        # cond5 = i5 > th5
        cond = (cond1 * cond2 * cond3) * 255
        
        return cond
    
    def load_sam(self, sam_checkpoint, type='automatic', model='vit_h', points_per_side=32):
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        sam = sam_model_registry[model](checkpoint=sam_checkpoint)
        sam.to(device=device)
        
        if type=='automatic':
            self.mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                # points_per_side=np.max([img_dict[img_suffix]['n_cols'],img_dict[img_suffix]['n_rows']]),
                points_per_side=points_per_side,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,  # Requires open-cv to run post-processing
            )
            print(f"Loaded SAM automatic maskgenerator: points per side={points_per_side}, device={device}")
        elif type=='manual':
            self.predictor = SamPredictor(sam)
            print("Loaded SAM predictor")
        
        
    def clear_sam(self):
        
        self.mask_generator = None
        self.predictor = None
        
        del self.mask_generator
        del self.predictor      
        
    def get_masks(self):
        
        ratio = (self.n_cols * self.plot_width) / (self.n_rows * self.plot_height)
        if ratio > 2:
            ratio = 2
        self.ratio = ratio
        
        if self.resize is None:
            img_size = 1500**2
            if ratio > 1:
                resize = (int(np.sqrt(img_size/ratio)), int(np.sqrt(img_size/ratio) * ratio))                
            else:                
                ratio = 1/ratio
                resize = (int(np.sqrt(img_size/ratio) * ratio), int(np.sqrt(img_size/ratio)))
            self.resize = resize
        
        print(f"Resized image: {self.resize}")
        
        self.scale_col = self.img_width / self.resize[1]
        self.scale_row =  self.img_height/ self.resize[0]
        img_resize = cv2.resize(self.img_array,(self.resize[1],self.resize[0]),interpolation=cv2.INTER_NEAREST)
        self.img_resize = img_resize
        
        masks = self.mask_generator.generate(img_resize)
        
        mask_arr = np.zeros((img_resize.shape[:2]))
        for mask_ in masks:
            mask_arr += mask_['segmentation'] * np.random.randint(1,len(masks))
        self.mask_arr = cv2.resize(mask_arr, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        
        return self.mask_arr
    
    def rotate_plot(self, line_length=100, line_gap=10):
        
        if self.ratio > 1:
            offset = 180
        else:
            offset = 90
        
        mask = np.array(self.mask_arr, dtype=np.uint8)*255
        edge = cv2.Canny(mask, 50, 150, apertureSize=3)
        
        edge_full = Image.fromarray(edge).resize((self.ncol, self.nrow))
        lines = probabilistic_hough_line(np.array(edge_full), threshold=10, line_length=line_length, line_gap=line_gap)
        
        slope = []
        for line in lines:
            p0, p1 = line
            eps = 1E-6
            slope.append(np.degrees(math.atan2((p0[1]-p1[1]+eps),(p1[0]-p0[0]+eps))))
            
        self.slope_deg = offset - np.median(slope)
        print(f"Estimated orientation angle: {self.slope_deg:.2f} degree")
        
        center_x = self.img_width/2
        center_y = self.img_height/2
        self.center_img = (center_x,center_y)
        
        center_x_geo, center_y_geo = pixel2world(self.geotransform, center_y, center_x)
        self.center_geo = (center_x_geo, center_y_geo)
        
        self.img_rotated = Image.fromarray(self.img_array).rotate(self.slope_deg, center=(center_x,center_y))
        
        return self.img_rotated
    
    
    def mask2polygon(self, masks):
        
        polygons={'id':[],'geometry':[]}
        # mask_raster=[]
        
        i=0
        for mask in masks:
            mask_, mask_num = label(mask, return_num=True)
            for j in range(mask_num):
                mask_tmp = (mask_==(j+1))
                polygon = polygonize(mask_tmp, convex=True, simplify_tolerance=1)[0]
                xy = np.array(polygon.exterior.xy).transpose() * np.array([self.scale_col, self.scale_row])
                x = xy[:,0]
                y = -xy[:,1]
                transformed_coords = (np.array([x,y])*self.x_spacing + np.array([[self.ext_left],[self.ext_up]])).transpose()
                polygon_loc = sPolygon(transformed_coords)
                
                if polygon_loc.area > (self.plot_width*self.plot_height)-self.delta_area and polygon_loc.area < (self.plot_width*self.plot_height)+self.delta_area:
                    polygons['id']=i
                    polygons['geometry'].append(polygon_loc)
                    # mask_raster.append(mask_tmp)
                    i+=1
        
        x_loc,y_loc,cent_loc = [],[],[]
        for polygon_loc in polygons['geometry']:
            # x,y, and centroid in geo-coordinate
            x_loc.append(np.array(polygon_loc.exterior.xy[0]))
            y_loc.append(np.array(polygon_loc.exterior.xy[1]))
            cent_loc.append([polygon_loc.centroid.x, polygon_loc.centroid.y])
                
        plot_rotated = {'id':[],'geometry':[]}

        for i,(x_loc,y_loc) in enumerate(cent_loc):
                
            ul = [x_loc - self.plot_width/2, y_loc + self.plot_height/2]
            ur = [x_loc + self.plot_width/2, y_loc + self.plot_height/2]
            br = [x_loc + self.plot_width/2, y_loc - self.plot_height/2]
            bl = [x_loc - self.plot_width/2, y_loc - self.plot_height/2]
            
            bbox = np.array([ul,ur,br,bl,ul])
            
            plot_rotated['id'].append(i)
            plot_rotated['geometry'].append(sPolygon(bbox))
            
        gdf = gpd.GeoDataFrame(plot_rotated, geometry='geometry')
        
        if self.cent_local:
            self.cent_local = cent_loc
        else:
            self.cent_local += cent_loc
        
        return gdf
        
        
    def initial_plots(self):
        
        self.cent_local = []
        img_rotated_resize = np.array(self.img_rotated.resize((self.resize[1],self.resize[0])), dtype=np.uint8)
        masks = self.mask_generator.generate(img_rotated_resize)
        mask_list = [mask['segmentation'] for mask in masks]
        # self.initial_mask_list = mask_list
        
        gdf_initial = self.mask2polygon(mask_list)
        self.gdf_initial = gdf_initial

        
        # polygons_arr = np.zeros(self.resize)
        # for i,mask in enumerate(mask_raster):
        #     polygons_arr += mask * (i+1)
        # filtered_mask = cv2.resize(np.array(polygons_arr, dtype=np.uint8), 
        #                            (self.img_width,self.img_height), interpolation=cv2.INTER_NEAREST)
        
        # self.initial_mask = mask_raster
        
        return gdf_initial
        
                
    def grid_filling(self):
        
        x_l,y_l = np.array(self.cent_local)[:,0],np.array(self.cent_local)[:,1]

        # # Combine the x and y coordinates into a single array
        # points = np.column_stack((x_l, y_l))

        # # Step 1: Use DBSCAN to detect outliers
        # # eps: maximum distance between points to be considered in the same neighborhood
        # # min_samples: minimum number of points to form a cluster
        # db = DBSCAN(eps=max(self.plot_height,self.plot_width), min_samples=3).fit(points)

        # # Step 2: Identify the core points (clusters) and noise points (outliers)
        # labels = db.labels_

        # # Points labeled as -1 are outliers
        # core_points = points[labels != -1]
        # x_l = core_points[:,0]
        # y_l = core_points[:,1]
        
        inliers = (x_l > self.ext_left) & (x_l < self.ext_right) & (y_l > self.ext_down) & (y_l < self.ext_up)
        x_l = x_l[inliers]
        y_l = y_l[inliers]

        # get del height
        ind = np.lexsort((-y_l.round(1),x_l.round(1)))
        x_ord,y_ord = x_l[ind],y_l[ind]
        dist_list=[]
        for i in range(len(x_ord)-1):
            dist = np.linalg.norm([x_ord[i+1]-x_ord[i], y_ord[i+1]-y_ord[i]])
            dist_list.append(dist)

        hist, edges = np.histogram(dist_list, np.arange(0,int(np.ceil(np.max(dist_list)))))
        dist_arr = np.array(dist_list)
        most_freq_edge = edges[np.argmax(hist)]
        ind_ = np.where((most_freq_edge<=dist_arr) & (dist_arr<(most_freq_edge+1)))
        del_height = np.mean(dist_arr[ind_]) - self.plot_height
        self.del_height = del_height
        
        # get del width
        ind = np.lexsort((x_l.round(1), y_l.round(1)))
        x_ord,y_ord = x_l[ind],y_l[ind]
        dist_list=[]
        for i in range(len(x_ord)-1):
            dist = np.linalg.norm([x_ord[i+1]-x_ord[i], y_ord[i+1]-y_ord[i]])
            dist_list.append(dist)

        hist, edges = np.histogram(dist_list, np.arange(0,int(np.ceil(np.max(dist_list)))))
        dist_arr = np.array(dist_list)
        most_freq_edge = edges[np.argmax(hist)]
        ind_ = np.where((most_freq_edge<=dist_arr) & (dist_arr<(most_freq_edge+1)))
        del_width = np.mean(dist_arr[ind_]) - self.plot_width
        self.del_width = del_width
        
        # grid fill
        xv, yv = np.meshgrid(np.linspace(x_l.min(),x_l.max(),self.n_cols), np.linspace(y_l.min(),y_l.max(),self.n_rows))
        self.grid_coords = np.array([xv.flatten(),yv.flatten()])
        
        rc=[]
        for r,c in itertools.product(range(self.n_rows),range(self.n_cols)):
            x,y = xv[r,c],yv[r,c]
            min_dist = min(np.sqrt((x_ord - x)**2 + (y_ord - y)**2))
            # missing_ind = np.where(distances < min(self.plot_width, self.plot_height) - max(del_width, del_height))[0]
            if min_dist > min(self.plot_width/2, self.plot_height/2):
                rc.append([r,c])
        
        # plt.scatter(xv,yv , label='grid')
        # plt.scatter(x_l,y_l, label='Initial')
        # plt.scatter(xv[np.array(rc)[:,0],np.array(rc)[:,1]],
        #             yv[np.array(rc)[:,0],np.array(rc)[:,1]],
        #             label='filled')
        # plt.legend()
        # plt.show()           
        
        plot_size = ((-self.plot_height/self.y_spacing), (self.plot_width/self.x_spacing))
        mask_size = (500,500)
        
        print(f"Detected missing plots: {len(rc)}")
        added_mask = []
        for r, c in rc:
            # print(r,c)
            x, y = [xv[r, c], yv[r, c]]
            x_img, y_img = (np.array([x, y]) - np.array([self.ext_left, self.ext_down])) / self.x_spacing
            y_img = self.nrow - y_img
            
            # Define the bounding box for clipping
            xmin = int(x_img - mask_size[1])
            xmax = int(x_img + mask_size[1])
            ymin = int(y_img - mask_size[0])
            ymax = int(y_img + mask_size[0])
            
            if xmin<0:
                x_center = int((xmax + xmin)/2)
                xmin = 0
            else:
                x_center  = int((xmax + xmin)/2)
                
            if ymin<0:
                y_center = int((ymax + ymin)/2)
                ymin = 0
            else:
                y_center = int((ymax + ymin)/2)
            
            # Define the bounding box for the box prompt
            box_xmin = max(0, int((x_center-xmin - plot_size[1]/2)))
            box_xmax = max(0, int((x_center-xmin + plot_size[1]/2)))
            box_ymin = max(0, int((y_center-ymin - plot_size[0]/2)))
            box_ymax = max(0, int((y_center-ymin + plot_size[0]/2)))
            
            xmin = max(0, int(x_img - mask_size[1]))
            xmax = max(0, int(x_img + mask_size[1]))
            ymin = max(0, int(y_img - mask_size[0]))
            ymax = max(0, int(y_img + mask_size[0]))
            
            # Clip the image
            clipped_img = np.array(self.img_rotated)[ymin:ymax, xmin:xmax]
            
            # Generate mask for the clipped image
            self.predictor.set_image(clipped_img)
            
            # # box prompt segmentation
            # mask, scores, _ = self.predictor.predict(
            #     point_coords = None,
            #     point_labels = None,
            #     box=np.array([box_xmin,box_ymin,box_xmax,box_ymax]),
            #     multimask_output=True,
            # )
            
            # point prompot segmentation 
            mask, _, _ = self.predictor.predict(
            point_coords=np.array([[x_center-xmin , y_center-ymin]]),
            point_labels=np.array([1]),
            multimask_output=False,
            )
            
            mask = mask[0, :, :]
            
            # plt.imshow(clipped_img)
            # # plt.plot([box_xmin,box_xmax,box_xmax,box_xmin,box_xmin],
            # #          [box_ymin,box_ymin,box_ymax,box_ymax,box_ymin],
            # #          c='r')
            # plt.scatter(x_center-xmin, y_center-ymin, c='r')
            # plt.imshow(mask, alpha=0.5, cmap='cividis')
            # plt.show()
            
            # Resize the mask to match the resized image dimensions
            mask_big = np.zeros(self.img_rotated.size[::-1], dtype=np.uint8)
            mask_big[ymin:ymax, xmin:xmax] = mask
            
            mask = cv2.resize(mask_big, (self.resize[1], self.resize[0]), interpolation=cv2.INTER_NEAREST)
                   
            added_mask.append(mask)
            
        # plt.imshow(np.sum(added_mask, axis=0))
            
                
        '''
        img_rotated_resize = np.array(self.img_rotated.resize((self.resize[1],self.resize[0])))
        self.predictor.set_image(np.array(img_rotated_resize))

        added_mask = []
        for r,c in rc:
            x,y = [xv[r,c],yv[r,c]]
            x_img, y_img = (np.array([x,y]) - np.array([self.ext_left,self.ext_down])) / self.x_spacing
            x_img_resized = x_img/self.scale_col
            y_img_resized = self.resize[0] - y_img/self.scale_row
            
            # # create a box for box prompt segmentation
            # xmin = int(x_img_resized - ((self.plot_width/self.x_spacing)/2)/self.scale_col)
            # xmax = int(x_img_resized + ((self.plot_width/self.x_spacing)/2)/self.scale_col)
            # ymin = int(y_img_resized - ((self.plot_height/self.y_spacing)/2)/self.scale_row)
            # ymax = int(y_img_resized + ((self.plot_height/self.y_spacing)/2)/self.scale_row)
            
            # mask, scores, _ = self.predictor.predict(
            #     point_coords = None,
            #     point_labels = None,
            #     box=np.array([xmin,ymin,xmax,ymax]),
            #     multimask_output=True,
            # )
            # bestidx = np.argmax(scores)
            # mask = mask[bestidx,:,:]
            
            mask, _, _ = self.predictor.predict(
                point_coords = np.array([[int(x_img_resized), int(y_img_resized)]]),
                point_labels = np.array([1]),
                multimask_output=True,
            )
            mask = mask[0,:,:]
            
            added_mask.append(mask)
        '''
        #     plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],c='r')
        
        # plt.figure(figsize=(10,10))
        # plt.imshow(img_rotated_resize)
        # plt.scatter(x_img_resized, y_img_resized, c='r')
        # plt.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],c='r')
        # plt.show()
    
        # plt.imshow(np.sum(added_mask, axis=0))
        # plt.show()
            
        # self.added_mask = added_mask
        # self.mask = self.initial_mask + self.added_mask        
        gdf_added = self.mask2polygon(added_mask)
        
        # cent_added = []
        # for mask in added_mask:
        #     mask_, mask_num = label(mask, return_num=True)
        #     for j in range(mask_num):
        #         mask_tmp = (mask_==(j+1))
        #         polygon = polygonize(mask_tmp, convex=True, simplify_tolerance=1)[0]
        #         if polygon.type == 'Polygon':
        #             xy = np.array(polygon.exterior.xy)
        #             xy[1, :] = self.resize[0] - xy[1, :]
        #             xy *= np.array([[self.scale_col], [self.scale_row]])
        #             x = xy[0,:]
        #             y = xy[1,:]
        #             transformed_coords = (np.array([x,y])*self.x_spacing + np.array([[self.ext_left],[self.ext_down]])).transpose()
        #             polygon_loc = sPolygon(transformed_coords)
        #             if polygon_loc.area > (self.plot_width*self.plot_height)-self.delta_area and polygon_loc.area < (self.plot_width*self.plot_height)+self.delta_area:
        #                 cent_added.append([polygon_loc.centroid.x, polygon_loc.centroid.y])

        # cent_filled = self.cent_local + cent_added
        # polygon_filled_loc = {'geometry':[]}
        # for x_loc,y_loc in cent_filled:
                        
        #     ul = [x_loc - self.plot_width/2, y_loc + self.plot_height/2] 
        #     ur = [x_loc + self.plot_width/2, y_loc + self.plot_height/2]
        #     br = [x_loc + self.plot_width/2, y_loc - self.plot_height/2]
        #     bl = [x_loc - self.plot_width/2, y_loc - self.plot_height/2]
            
        #     bbox_loc = np.array([ul,ur,br,bl,ul])
        #     poly = sPolygon(bbox_loc)
        #     if self.geometry.contains(poly):
        #        polygon_filled_loc['geometry'].append(poly)

        # gdf_filled = gpd.GeoDataFrame(polygon_filled_loc, geometry='geometry')
        
        gdf_filled = gpd.GeoDataFrame(pd.concat([self.gdf_initial, gdf_added], ignore_index=True))
        self.gdf_final = gdf_filled
        
        return gdf_filled

    
    def grid_remove(self, gdf):
        
        cc_coverage=[]
        drop_ind=[]
        dst_crs = f'EPSG:{self.epsg}'
        
        with rasterio.open(self.data_product_url) as src:
            
            if src.crs.to_string() != dst_crs:
                for i,geom in enumerate(gdf.geometry):
                    # Convert the Shapely geometry to a GeoJSON-like dict
                    geom_geojson = mapping(geom)
                    # Reproject the geometry to the raster's CRS
                    reprojected_geom = transform_geom(dst_crs, src.crs.to_string(), geom_geojson)
                    try:
                        clip, _ = rasterio.mask.mask(src, [reprojected_geom], crop=True)
                    except:
                        drop_ind += [i]
                        continue
                    arr = clip.transpose([1,2,0])
                    canopeo = self.canopeo(arr)
                    cc_coverage.append(np.sum(canopeo) / canopeo.size)
            else:
                for i,geom in enumerate(gdf.geometry):
                    try:
                        clip, _ = rasterio.mask.mask(src, [geom], crop=True)
                    except:
                        drop_ind += [i]
                        continue
                    arr = clip.transpose([1,2,0])
                    canopeo = self.canopeo(arr)
                    cc_coverage.append(np.sum(canopeo) / canopeo.size)
        
        self.cc_coverage = cc_coverage
        
        all_boxes = gdf.geometry.apply(lambda geom: list(geom.bounds)).tolist()
        boxes_tensor = torch.tensor(all_boxes).double()
        cc_tensor = torch.tensor(cc_coverage).double()

        keep_indices = nms(boxes_tensor, cc_tensor, self.iou_threshold)
        drop_ind += [i for i in range(len(gdf)) if i not in keep_indices]

        kdtree = KDTree(np.array(self.cent_local))
        dist, ind = kdtree.query(np.array(self.cent_local), k=3)

        for i in range(len(self.cent_local)):
            if i not in drop_ind:
                if (dist[i,1:]<self.dist_thr).any():
                    redundant_ind = ind[i,:][np.where(dist[i,:]<self.dist_thr)]
                    # print(redundant_ind)
                    best_cc_ind = np.argmax([cc_coverage[i] for i in redundant_ind])
                    # print(best_cc_ind)
                    drop_ind += [val for i,val in enumerate(redundant_ind) if i!=best_cc_ind]
                    
        kdtree2 = KDTree(self.grid_coords.T)
        for i,(x_loc,y_loc) in enumerate(self.cent_local):
            dist2, ind2 = kdtree2.query(np.array([x_loc,y_loc]).reshape(1,-1), k=1)
            if dist2>self.dist_thr2:
                drop_ind.append(i)
                
        cc_cov_cut_arr = np.where(np.array(cc_coverage)<self.cc_coverage_thr)[0]
        drop_ind = np.unique(list(drop_ind) + list(cc_cov_cut_arr))

        gdf.drop(drop_ind, inplace=True)
        self.gdf_final = gdf
        
        return gdf
    
    
    def assign_row_col(self, gdf):
        
        polygon_row_col = {'id':[],
                           'geometry':[], 
                           'row':[], 
                           'col':[]}
        
        kdtree = KDTree(self.grid_coords.T)
        for i,geom in enumerate(gdf.geometry):
            x = geom.centroid.x
            y = geom.centroid.y
            dist, ind = kdtree.query(np.array([x,y]).reshape(1,-1), k=1)
            if dist < self.dist_thr:
                index = ind[0][0] + 1
                polygon_row_col['id'].append(i)
                polygon_row_col['geometry'].append(geom)
                polygon_row_col['row'].append(self.n_rows - (index // self.n_cols))
                polygon_row_col['col'].append(index % self.n_cols)
        
        gdf_row_col = gpd.GeoDataFrame(polygon_row_col, geometry='geometry')
        # gdf_row_col['geometry'] = gdf_row_col['geometry'].apply(lambda geom: rotate(geom, self.slope_deg, origin=self.center_geo, use_radians=False))
        self.gdf_final = gdf_row_col
        
        return gdf_row_col
        
    def delete(self, id):
        
        self.gdf_final.drop(index=id, inplace = True)
        
        return self.gdf_final
    
    
    def add(self, polygon):
        
        self.load_sam(self.sam_checkpoint, type='manual')
        img_rotated_resize = np.array(self.img_rotated.resize((self.resize[1],self.resize[0])))
        self.predictor.set_image(np.array(img_rotated_resize))
        
        gdf_polygon = gpd.GeoDataFrame.from_features([polygon]).set_crs("EPSG:4326")
        gdf_polygon.to_crs(f"EPSG:{self.epsg}", inplace=True)
        
        x = np.array(gdf_polygon.geometry[0].exterior.coords.xy[0])
        y = np.array(gdf_polygon.geometry[0].exterior.coords.xy[1])        
        coords = (np.array([x,y]).T - np.array([self.ext_left,self.ext_down])) / self.x_spacing
        
        # Define the center point for rotation
        center_x = self.img_width / 2
        center_y = self.img_height / 2

        # Rotate the coordinates
        rotated_coords = [rotate(Point(coord), self.slope_deg, origin=(center_x, center_y)) for coord in coords]
        rotated_coords = np.array([[point.x, point.y] for point in rotated_coords])
        
        # Scale the coordinates correctly
        rotated_coords[:, 0] *= self.resize[1] / self.img_width
        rotated_coords[:, 1] *= self.resize[0] / self.img_height
        rotated_coords[:, 1] = self.resize[0] - rotated_coords[:, 1]
        
        xmin = int(min(rotated_coords[:,0]))
        xmax = int(max(rotated_coords[:,0]))
        ymin = int(min(rotated_coords[:,1]))
        ymax = int(max(rotated_coords[:,1]))
        
        mask, _, _ = self.predictor.predict(
            point_coords = None,
            point_labels = None,
            box=np.array([xmin,ymin,xmax,ymax]),
            multimask_output=False,
        )
        self.added_mask = mask[0,:,:]
        
        gdf_prev = self.gdf_final
        gdf_prev['geometry'] = gdf_prev['geometry'].apply(lambda geom: rotate(geom, self.slope_deg, origin=self.center_geo, use_radians=False))
        
        gdf_added = self.mask2polygon([mask[0,:,:]])
        gdf_concat = gpd.GeoDataFrame(pd.concat([gdf_prev, gdf_added], ignore_index=True))
        
        gdf_assigned = self.assign_row_col(gdf_concat)
        gdf_assigned['geometry'] = gdf_assigned['geometry'].apply(lambda geom: rotate(geom, -self.slope_deg, origin=self.center_geo, use_radians=False))
        gdf_assigned.set_crs(f'EPSG:{self.epsg}', inplace=True)
        self.gdf_final = gdf_assigned
        
        # polygon = polygonize(mask[0,:,:], convex=True, simplify_tolerance=1.0)[0]
        
        # # Rotate the polygon coordinates back to the original
        # polygon_coords = np.array(polygon.exterior.coords) 
        # polygon_coords[:, 1] = self.resize[0] - polygon_coords[:, 1]
        # # resize
        # xy = np.array(polygon_coords) * np.array([self.scale_col, self.scale_row])
        # polygon_loc = sPolygon(xy)   
        # # box
        # cent_local = [polygon_loc.centroid.x, polygon_loc.centroid.y]
        # x_loc, y_loc = cent_local
        # ul = [x_loc - (self.plot_width/2)/self.x_spacing, y_loc + (self.plot_height/2)/-self.y_spacing]
        # ur = [x_loc + (self.plot_width/2)/self.x_spacing, y_loc + (self.plot_height/2)/-self.y_spacing]
        # br = [x_loc + (self.plot_width/2)/self.x_spacing, y_loc - (self.plot_height/2)/-self.y_spacing]
        # bl = [x_loc - (self.plot_width/2)/self.x_spacing, y_loc - (self.plot_height/2)/-self.y_spacing]
        # bbox_loc = np.array([ul, ur, br, bl, ul])    
        
        
        
        # # rotate
        # original_coords = [rotate(Point(coord), -self.slope_deg, origin=(center_x, center_y)) for coord in bbox_loc]
        # original_coords = np.array([[point.x, point.y] for point in original_coords])
        # # geo
        # transformed_coords = (np.array([x,y])*self.x_spacing + np.array([[self.ext_left],[self.ext_up]])).transpose()
        # bbox_polygon = sPolygon(transformed_coords)

        # polygon_row_col = {'id':[],
        #                    'geometry':[], 
        #                    'row':[], 
        #                    'col':[]}
        
        # gdf = self.gdf_final
        # polygon_row_col['id'].append(gdf.index[-1] + 1)
        # polygon_row_col['geometry'].append(bbox_polygon)
        # polygon_row_col['row'].append(self.n_rows - (gdf.index[-1] + 1) // self.n_cols)
        # polygon_row_col['col'].append((gdf.index[-1] + 1) % self.n_cols)    
        
        # # Create a GeoDataFrame for the bounding boxes
        # bbox_gdf = gpd.GeoDataFrame(polygon_row_col, geometry='geometry')

        # # Concatenate the bounding boxes GeoDataFrame with the existing gdf
        # gdf = gpd.GeoDataFrame(pd.concat([gdf, bbox_gdf], ignore_index=True))
        # gdf.set_crs(f'EPSG:{self.epsg}', inplace=True)
        # self.gdf_final = gdf
        
        return gdf_assigned
    
    
    def to_geojson(self, gdf, rotation=False):
        
        if rotation:
            gdf['geometry'] = gdf['geometry'].apply(lambda geom: rotate(geom, -self.slope_deg, origin=self.center_geo, use_radians=False))
        
        gdf_4326 = gdf.to_crs('EPSG:4326')
        gdf_geojson = gdf_4326.to_json()
        
        return gdf_geojson
    
    
    def evaluation(self, gt_filename, result_filename=None):
        
        # Read your GeoJSON files (adjust the file names as needed)
        gt_gdf = gpd.read_file(gt_filename)
        
        if result_filename is not None:
            result_gdf = gpd.read_file(result_filename)
        else:
            result_gdf = self.gdf_final
            
        if gt_gdf.crs is not None:
            gt_gdf.to_crs(result_gdf.crs.to_string(), inplace=True)
        else:
            gt_gdf.set_crs("EPSG:4326", inplace=True)
            gt_gdf.to_crs(result_gdf.crs.to_string(), inplace=True)
        

        ### Pixel-based Evaluation ###
        # Compute union bounds from both GeoDataFrames
        minx = min(gt_gdf.total_bounds[0], result_gdf.total_bounds[0])
        miny = min(gt_gdf.total_bounds[1], result_gdf.total_bounds[1])
        maxx = max(gt_gdf.total_bounds[2], result_gdf.total_bounds[2])
        maxy = max(gt_gdf.total_bounds[3], result_gdf.total_bounds[3])

        # Define a resolution (units per pixel; adjust as needed)
        resolution = self.geotransform[1]
        width = int((maxx - minx) / resolution)
        height = int((maxy - miny) / resolution)

        # Define an affine transform (note: y-axis is reversed)
        transform = Affine(resolution, 0, minx, 0, -resolution, maxy)

        # Rasterize the reference polygons into a binary mask
        ref_mask = rasterize(
            [(geom, 1) for geom in gt_gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Rasterize the result polygons into a binary mask
        res_mask = rasterize(
            [(geom, 1) for geom in result_gdf.geometry],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )

        # Compute pixel-level Intersection over Union (IoU)
        intersection = np.logical_and(ref_mask, res_mask).sum()
        union = np.logical_or(ref_mask, res_mask).sum()
        iou_pixel = intersection / union if union > 0 else 0

        # Compute pixel-based precision, recall, and F1 score
        # Here, True Positives (TP): pixels where both masks equal 1
        TP = np.logical_and(ref_mask == 1, res_mask == 1).sum()
        FP = np.logical_and(ref_mask == 0, res_mask == 1).sum()
        FN = np.logical_and(ref_mask == 1, res_mask == 0).sum()

        precision_pixel = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall_pixel = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_pixel = 2 * (precision_pixel * recall_pixel) / (precision_pixel + recall_pixel) if (precision_pixel + recall_pixel) > 0 else 0

        print("Pixel-based Metrics:")
        print("IoU:", f"{iou_pixel*100:.2f}")
        print("Precision:", f"{precision_pixel*100:.2f}")
        print("Recall:", f"{recall_pixel*100:.2f}")
        print("F1 Score:", f"{f1_pixel*100:.2f}")

        ### Polygon-based Evaluation ###
        # Helper function to compute IoU for two polygons
        def polygon_iou(poly1, poly2):
            if not poly1.intersects(poly2):
                return 0
            inter_area = poly1.intersection(poly2).area
            union_area = poly1.union(poly2).area
            return inter_area / union_area if union_area > 0 else 0

        # One-to-one matching: for each result polygon, find an unmatched reference polygon with IoU >= 0.8
        matched_refs = set()
        TP_count = 0

        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.90, 0.95]
        precisions =[]
        recalls = []
        f1s = []
        for iou_threshold in iou_thresholds:
            TP_count = 0
            matched_refs = set()
            
            # Loop over result polygons
            for res_poly in result_gdf.geometry:
                best_iou = 0
                best_ref_idx = None
                # Try to match with each reference polygon that hasn't been matched yet
                for idx, ref_poly in gt_gdf.geometry.items():
                    if idx in matched_refs:
                        continue
                    iou_val = polygon_iou(res_poly, ref_poly)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_ref_idx = idx
                if best_iou >= iou_threshold:
                    TP_count += 1
                    matched_refs.add(best_ref_idx)

            FP_count = len(result_gdf) - TP_count
            FN_count = len(gt_gdf) - len(matched_refs)

            precision = TP_count / (TP_count + FP_count) if (TP_count + FP_count) > 0 else 0
            recall = TP_count / (TP_count + FN_count) if (TP_count + FN_count) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        print("\nPolygon-based Metrics (@50):")
        print("Precision:", f"{precisions[0]*100:.2f}")
        print("Recall:", f"{recalls[0]*100:.2f}")
        print("F1 Score:", f"{f1s[0]*100:.2f}")
        
        print("\nPolygon-based Metrics (@50-95):")
        print("Precision:", f"{np.mean(precisions)*100:.2f}")
        print("Recall:", f"{np.mean(recalls)*100:.2f}")
        print("F1 Score:", f"{np.mean(f1s)*100:.2f}")

    
        
        
    def automatic_detection(self, save=False):
        
        # load image
        self.load_image()
        print(f"Loaded image: {self.filename}")
        
        # rotate if needed
        self.load_sam(self.sam_checkpoint, points_per_side=32)
        self.get_masks()
        self.rotate_plot()
        
        # initial plots
        self.clear_sam()
        self.load_sam(self.sam_checkpoint, points_per_side=self.points_per_side)   
        self.initial_plots()
        print(f"Initial plots: {len(self.gdf_initial['geometry'])}")
        
        # refining plots
        self.load_sam(self.sam_checkpoint, type='manual')
        gdf_filled = self.grid_filling()
        gdf_removed = self.grid_remove(gdf_filled)
        print(f"Refined plots: {len(gdf_removed)}")
        
        # assign rows and columns id
        gdf_final = self.assign_row_col(gdf_removed)
        print(f"Assigned rows and columns")
        
        gdf_final.set_crs(f'EPSG:{self.epsg}', inplace=True)
        self.gdf_final = gdf_final
        gdf_geojson = self.to_geojson(gdf_final, rotation=True)
        if save:
            gdf_final.to_crs('EPSG:4326').to_file(self.out_filename, driver='GeoJSON')
        print(f"Process completed")
        
        self.final_geojson = gdf_geojson
        self.clear_sam()
        
        return gdf_geojson