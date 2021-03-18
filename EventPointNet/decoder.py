import EventPointNet.nets as nets
import numpy as np

class CDecoder():
    def __init__(self, iWidth, iHeight, iCell = 8, fThreshold = 0.015, nms_dist = 4):
        self.__iCell = iCell
        self.__iWidth = iWidth
        self.__iHeight = iHeight
        self.__iWidthC = self.__iWidth / self.__iCell
        self.__iHeightC = self.__iHeight / self.__iCell
        self.__fThreshold = fThreshold
        self.__nms_dist = nms_dist

    def KeypointDecoder(self, tKeyPoint):
        tDense = torch.nn.functional.softmax(tKeyPoint, dim=0)
        npDense = tDense.data.cpu().numpy().squeeze()
        npRmDustbin = npDense[:-1, :, :]
        npRmDustbin = npRmDustbin.transpose(1, 2, 0)
        npHeatmap = np.reshape(npRmDustbin, [self.__iHeightC, self.__iWidthC, self.__iCell, self.__iCell])
        npHeatmap = np.transpose(npHeatmap, [0, 2, 1, 3])
        npHeatmap = np.reshape(npHeatmap, [self.__iHeight, self.iWidth])
        iXs, iYs, = np.where(npHeatmap >= self.__fThreshold)
        oPoints = np.zeros((3, len(iXs)))
        oPoints[0, :] = iXs
        oPoints[1, :] = iYs
        oPoints[2, :] = npHeatmap[iXs, iYs]
        oPoints, _ = self.__nms_fast(oPoints, self.__iHeight, self.__iWidth, dist_thresh=self.__nms_dist)
        ptSorted = np.argsort(oPoints[2, :])
        oPoints = oPoints[:, ptSorted[::-1]]
        return oPoints

    def DescriptorDecoder(self):
        print("Desc decoder")

    def __nms_fast(self, in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds
