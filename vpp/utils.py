import numpy as np
import cv2

def transform_inv(T):
    T_inv = np.eye(4)
    T_inv[:3,:3] = T[:3,:3].T
    T_inv[:3,3] = -1.0 * ( T_inv[:3,:3] @ T[:3,3] )
    return T_inv

def compute_camera_parameters(width, height, HFoV):
    """
    Computes the focal length (f), center point coordinates (c_x, c_y) based on
    the image width, height, and horizontal field of view (HFoV).

    Parameters:
    - width: The width of the image (in pixels).
    - height: The height of the image (in pixels).
    - HFoV: Horizontal field of view (in radians).

    Returns:
    - A 3x3 numpy array representing the camera matrix.
    """
    # Focal length calculation
    f = (0.5 * width) / np.tan(0.5 * HFoV)
    
    # Image center coordinates
    c_x = (width - 1) * 0.5
    c_y = (height - 1) * 0.5
    
    return np.array([[f,0,c_x],[0,f,c_y],[0,0,1]]).astype(np.float32)

def get_view(pcd:np.ndarray, rotomtx, K, D, height, width):

    _pcd = pcd[:,:3]
    _pcd = np.concatenate([_pcd, np.ones((_pcd.shape[0],1))], axis=1)
    rotated_cloud = (rotomtx @ _pcd.T).T
    
    if len(rotated_cloud) > 0:
        
        camera_points = rotated_cloud[:,:3]
        colors_points = np.concatenate([pcd[:,3:4] for _ in range(3)], axis=1)

        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        imgpts, _ = cv2.projectPoints(camera_points, rvecs, tvecs, K, D)

        imgpts = imgpts[:,0,:]
        valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < height) & \
                    (imgpts[:, 0] >= 0) & (imgpts[:, 0] < width)
        imgpts = imgpts[valid_points,:]

        depth = camera_points[valid_points,2]
        colors = colors_points[valid_points,0]

        depth_image = np.zeros([height, width]) + np.inf
        depth_image[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = depth

        depth_image[ depth_image==0.0 ] = np.inf
        depth_image[ np.isinf(depth_image) ] = 0.0

        color_image = np.zeros([height, width]) + np.inf
        color_image[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = colors

        color_image[ color_image==0.0 ] = np.inf
        color_image[ np.isinf(color_image) ] = 0.0

        return depth_image, color_image

    return np.zeros([height, width]), np.zeros([height, width])

def reproject_depth(W_start, H_start, K, depth, RT, K_end, W_end, H_end):
    xx, yy = np.meshgrid(np.arange(W_start), np.arange(H_start))
    points_grid = np.stack(((xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones_like(xx)), axis=0) * depth
    mask = np.ones((H_start, W_start), dtype=bool)
    mask[depth<=0] = False
    depth_pts = points_grid.transpose(1,2,0)[mask]

    camera_points = (RT @ np.vstack([depth_pts.T, np.ones(depth_pts.shape[0])])).T 

    if camera_points.shape[0] == 0:
        return np.zeros([H_end, W_end])

    rvecs = np.zeros((3,1)) # cv2.Rodrigues(np.eye(3))[0]
    tvecs = np.zeros((3,1))
    D_end = np.zeros((4,1))

    _camera_points = camera_points[:,:3]

    imgpts, _ = cv2.projectPoints(_camera_points, rvecs, tvecs, K_end, D_end)
    
    imgpts = imgpts[:,0,:]
    valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < H_end) & \
                (imgpts[:, 0] >= 0) & (imgpts[:, 0] < W_end)
    imgpts = imgpts[valid_points,:]

    _end_depth = camera_points[valid_points,2]

    end_depth = np.zeros([H_end, W_end])
    end_depth[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = _end_depth

    return end_depth


def sample_hints(hints, probability=0.20):
    validhints = hints>0
    new_validhints = (validhints * (np.random.random_sample(validhints.shape) < probability))
    new_hints = hints * new_validhints  
    new_hints[new_validhints==0] = 0
    return new_hints

def reproject_depth_cache(W_start, H_start, K, depth, RT, K_end, W_end, H_end, points_grid=None):
    if points_grid is None:
        xx, yy = np.meshgrid(np.arange(W_start), np.arange(H_start))
        points_grid = np.stack(((xx-K[0,2])/K[0,0], (yy-K[1,2])/K[1,1], np.ones_like(xx)), axis=0)
    depth_points_grid = points_grid * depth

    mask = np.ones((H_start, W_start), dtype=bool)
    mask[depth<=0] = False
    depth_pts = depth_points_grid.transpose(1,2,0)[mask]

    camera_points = (RT @ np.vstack([depth_pts.T, np.ones(depth_pts.shape[0])])).T 

    if camera_points.shape[0] == 0:
        return np.zeros([H_end, W_end]), points_grid

    rvecs = np.zeros((3,1)) # cv2.Rodrigues(np.eye(3))[0]
    tvecs = np.zeros((3,1))
    D_end = np.zeros((4,1))

    _camera_points = camera_points[:,:3]

    imgpts, _ = cv2.projectPoints(_camera_points, rvecs, tvecs, K_end, D_end)
    
    imgpts = imgpts[:,0,:]
    valid_points = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < H_end) & \
                (imgpts[:, 0] >= 0) & (imgpts[:, 0] < W_end)
    imgpts = imgpts[valid_points,:]

    _end_depth = camera_points[valid_points,2]

    end_depth = np.zeros([H_end, W_end])
    end_depth[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = _end_depth

    return end_depth, points_grid


def sample_hints_cache(hints, probability=0.20, sample_img = None):
    validhints = hints>0
    if sample_img is None:
        sample_img = np.random.random_sample(validhints.shape)

    new_validhints = (validhints * (sample_img < probability))
    new_hints = hints * new_validhints  
    new_hints[new_validhints==0] = 0
    return new_hints, sample_img

def add_title_description(image, title, description, title_font=cv2.FONT_HERSHEY_SIMPLEX, desc_font=cv2.FONT_HERSHEY_SIMPLEX, 
                          title_font_scale=1, desc_font_scale=0.5, title_color=(255, 255, 255), desc_color=(255, 255, 255), 
                          title_thickness=2, desc_thickness=1):
    # Get the image dimensions
    height, width = image.shape[:2]
    
    # Calculate the position for the title (center top)
    title_size = cv2.getTextSize(title, title_font, title_font_scale, title_thickness)[0]
    title_x = (width - title_size[0]) // 2
    title_y = title_size[1] + 10  # 10 pixels from the top
    
    # Calculate the position for the description (center bottom)
    desc_size = cv2.getTextSize(description, desc_font, desc_font_scale, desc_thickness)[0]
    desc_x = (width - desc_size[0]) // 2
    desc_y = height - 10  # 10 pixels from the bottom
    
    # Add the title to the image
    cv2.putText(image, title, (title_x, title_y), title_font, title_font_scale, title_color, title_thickness, cv2.LINE_AA)
    
    # Add the description to the image
    cv2.putText(image, description, (desc_x, desc_y), desc_font, desc_font_scale, desc_color, desc_thickness, cv2.LINE_AA)
    
    return image


def guided_metrics(disp, gt, valid):
    error = np.abs(disp-gt)
    error[valid==0] = 0
    
    bad1 = (error[valid>0] > 1.).astype(np.float32).mean()
    bad2 = (error[valid>0] > 2.).astype(np.float32).mean()
    bad3 = (error[valid>0] > 3.).astype(np.float32).mean()
    bad4 = (error[valid>0] > 4.).astype(np.float32).mean()
    avgerr = error[valid>0].mean()
    rms = (disp-gt)**2
    rms = np.sqrt( rms[valid>0].mean() )
    return {'bad 1.0':bad1, 'bad 2.0':bad2, 'bad 3.0': bad3, 'bad 4.0':bad4, 'avgerr':avgerr, 'rms':rms, 'errormap':error*(valid>0)}


# a function to resize an image to a defined H W size
def resize_image(image, H, W):
    if isinstance(image, list):
        return [cv2.resize(img.copy(), (W, H), interpolation=cv2.INTER_LINEAR) for img in image]
    
    return cv2.resize(image.copy(), (W, H), interpolation=cv2.INTER_LINEAR)


