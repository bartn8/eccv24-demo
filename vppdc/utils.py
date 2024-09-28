import cv2
import numpy as np

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
        colors = colors_points[valid_points,:]

        depth_image = np.zeros([height, width]) + np.inf
        depth_image[imgpts[:,1].astype(int), imgpts[:,0].astype(int)] = depth

        depth_image[ depth_image==0.0 ] = np.inf
        depth_image[ np.isinf(depth_image) ] = 0.0

        color_image = np.zeros([height, width, 3]) + np.inf
        color_image[imgpts[:,1].astype(int), imgpts[:,0].astype(int), :] = colors

        color_image[ color_image==0.0 ] = np.inf
        color_image[ np.isinf(color_image) ] = 0.0

        return depth_image, color_image

    return np.zeros([height, width]), np.zeros([height, width])

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

