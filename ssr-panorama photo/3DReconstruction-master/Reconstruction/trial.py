import cv2
import numpy as np 
import glob
from tqdm import tqdm
import PIL.ExifTags
import PIL.Image
from matplotlib import pyplot as plt 
import imutils
import time

frameRate = (float(1)/float(10)) #//it will capture image in each 1/framerate second
cap_l = cv2.VideoCapture(0)
cap_l.set(cv2.CAP_PROP_FPS, frameRate**-1)
# # cap_r = cv2.VideoCapture(0)

i=0
def getFrames(sec,frameRate):

    # time.sleep(frameRate)
    path_img_l = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/database_l/frame_{}.jpg".format(sec)
    path_img_r = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/database_r/frame_{}.jpg".format(sec)
    hasFrames_l, frame_l = cap_l.read()
    hasFrames_r, frame_r = cap_l.read()
    global img_l
    global img_r
    img_l = path_img_l
    img_r = path_img_r

    if (hasFrames_l & hasFrames_r):
        cv2.imwrite(path_img_l, frame_l)     # save frame_l as JPG file
        cv2.imwrite(path_img_r, frame_r)     # save frame_r as JPG file
        cv2.imshow("img_l", frame_l)
        cv2.imshow("img_r", frame_r)
    return hasFrames_l & hasFrames_l



global sec
sec = 0
frameRate = (float(1)/float(10)) #//it will capture image in each 1/framerate second
success = getFrames(sec,frameRate)


while success:

    sec = sec + frameRate
    sec = round(sec, 2)
    
    if sec >= (60*frameRate):
        sec = 0

    success = getFrames(sec, frameRate)

    def create_output(vertices, colors, filename):
        colors = colors.reshape(-1,3)
        vertices = np.hstack([vertices.reshape(-1,3),colors])

        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header
            '''
        with open(filename, "w") as f:
            f.write(ply_header %dict(vert_num=len(vertices)))
            np.savetxt(f,vertices,'%f %f %f %d %d %d')

#Function that Downsamples image x number (reduce_factor) of times. 
    def downsample_image(image, reduce_factor):
        for i in range(0,reduce_factor):
            #Check if image is color or grayscale
            if len(image.shape) > 2:
                row,col = image.shape[:2]
            else:
                row,col = image.shape

            image = cv2.pyrDown(image, dstsize= (col//2, row // 2))
        return image


    #=========================================================
    # Stereo 3D reconstruction 
    #=========================================================

    #Load camera parameters
    ret = np.load("/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/camera_params/ret.npy")
    K = np.load("/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/camera_params/K.npy")
    dist = np.load("/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/camera_params/dist.npy")

    #Specify image paths
    # img_path1 = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/reconstruct_this/left.jpg"
    # img_path2 = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/reconstruct_this/right.jpg"

    #Load pictures
    print(img_l)
    print(img_r)
    img_1 = cv2.imread(img_l)
    img_2 = cv2.imread(img_r)

    #Get height and width. Note: It assumes that both pictures are the same size. They HAVE to be same size and height. 
    h,w = img_2.shape[:2]

    #Get optimal camera matrix for better undistortion 
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

    #Undistort images
    img_1_undistorted = cv2.undistort(img_1, K, dist, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(img_2, K, dist, None, new_camera_matrix)

    #Downsample each image 3 times (because they're too big)
    img_1_downsampled = downsample_image(img_1_undistorted,3)
    img_2_downsampled = downsample_image(img_2_undistorted,3)

    cv2.imwrite('undistorted_left.jpg', img_1_downsampled)
    cv2.imwrite('undistorted_right.jpg', img_2_downsampled)


    #Set disparity parameters
    #Note: disparity range is tuned according to specific parameters obtained through trial and error. 
    win_size = 5
    min_disp = -1
    max_disp = 63 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16

    #Create Block matching object. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
        numDisparities = num_disp,
        blockSize = 5,
        uniquenessRatio = 5,
        speckleWindowSize = 5,
        speckleRange = 5,
        disp12MaxDiff = 2,
        P1 = 8*3*win_size**2,#8*3*win_size**2,
        P2 =32*3*win_size**2) #32*3*win_size**2)

    #Compute disparity map
    print ("\nComputing the disparity  map...")
    disparity_map = stereo.compute(img_1_downsampled, img_2_downsampled)

    #Show disparity map before generating 3D cloud to verify that point cloud will be usable. 
    #_________________________
    #____________________________
    
    plt.imshow(disparity_map,"gray")
    # plt.show()
    plt_address = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/database_plt/gray_plt_{}.jpg".format(sec)
    plt.savefig(plt_address)
    img_plt = cv2.imread(plt_address)
    cv2.imshow("plt", img_plt)
    #Generate  point cloud. 
    print ("\nGenerating the 3D map...")

    #Get new downsampled width and height 
    h,w = img_2_downsampled.shape[:2]

    #Load focal length. 
    focal_length = np.load("/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/camera_params/FocalLength.npy")

    #Perspective transformation matrix
    #This transformation matrix is from the openCV documentation, didn't seem to work for me. 
    Q = np.float32([[1,0,0,-w/2.0],
                    [0,-1,0,h/2.0],
                    [0,0,0,-focal_length],
                    [0,0,1,0]])

    #This transformation matrix is derived from Prof. Didier Stricker's power point presentation on computer vision. 
    #Link : https://ags.cs.uni-kl.de/fileadmin/inf_ags/3dcv-ws14-15/3DCV_lec01_camera.pdf
    Q2 = np.float32([[1,0,0,0],
                    [0,-1,0,0],
                    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
                    [0,0,0,1]])

    #Reproject points into 3D
    points_3D = cv2.reprojectImageTo3D(disparity_map, Q2)
    #Get color points
    colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)

    #Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()

    #Mask colors and points. 
    output_points = points_3D[mask_map]
    output_colors = colors[mask_map]

    #Define name for output file
    output_file = 'reconstructed.ply'

    #Generate point cloud 
    print ("\n Creating the output file... \n")
    create_output(output_points, output_colors, output_file)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap_l.release()
cv2.destroyAllWindows()