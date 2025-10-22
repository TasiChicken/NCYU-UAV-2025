import cv2
import numpy as np

# Global variables for mouse callback
selected_points = []
temp_image = None
original_image = None

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to capture clicked points"""
    global selected_points, temp_image, original_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(selected_points) < 4:
            selected_points.append([x, y])
            print(f"Point {len(selected_points)}: ({x}, {y})")
            
            # Draw the point on the image
            temp_image = original_image.copy()
            
            # Draw all selected points
            for i, point in enumerate(selected_points):
                cv2.circle(temp_image, tuple(point), 5, (0, 255, 0), -1)
                cv2.putText(temp_image, f"{i+1}", (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw lines connecting points if we have more than 1
            if len(selected_points) > 1:
                for i in range(len(selected_points) - 1):
                    cv2.line(temp_image, tuple(selected_points[i]), 
                            tuple(selected_points[i+1]), (255, 0, 0), 2)
            
            # Connect last point to first if we have all 4 points
            if len(selected_points) == 4:
                cv2.line(temp_image, tuple(selected_points[3]), 
                        tuple(selected_points[0]), (255, 0, 0), 2)
                cv2.putText(temp_image, "Press SPACE to confirm or 'r' to reset", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("Select Billboard Corners", temp_image)

def select_billboard_corners(image):
    """Interactive function to select billboard corners by clicking"""
    global selected_points, temp_image, original_image
    
    selected_points = []
    original_image = image.copy()
    temp_image = original_image.copy()
    
    # Resize image for better display if too large
    display_image = temp_image.copy()
    scale_factor = 1.0
    
    if image.shape[1] > 1200 or image.shape[0] > 800:
        scale_factor = min(1200/image.shape[1], 800/image.shape[0])
        new_width = int(image.shape[1] * scale_factor)
        new_height = int(image.shape[0] * scale_factor)
        display_image = cv2.resize(display_image, (new_width, new_height))
        temp_image = display_image.copy()
        original_image = display_image.copy()
    
    cv2.namedWindow("Select Billboard Corners", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Select Billboard Corners", mouse_callback)
    
    # Add instructions
    cv2.putText(temp_image, "Click 4 corners of billboard screen in order:", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(temp_image, "1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(temp_image, "Press 'r' to reset, SPACE to confirm, ESC to cancel", (10, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Select Billboard Corners", temp_image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC key
            print("❌ Selection cancelled")
            cv2.destroyWindow("Select Billboard Corners")
            return None
            
        elif key == ord('r'):  # Reset
            print("🔄 Resetting selection...")
            selected_points = []
            temp_image = original_image.copy()
            cv2.putText(temp_image, "Click 4 corners of billboard screen in order:", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(temp_image, "1. Top-left  2. Top-right  3. Bottom-right  4. Bottom-left", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(temp_image, "Press 'r' to reset, SPACE to confirm, ESC to cancel", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Select Billboard Corners", temp_image)
            
        elif key == 32 and len(selected_points) == 4:  # SPACE key
            print("✅ Selection confirmed!")
            cv2.destroyWindow("Select Billboard Corners")
            
            # Scale points back to original image size if we resized
            if scale_factor != 1.0:
                scaled_points = []
                for point in selected_points:
                    scaled_x = int(point[0] / scale_factor)
                    scaled_y = int(point[1] / scale_factor)
                    scaled_points.append([scaled_x, scaled_y])
                return np.array(scaled_points, dtype=np.int32)
            else:
                return np.array(selected_points, dtype=np.int32)
    
    return None

def bilinear_interpolate(img, x, y):
    """
    Bilinear interpolation for (x, y) in source image.
    """
    h, w, c = img.shape

    if x < 0 or x >= w-1 or y < 0 or y >= h-1:
        return np.array([0, 0, 0], dtype=np.uint8)

    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = x0 + 1, y0 + 1

    dx, dy = x - x0, y - y0

    # Four neighbors
    I00 = img[y0, x0].astype(np.float32)
    I10 = img[y0, x1].astype(np.float32)
    I01 = img[y1, x0].astype(np.float32)
    I11 = img[y1, x1].astype(np.float32)

    # Interpolation
    top = (1 - dx) * I00 + dx * I10
    bottom = (1 - dx) * I01 + dx * I11
    value = (1 - dy) * top + dy * bottom

    return np.clip(value, 0, 255).astype(np.uint8)

def warp_perspective(img, M, dsize):
    """
    Manual perspective warp with bilinear interpolation.
    """
    h, w = dsize
    warped = np.zeros((h, w, 3), dtype=np.uint8)

    Minv = np.linalg.inv(M)

    for y in range(h):
        for x in range(w):
            dst = np.array([x, y, 1])
            src = Minv @ dst
            src /= src[2]

            sx, sy = src[0], src[1]
            if 0 <= sx < img.shape[1]-1 and 0 <= sy < img.shape[0]-1:
                warped[y, x] = bilinear_interpolate(img, sx, sy)

    return warped



def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    # Load the billboard image
    billboard_image_path = input("Enter the path to your billboard image (or press Enter for default): ").strip()
    if not billboard_image_path:
        print("❌ Please provide the path to your billboard image")
        return
    
    try:
        billboard_scene = cv2.imread(billboard_image_path)
        if billboard_scene is None:
            print(f"❌ Could not load image from {billboard_image_path}")
            return
        print(f"✅ Loaded billboard image: {billboard_image_path}")
    except:
        print(f"❌ Error loading image from {billboard_image_path}")
        return
    
    scene_height, scene_width = billboard_scene.shape[:2]
    
    # Interactive coordinate selection
    print("🖱️  Interactive coordinate selection mode")
    print("📍 You will click on 4 corners of the billboard screen area")
    print("🔄 Click in order: Top-left → Top-right → Bottom-right → Bottom-left")
    
    billboard_corners = select_billboard_corners(billboard_scene)
    
    if billboard_corners is None:
        print("❌ No coordinates selected. Exiting...")
        cap.release()
        return
    
    print("✅ Billboard corners selected successfully!")
    print("Selected coordinates:")
    corner_names = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
    for i, (name, corner) in enumerate(zip(corner_names, billboard_corners)):
        print(f"  {i+1}. {name}: ({corner[0]}, {corner[1]})")
    
    # Define source points (entire camera frame) and destination points (billboard corners)
    ret, test_frame = cap.read()
    if not ret:
        print("❌ Cannot read from camera")
        return
    
    frame_height, frame_width = test_frame.shape[:2]
    src_pts = np.float32([
        [0, 0],                           # top-left
        [frame_width-1, 0],               # top-right
        [frame_width-1, frame_height-1],  # bottom-right
        [0, frame_height-1]               # bottom-left
    ])
    
    dst_pts = np.float32(billboard_corners)
    
    print("🎬 Starting billboard video projection...")
    print("📹 Camera feed will be warped onto your billboard image")
    print("🔧 Adjust billboard_corners in code if needed to match screen area precisely")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a copy of the original billboard scene
        display_scene = billboard_scene.copy()
        
        # Get perspective transform matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # Warp the camera frame onto billboard using OpenCV
        warped_frame = cv2.warpPerspective(frame, M, (scene_width, scene_height))
        
        # Create a mask for the billboard screen area
        billboard_mask = np.zeros((scene_height, scene_width), dtype=np.uint8)
        cv2.fillPoly(billboard_mask, [billboard_corners], 255)
        
        # Apply the warped camera feed only to the billboard screen area
        display_scene[billboard_mask > 0] = warped_frame[billboard_mask > 0]
        
        # Optional: Add a subtle border around the screen area for visibility
        cv2.polylines(display_scene, [billboard_corners], True, (0, 255, 0), 2)
        
        # Add status information
        cv2.putText(display_scene, "Live Camera Feed on Billboard", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_scene, "Press 'q' to quit", (20, scene_height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Resize for better display if image is too large
        if scene_width > 1200 or scene_height > 800:
            scale = min(1200/scene_width, 800/scene_height)
            new_width = int(scene_width * scale)
            new_height = int(scene_height * scale)
            display_scene = cv2.resize(display_scene, (new_width, new_height))

        cv2.imshow("Original Camera", frame)
        cv2.imshow("AR Billboard", display_scene)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Billboard AR projection ended")


if __name__ == "__main__":
    main()
