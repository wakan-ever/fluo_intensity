import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

def segment_follicle(image, intensity_percentile=96.3, size_threshold_fraction=0.25):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Normalize image
    image_norm = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # Apply thresholding
    threshold_value = np.percentile(image_norm, intensity_percentile)
    _, mask = cv2.threshold(image_norm, threshold_value, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    
    # Identify connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    # Find the largest connected component
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        largest_label = None
    
    # Define size threshold for filtering small objects
    size_threshold = stats[largest_label, cv2.CC_STAT_AREA] * size_threshold_fraction if largest_label else 0
    
    # Create final refined segmentation mask
    final_follicle_mask = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > size_threshold:
            final_follicle_mask[labels == i] = 255
    
    # Compute intensity metrics
    mean_intensity = np.mean(image_gray[final_follicle_mask > 0])
    total_intensity = np.sum(image_gray[final_follicle_mask > 0])
    
    return final_follicle_mask, mean_intensity, total_intensity

def main():
    st.title("Fluorescence Intensity Measurement Tool")
    st.write("Upload an image, adjust parameters, and visualize the follicle segmentation.")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "tif", "tiff"])
    intensity_percentile = st.slider("Intensity Percentile", 90.0, 99.9, 96.3, 0.1)
    size_threshold_fraction = st.slider("Size Threshold Fraction", 0.0, 1.0, 0.25, 0.05)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image
        follicle_mask, mean_intensity, total_intensity = segment_follicle(image, intensity_percentile, size_threshold_fraction)
        
        # Display segmentation result
        fig, ax = plt.subplots()
        ax.imshow(follicle_mask, cmap='gray')
        ax.set_title("Segmented Follicle Mask")
        ax.axis("off")
        st.pyplot(fig)
        
        # Display intensity metrics
        st.write(f"**Mean Intensity:** {mean_intensity:.2f}")
        st.write(f"**Total Intensity:** {total_intensity:.2f}")
    
    # Explanation of the app functionality
    st.header("How This App Works")
    st.write("This tool is designed to measure fluorescent intensity from images of stained follicles.")
    st.write("**1. Image Upload:** Upload an image in JPG, PNG, or TIFF format.")
    st.write("**2. Adjust Parameters:** Users can fine-tune two key parameters:")
    
    st.subheader("Intensity Percentile")
    st.write("**Definition:** This parameter determines the intensity threshold used for segmentation. Pixels with intensity values above this percentile are considered as part of the follicle, while others are ignored.")
    st.write("**How it works:**")
    st.write("- The grayscale image is normalized to ensure intensity values range between 0 and 255.")
    st.write("- The percentile threshold is computed based on the image histogram.")
    st.write("- Pixels above this threshold are set to white (255), and others are set to black (0), forming a binary mask.")
    st.write("**Impact of different values:")
    st.write("- Lower values (e.g., 90%): More pixels are included in the follicle mask, potentially leading to more noise.")
    st.write("- Higher values (e.g., 99%): Only the brightest regions are included, making the segmentation more selective but possibly missing follicle details.")
    
    st.subheader("Size Threshold Fraction")
    st.write("**Definition:** This parameter is used to filter out small objects (e.g., artifacts or dye spots) by comparing their sizes to the largest detected follicle.")
    st.write("**How it works:**")
    st.write("- The largest connected component in the binary mask is identified as the follicle.")
    st.write("- The size threshold is computed as: size_threshold = largest follicle area × size_threshold_fraction")
    st.write("- Any connected components (blobs) smaller than this threshold are removed.")
    st.write("**Impact of different values:**")
    st.write("- Lower values (e.g., 0.05 or 5%): Smaller structures are retained, which might include noise.")
    st.write("- Higher values (e.g., 0.5 or 50%): Only large structures are preserved, potentially eliminating useful parts of the follicle.")
    
    st.write("**3. Segmentation & Visualization:** The app processes the image to detect follicles and display the segmented mask.")
    st.write("**4. Intensity Measurement:** The tool calculates and reports the mean and total fluorescent intensity within the segmented follicle region.")
    st.write("Hope this helps for understanding how this app works on analyzing follicle fluorescence in biological imaging applications.")

if __name__ == "__main__":
    main()
