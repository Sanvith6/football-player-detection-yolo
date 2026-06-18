import os
import pillow_avif
from PIL import Image
from ultralytics import YOLO

def main():
    # Paths
    model_path = r"C:\Users\sanvi\runs\detect\football_player_detection\yolo26m_gpu_training-3\weights\best.pt"
    input_image = r"C:\project\YOLO26\save.jpeg"
    
    # Generate output name dynamically from input image name
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    output_image_name = f"{base_name}_prediction.jpg"
    
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}")
        return

    if not os.path.exists(input_image):
        print(f"Error: Input image not found at {input_image}")
        return

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)

    # Open the image using PIL (which loads it correctly with pillow_avif)
    print(f"Opening image: {input_image}")
    img = Image.open(input_image)

    # Perform prediction
    print("Running prediction...")
    results = model.predict(source=img, save=False, verbose=False)
    result = results[0]  # Get results for the single image

    # Process detections
    print("\n--- Detection Results ---")
    if len(result.boxes) == 0:
        print("No objects detected.")
    else:
        print(f"Found {len(result.boxes)} object(s):")
        # Class mapping dictionary
        class_names = result.names
        
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].item())
            class_name = class_names.get(class_id, f"Unknown ({class_id})")
            confidence = box.conf[0].item()
            xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
            
            print(f"[{i+1}] {class_name.upper()} | Confidence: {confidence:.2f} | Bounding Box: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")

    # Save output image to workspace
    print(f"\nSaving annotated image to {output_image_name}...")
    result.save(filename=output_image_name)
    
    # Display the image with detected objects
    print("Displaying image...")
    result.show()
    
    print("Done!")

if __name__ == "__main__":
    main()
