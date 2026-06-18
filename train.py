import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO

def main():
    # Force GPU usage
    device = "0"
    print(f"Using device: {device}")
    
    # Path to the corrected data.yaml
    yaml_path = r"c:\project\YOLO26\Football-Player-Detection.v1i.yolo26\data.yaml"
    
    # Load YOLO model (yolo26m.pt is the new Medium YOLO26 model)
    print("Loading YOLO model...")
    model = YOLO("yolo26m.pt")
    
    # Run training
    print("Starting training...")
    results = model.train(
        data=yaml_path,
        epochs=30,         # Set epochs to 30
        imgsz=640,         # Image size
        batch=4,           # Reduced batch size to fit in RTX 3050 4GB VRAM
        device=device,     # Device selection (GPU '0' or CPU)
        project="football_player_detection",
        name="yolo26m_gpu_training"
    )
    
    print("Training completed successfully!")
    print(f"Model saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
