import requests
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time

class YOLOAPITester:
    def __init__(self, api_url="http://localhost:8080"):
        self.api_url = api_url
        
    def test_single_image(self, image_path):
        """Test API with a single image and visualize results"""
        # Time the request
        start_time = time.time()
        
        # Send request
        files = {"file": open(image_path, "rb")}
        response = requests.post(f"{self.api_url}/predict", files=files)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            # Load and display the image
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions
            predictions = response.json()["predictions"]
            self._draw_predictions(img, predictions)
            
            # Print results
            print(f"Request completed in {elapsed_time:.2f} seconds")
            print(f"Found {len(predictions)} objects")
            for pred in predictions:
                print(f"Class: {pred['class_name']}, Confidence: {pred['confidence']:.2f}")
            
            return predictions
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    
    def test_performance(self, image_path, num_requests=10):
        """Test API performance with multiple requests"""
        times = []
        files = {"file": open(image_path, "rb")}
        
        print(f"Testing performance with {num_requests} requests...")
        
        for i in range(num_requests):
            start_time = time.time()
            response = requests.post(f"{self.api_url}/predict", files=files)
            print(response.json())
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            
            print(f"Request {i+1}: {elapsed_time:.2f} seconds")
        
        avg_time = sum(times) / len(times)
        print(f"\nAverage response time: {avg_time:.2f} seconds")
        print(f"Min time: {min(times):.2f} seconds")
        print(f"Max time: {max(times):.2f} seconds")
    
    def _draw_predictions(self, image, predictions):
        """Helper function to draw bounding boxes and labels"""
        for pred in predictions:
            bbox = pred["bbox"]
            label = f"{pred['class_name']} {pred['confidence']:.2f}"
            
            # Draw rectangle
            cv2.rectangle(image,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        (0, 255, 0),
                        2)
            
            # Draw label
            cv2.putText(image,
                       label,
                       (int(bbox[0]), int(bbox[1] - 10)),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       (0, 255, 0),
                       2)
        
        # Display the image
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

def main():
    # Initialize tester
    tester = YOLOAPITester()
    
    # Test with a single image
    image_path = "u3.jpg"  # Replace with your image path
    
    print("Testing single image prediction...")
    predictions = tester.test_single_image(image_path)
    
    print("\nTesting API performance...")
    tester.test_performance(image_path, num_requests=5)

if __name__ == "__main__":
    main() 