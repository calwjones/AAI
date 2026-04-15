import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import io
import base64

class Explainer:
    """Generates Grad-CAM explanations for image classification predictions"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
    
    def generate_gradcam(self, image_bytes, last_conv_layer_name="auto"):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            image_bytes: Raw image bytes
            last_conv_layer_name: Name of last conv layer (auto-detects if "auto")
        
        Returns:
            dict with heatmap_base64, explanation text, confidence
        """
        # if not self.model_manager.is_loaded():
        #     return {"error": "No model loaded"}
        
        # model = self.model_manager.get_model()

            # Use fallback model for development if no model loaded
        if not self.model_manager.is_loaded():
            model = self._get_fallback_model()
            print("⚠️  Using fallback MobileNetV2 for development")
        else:
            model = self.model_manager.get_model()
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))  # Adjust based on model's input size
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Auto-detect last conv layer if needed
        if last_conv_layer_name == "auto":
            last_conv_layer_name = self._find_last_conv_layer(model)
        
        # Create gradient model
        grad_model = keras.models.Model(
            inputs=model.input,
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        
        # Get gradients of the predicted class with respect to conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        for i in range(len(pooled_grads)):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        # Create heatmap
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(heatmap, (img.width, img.height))
        
        # Convert to colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Overlay on original image
        img_array_display = np.array(img)
        superimposed = cv2.addWeighted(img_array_display, 0.6, heatmap, 0.4, 0)
        
        # Convert to base64 for JSON response
        _, buffer = cv2.imencode('.png', superimposed)
        heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Generate explanation text
        pred_index_int = int(pred_index)
        confidence = float(predictions[0][pred_index])

        # Get predicted class name
        predicted_class = self._get_class_name(model, pred_index_int, predictions)

        explanation = self._generate_explanation_text(heatmap, confidence, predicted_class)

        return {
            "heatmap_base64": heatmap_base64,
            "explanation": explanation,
            "confidence": confidence,
            "layer_used": last_conv_layer_name,
            "predicted_class": predicted_class,  # ADD THIS
            "prediction_index": pred_index_int
        }
    
    def _find_last_conv_layer(self, model):
        """Find the last convolutional layer in the model"""
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                return layer.name
        return model.layers[-2].name  # Fallback
    
    def _generate_explanation_text(self, heatmap, confidence, predicted_class):
        """Generate human-readable explanation from heatmap"""
        # Find regions of high activation
        high_activation = np.where(heatmap > 0.7 * np.max(heatmap))
        
        if len(high_activation[0]) > 0:
            # Determine which region
            avg_y = np.mean(high_activation[0])
            avg_x = np.mean(high_activation[1])
            height, width = heatmap.shape[:2]
            
            if avg_y < height * 0.33:
                region = "top"
            elif avg_y > height * 0.66:
                region = "bottom"
            else:
                region = "middle"
            
            if avg_x < width * 0.33:
                region += "-left"
            elif avg_x > width * 0.66:
                region += "-right"
            else:
                region += "-center"
            
            return f"Predicted as '{predicted_class}' with {confidence:.1%} confidence. Model focused on {region} region."
        else:
            return f"Predicted as '{predicted_class}' with {confidence:.1%} confidence based on overall image features."
        
    def _get_fallback_model(self):
        """Load MobileNetV2 as fallback for development"""
        from tensorflow.keras.applications import MobileNetV2
        
        model = MobileNetV2(
            weights='imagenet',
            include_top=True,
            input_shape=(224, 224, 3)
        )
        
        return model
    def _get_class_name(self, model, pred_index, predictions):
        """Get human-readable class name for prediction"""
        
        # Check if this is MobileNetV2 (ImageNet) by checking output shape
        if predictions.shape[-1] == 1000:  # ImageNet has 1000 classes
            try:
                from tensorflow.keras.applications.mobilenet_v2 import decode_predictions
                # Use the actual predictions
                decoded = decode_predictions(predictions, top=1)[0][0]
                # decoded format: ('n02834778', 'banana', 0.9362)
                return decoded[1]  # Returns 'banana'
            except Exception as e:
                print(f"Could not decode prediction: {e}")
                return f"ImageNet_Class_{pred_index}"
        
        # For Person 2's custom model (when integrated)
        # Will have 2 classes: ['fresh', 'rotten']
        class_labels = {
            0: "Fresh",
            1: "Rotten"
        }
        return class_labels.get(pred_index, f"Class_{pred_index}")
