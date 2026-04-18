import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from PIL import Image
import io
import base64
from .quality_scorer import grade_image_bytes
 
class Explainer:
    """Generates Grad-CAM explanations for image classification predictions"""
    
    def __init__(self, model_manager):
        self.model_manager = model_manager
        # Load Charles's model directly
        self.custom_model = self._load_custom_model()
    
    def _load_custom_model(self):
        """Load Charles's fresh/rotten classifier"""
        try:
            model_path = 'models/best_model_phase2.keras'
            model = keras.models.load_model(model_path)
            print(f"Loaded custom model from {model_path}")
            return model
        except Exception as e:
            print(f"Could not load custom model: {e}")
            return None
    
    def generate_gradcam(self, image_bytes, last_conv_layer_name="auto"):
        """
        Generate Grad-CAM heatmap for an image
        
        Args:
            image_bytes: Raw image bytes
            last_conv_layer_name: Name of last conv layer (auto-detects if "auto")
        
        Returns:
            dict with heatmap_base64, explanation text, confidence
        """
        # Use custom model if available, otherwise fallback
        if self.custom_model is not None:
            model = self.custom_model
            print("Using custom fresh/rotten model")
        elif self.model_manager.is_loaded():
            model = self.model_manager.active_model
            print("Using ModelManager's loaded model")
        else:
            model = self._get_fallback_model()
            print("Using fallback MobileNetV2 for development")
        
        # Load and preprocess image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((224, 224))  # Adjust based on model's input size
 
        # Get quality grading from scorer
        try:
            grading_result = grade_image_bytes(image_bytes, model)
        except Exception as e:
            print(f"Grading failed: {e}")
            grading_result = None
 
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        # Find the nested MobileNetV2 base model
        base_model = model.get_layer("mobilenetv2_1.00_224")
        print(f"✅ Found nested base model: {base_model.name}")
        
        if last_conv_layer_name == "auto":
            last_conv_layer_name = "out_relu"
        
        # Build a model that exposes:
        # 1. inner conv activation
        # 2. base model output
        base_cam_model = keras.models.Model(
            inputs=base_model.input,
            outputs=[
                base_model.get_layer(last_conv_layer_name).output,
                base_model.output,
            ],
        )
        
        print(f"✅ Using inner conv layer: {last_conv_layer_name}")
        
        # Rebuild a connected graph from the outer model input
        inputs = model.input
        conv_tensor, x = base_cam_model(inputs)
        
        x = model.get_layer("global_average_pooling2d")(x)
        x = model.get_layer("dropout")(x)
        x = model.get_layer("feature_layer")(x)
        x = model.get_layer("dropout_1")(x)
        predictions_tensor = model.get_layer("classification")(x)
        
        # Build connected Grad-CAM model
        grad_model = keras.models.Model(
            inputs=inputs,
            outputs=[conv_tensor, predictions_tensor],
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array, training=False)
            
            if predictions.shape[-1] == 1:
                class_channel = predictions[:, 0]
                pred_index = tf.cast(predictions[0][0] >= 0.5, tf.int32)
            else:
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
        
        # Get gradients of the predicted class with respect to conv layer
        grads = tape.gradient(class_channel, conv_outputs)
        
        if grads is None:
            raise ValueError(
                f"Gradients are None for layer '{last_conv_layer_name}'. "
                "The selected layer may not be connected properly to the prediction output."
            )
        
        # Handle different gradient shapes (4D for CNNs, 2D for GAP models)
        if len(grads.shape) == 4:
            # Standard CNN: (batch, height, width, channels)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        elif len(grads.shape) == 2:
            # GlobalAveragePooling: (batch, channels)
            pooled_grads = tf.reduce_mean(grads, axis=0)
        else:
            raise ValueError(f"Unexpected gradient shape: {grads.shape}")
        
        # Weight conv outputs by gradients
        conv_outputs = conv_outputs[0]
        pooled_grads = pooled_grads.numpy()
        conv_outputs = conv_outputs.numpy()
        
        # Apply pooled gradients based on shape
        if len(conv_outputs.shape) == 3:
            # Standard CNN: (height, width, channels)
            for i in range(len(pooled_grads)):
                conv_outputs[:, :, i] *= pooled_grads[i]
            # Create heatmap
            heatmap = np.mean(conv_outputs, axis=-1)
        elif len(conv_outputs.shape) == 1:
            # GlobalAveragePooling: (channels,)
            weighted = conv_outputs * pooled_grads
            # Create a simple heatmap (uniform since no spatial info)
            heatmap = np.full((7, 7), np.mean(weighted))  # 7x7 placeholder
        else:
            raise ValueError(f"Unexpected conv_output shape: {conv_outputs.shape}")
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
        
        # Use prediction and confidence
        if grading_result:
            predicted_class = grading_result.get('prediction')  # "Fresh" or "Rotten"
            confidence = grading_result.get('confidence')       # Proper confidence
        else:
            # Fallback if grading failed
            pred_index_int = int(pred_index)
            confidence = float(predictions[0][pred_index])
            predicted_class = self._get_class_name(model, pred_index_int, predictions)
 
        explanation = self._generate_explanation_text(heatmap, confidence, predicted_class)
 
        return {
            "heatmap_base64": heatmap_base64,
            "explanation": explanation,
            "confidence": confidence,
            "layer_used": last_conv_layer_name,
            "predicted_class": predicted_class,
            "grade": grading_result.get('grade') if grading_result else 'N/A',
            "color_score": grading_result.get('color_score') if grading_result else None,
            "size_score": grading_result.get('size_score') if grading_result else None,
            "ripeness_score": grading_result.get('ripeness_score') if grading_result else None
        }
    
    def _find_last_conv_layer(self, model):
        """Return the known last conv layer for Charles's MobileNetV2 architecture"""
        return "out_relu"
    
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
                return decoded[1]  # Returns 'banana'
            except Exception as e:
                print(f"Could not decode prediction: {e}")
                return f"ImageNet_Class_{pred_index}"
        
        class_labels = {
            0: "Fresh",
            1: "Rotten"
        }
        return class_labels.get(pred_index, f"Class_{pred_index}")
 