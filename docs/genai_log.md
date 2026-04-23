# GenAI Usage Log

**Advanced AI (UFCFUR-15-3) — BRFN Group Project**

*Tool used: Claude (Anthropic) unless stated otherwise*

---

## Log

| Date | Who | What we asked | What it gave us | Did it work / what we changed |
|---|---|---|---|---|
| 09/04/26 | Charles | Asked about best CNN architecture for classifying fruit/veg images with limited training time | Suggested MobileNetV2 with transfer learning over VGG16 due to smaller size and faster training, while still using residual connections | Useful starting point. We compared this against what was covered in our transfer learning lecture (VGG16) and chose MobileNetV2 based on the speed advantage for our timeline |
| 12/04/26 | Charles | Debugging: model evaluation showing 53% accuracy after reloading notebook in Colab | Identified that the wrong model file was being loaded (phase 1 instead of phase 2). Suggested explicitly loading best_model_phase2.keras | Correct diagnosis. Saved hours of confusion. The actual model had 99.7% accuracy. It was purely a loading issue |
| 13/04/26 | Group | Asked about how to generate realistic synthetic order data that matches our DESD database structure | Generated a Python script producing 25k order rows across 250 customers and 60 products with repeat ordering patterns and seasonal variation | The data structure was correct but we found overlapping customer/producer IDs which would cause issues in Django. Had to fix the ID ranges manually. Also reviewed the output to make sure ordering patterns looked realistic |
| 13/04/26 | Tommy | Asked for help structuring the order prediction notebook with temporal train/test split | Provided a feature engineering approach using cutoff date to avoid data leakage | First attempt had a data leakage bug where the target was derived from a training feature (order_count > 1). AI helped identify the issue and suggested the temporal split approach which we implemented |
| 14/04/26 | Callum | Asked about structuring FastAPI service to handle both .keras and .pkl model uploads | Suggested a ModelManager class with file extension detection and appropriate loading (keras.load_model vs pickle.load) | Worked well as a starting point. We extended it to add version tracking and metadata storage which wasn't in the original suggestion |
| 15/04/26 | Aaron | Asked how to implement Grad-CAM for MobileNetV2 to show what the model focuses on | Provided code for extracting gradients from the last convolutional layer and generating heatmap overlays | Required significant debugging to work with our specific model architecture. The layer names didn't match and we had to handle the nested base model structure. Took several iterations to get working |
| 16/04/26 | Group | Asked for a Django management command to load our synthetic AI training data into the DESD marketplace so the AI services could be tested against a populated database end-to-end | Generated a seed_orders.py script that creates users, products, orders and order items from the CSV | Missed the ProducerProfile model initially, had to add that ourselves. Also needed to handle the auto_now_add field on Order.created_at which required a workaround with .update() |

---

## Overall Evaluation

GenAI was most useful for debugging issues (model loading, data leakage identification) and for generating boilerplate code that we then customised. It was less useful for domain-specific decisions like tuning the quality scoring thresholds and choosing augmentation parameters. Those required our own judgement based on the dataset characteristics and the module content.

Key lesson: GenAI output always needed verification. The synthetic data had ID overlap bugs, the first order prediction approach had data leakage, and the Grad-CAM code needed layer name fixes. Using it effectively meant treating it as a starting point rather than a finished solution.