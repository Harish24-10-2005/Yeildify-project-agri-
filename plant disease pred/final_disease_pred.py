import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models

l_classes = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy', 'Blueberry_healthy', 'Cherry_Powdery_mildew', 'Cherry_healthy', 'Corn_Cercospora_leaf_spot Gray_leaf_spot', 'Corn_Common_rust', 'Corn_Northern_Leaf_Blight', 'Corn_healthy', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)', 'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach__Bacterial_spot', 'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_Late_blight', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_Leaf_scorch', 'Strawberry_healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato_Tomato_mosaic_virus', 'Tomato__healthy']
num_classes = len(l_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = models.densenet121(pretrained=False)  
loaded_model.classifier = nn.Linear(loaded_model.classifier.in_features, num_classes)  
loaded_model.load_state_dict(torch.load('plant_disease_model.pth'))
loaded_model = loaded_model.to(device)
loaded_model.eval()

from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

image_path = "test_dataset/Corn___Common_rust/image (102).JPG"
image = Image.open(image_path)

input_image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension and move to device

with torch.no_grad():
    output = loaded_model(input_image)
    _, predicted_class = torch.max(output, 1)
predicted_label = l_classes[predicted_class.item()]
healthy = ['Apple_healthy', 'Blueberry_healthy', 'Cherry_healthy', 'Corn_healthy', 'Grape__healthy', 'Peach_healthy', 'Pepper,_bell_healthy', 'Potato_healthy', 'Raspberry_healthy', 'Soybean_healthy', 'Strawberry_healthy', 'Tomato__healthy']
message = """
                Black rot, caused by the fungus Botryosphaeria obtusa, is a serious disease affecting apple trees. While it's challenging to completely cure an established case of black rot, you can manage and prevent its spread using various strategies. Here are some approaches:
                
                1. Pruning: Prune infected branches and remove all diseased fruit from the tree. This helps prevent the spread of the fungus to healthy parts of the tree.
                
                2. Sanitation: Clean up fallen leaves, fruit, and any other debris around the tree. This reduces the number of fungal spores that can overwinter and reinfect the tree in the following growing season.
                
                3. Chemical Control: Apply fungicides labeled for black rot control. Copper-based fungicides and fungicides containing captan or thiophanate-methyl are commonly used for managing black rot. Follow the manufacturer's instructions for application rates and timing.
                
                4. Resistant Varieties: Planting apple tree varieties that are resistant or less susceptible to black rot can help reduce the severity of the disease. Consult with local nurseries or extension services for recommendations on resistant apple varieties.
                
                5. Pruning and Air Circulation: Proper pruning to improve air circulation within the canopy can help reduce humidity levels, which in turn discourages fungal growth. This includes thinning out dense foliage and ensuring adequate spacing between branches.
                
                6. Fruit Bagging: Bagging developing fruit with breathable covers can protect them from fungal infection. This is especially useful in orchards with a history of black rot.
                
                7. Integrated Pest Management (IPM): Implement an integrated approach to managing pests and diseases in the orchard. This includes cultural practices, chemical treatments, and biological controls tailored to the specific conditions of the orchard.
                
                8. Monitoring: Regularly inspect apple trees for symptoms of black rot, such as dark, sunken lesions on fruit and leaves. Early detection allows for timely intervention and better disease management.
                
                9. Consultation with Experts: If black rot persists despite your efforts, seek advice from local agricultural extension services or plant pathology experts. They can provide personalized recommendations based on your specific situation and local conditions.
            """
print("Predicted disease:", predicted_label)
if predicted_label in healthy:
    print("your plant is healthy")
else:
    import json
    
    with open('disease_info.json', 'r') as json_file:
        disease_info = json.load(json_file)
    
    def print_disease_details(disease_name):
        if disease_name in disease_info:
            disease_details = disease_info[disease_name]
            description = disease_details.get('Description', 'Description not available.')
            management_strategies = disease_details.get('Management Strategies', {})
            if description == 'Description not available.' and not management_strategies:
                print(f"No information available for '{disease_name}'. Please consult with a plant pathology expert.")
            else:
                print("Description:", description)
                print("Management Strategies:")
                for strategy, details in management_strategies.items():
                    print(f"\n{strategy}:")
                    for item in details:
                        print("-", item)
        else:
            print(f"Disease '{disease_name}' not found. Please check the spelling or consult with a plant pathology expert.")


    if predicted_label =='Apple_Black_rot':
        print(message)
    else:
        print_disease_details(predicted_label)


