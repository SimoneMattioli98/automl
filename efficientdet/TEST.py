import model_inspect_custom as mic

inspector = mic.Inspector()

name = "efficientdet-d0"
ckpt_path = "efficientdet-d0"
saved_model_dir = "savedmodel"

d0_inspector = inspector.get_model_inspector(name, ckpt_path, saved_model_dir)

mode = "saved_model"

# d0_inspector.export_saved_model() # TO DO ONLY THE FIRST TIME 

model_driver = d0_inspector.get_model_driver()

image_path = "testdata/img1.jpg"
output_dir = "output_dir"
detections = d0_inspector.saved_model_inference(image_path, output_dir, model_driver)

print(detections)

