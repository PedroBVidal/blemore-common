import os, argparse, cv2, torch, random
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

# Fix for PyTorch 2.6+ custom object loading
from transformers.modeling_outputs import BaseModelOutputWithPooling
if hasattr(torch.serialization, 'add_safe_globals'):
	torch.serialization.add_safe_globals([BaseModelOutputWithPooling])

parser = argparse.ArgumentParser()
parser.add_argument('--data_path')
parser.add_argument('--features_path')
args = parser.parse_args()

batch_size = 2048

# Load model ONCE outside the loop for speed
print("Loading CLIP model...", flush=True)
processor_clip = AutoProcessor.from_pretrained('openai/clip-vit-base-patch32')
model_clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')

def features_path_part(features_path, p):
	return features_path.replace('.pt', '_'+str(p).zfill(3)+'.pt')

def extract_features_batch(batch):
	with torch.no_grad():
		inputs_clip = processor_clip(images=batch, return_tensors='pt')
		outputs = model_clip.get_image_features(**inputs_clip)

		# 1. Check if the output is the object (BaseModelOutputWithPooling)
		# If it is, we need the pooler_output or image_embeds
		if not isinstance(outputs, torch.Tensor):
			# Try to get the specific embedding attribute
			features = getattr(outputs, 'image_embeds', None)
			if features is None:
				# Fallback to the first element if attribute name varies
				features = outputs[0]
		else:
			features = outputs

		# CRITICAL FIX: Convert the CLIP object into a plain Tensor
		# and move to CPU to avoid pickling issues later.
		return features.detach().cpu()

# Support multiple video formats
valid_extensions = ('.mov', '.mp4', '.avi', '.mkv')
video_names = [v for v in os.listdir(args.data_path) if v.lower().endswith(valid_extensions)]
random.shuffle(video_names)

for video_name in video_names:
	video_path = os.path.join(args.data_path, video_name)
	
	# FIX: Correctly handle .mov vs .mp4 extensions
	base_name = os.path.splitext(video_name)[0]
	features_path = os.path.join(args.features_path, base_name + '.pt')

	if os.path.isfile(features_path):
		print(f'Features already exist: {features_path}', flush=True)
		continue
	
	if not os.path.isdir(os.path.dirname(features_path)):
		os.makedirs(os.path.dirname(features_path))

	print(f'Processing: {video_path}', flush=True)
	
	frames = []
	part_count = 0
	video = cv2.VideoCapture(video_path)
	total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	
	pbar = tqdm(total=total_frames)
	for i in range(total_frames):
		success, frame = video.read()
		if not success:
			break
		
		# Convert BGR (OpenCV) to RGB (PIL/CLIP)
		img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
		frames.append(img)
		
		if (i + 1) % batch_size == 0:
			features = extract_features_batch(frames)
			torch.save(features, features_path_part(features_path, part_count))
			frames = []
			part_count += 1
		pbar.update(1)
	pbar.close()

	if frames:
		features = extract_features_batch(frames)
		torch.save(features, features_path_part(features_path, part_count))
		part_count += 1
	
	video.release()

	# OPTIMIZED CONCATENATION: Load all parts and merge in memory
	print(f"Merging {part_count} parts...", flush=True)
	all_tensors = []
	for j in range(part_count):
		part_file = features_path_part(features_path, j)
		# Use weights_only=False inside the load function
		all_tensors.append(torch.load(part_file, weights_only=False))
	
	final_tensor = torch.cat(all_tensors, dim=0)
	torch.save(final_tensor, features_path)

	# Cleanup parts
	for j in range(part_count):
		os.remove(features_path_part(features_path, j))

	# TO THIS (if you still get errors):
	if torch.is_tensor(final_tensor):
		print('Done:', features_path, "Shape:", final_tensor.shape, flush=True)
	else:
		# If for some reason it's still the object:
		print('Done:', features_path, "Shape:", final_tensor[0].shape, flush=True) 