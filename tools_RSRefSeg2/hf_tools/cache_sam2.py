from sam2.build_sam import build_sam2_hf

model_names = [
	'facebook/sam2.1-hiera-tiny',
	'facebook/sam2.1-hiera-small',
	'facebook/sam2.1-hiera-base-plus',
	'facebook/sam2.1-hiera-large']
cache_dir = 'mode_cache'
for model_name in model_names:
	sam_model = build_sam2_hf(model_name, cache_dir=cache_dir)

