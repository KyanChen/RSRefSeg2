import os.path
from huggingface_hub import snapshot_download
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility

YOUR_ACCESS_TOKEN = '请从https://modelscope.cn/my/myaccesstoken 获取SDK令牌'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)


model_names = [
	'facebook/sam2.1-hiera-tiny',
	'facebook/sam2.1-hiera-small',
	'facebook/sam2.1-hiera-base-plus',
	'facebook/sam2.1-hiera-large']

for model_name in model_names:
	owner_name = 'KyanChen'
	model_name = os.path.basename(model_name)
	model_id = f"{owner_name}/{model_name}"

	api.create_model(
		model_id,
		visibility=ModelVisibility.PUBLIC,
		license=Licenses.APACHE_V2,
		chinese_name=model_name
	)


	local_dir = f'work_dirs/{os.path.basename(model_name)}'
	os.makedirs(local_dir, exist_ok=True)
	path = snapshot_download(model_name, local_dir=local_dir, local_dir_use_symlinks=False)
	print(path)

	api.upload_folder(
		repo_id=f"{owner_name}/{model_name}",
		folder_path=path,
		commit_message='upload model folder to repo',
	)
