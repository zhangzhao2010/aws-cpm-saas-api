import os
import shutil
import boto3
import tarfile
import sagemaker
from sagemaker.model import Model
import jinja2
from pathlib import Path

from model_config import model_config
import config
import glovar
import utils
import logger

log_obj = logger.Logger()

# aws service client
s3_client = boto3.client("s3")

# deploy model to endpoint
def deploy_model(image_uri, model_data, role, endpoint_name, instance_type, sagemaker_session):
    """Helper function to create the SageMaker Endpoint resources and return a predictor"""
    model = Model(
                image_uri=image_uri, 
                model_data=model_data, 
                role=role
            )
    
    return model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name,
        container_startup_health_check_timeout=60*15,
        wait=False
    )

def prepare_model_package(model_info, delta_path):
    # 1. prepare model package
    config_model_path = f"{os.path.dirname(__file__)}/models/{model_info['local_path']}/"
    tmp_model_path = f"/tmp/{glovar.aws_request_id}/model/"

    if not os.path.exists(tmp_model_path):
        os.makedirs(tmp_model_path)

    for item in os.listdir(config_model_path):
        source_item = os.path.join(config_model_path, item)
        destination_item = os.path.join(tmp_model_path, item)

        if os.path.isfile(source_item):
            shutil.copy2(source_item, destination_item)
        elif os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)
    
    jinja_env = jinja2.Environment()
    template = jinja_env.from_string(Path(f"{tmp_model_path}serving.template").open().read())
    Path(f"{tmp_model_path}/serving.properties").open("w").write(template.render(s3url=model_info['s3_uri']))

    # 2. download delta
    if delta_path != '':
        delta_bucket, delta_key = utils.parse_s3_url(delta_path)
        print(delta_bucket, delta_key, f"{tmp_model_path}delta.pt")
        s3_client.download_file(delta_bucket, delta_key, f"{tmp_model_path}delta.pt")
    
    # 3. package
    tar_file_path = f"/tmp/{glovar.aws_request_id}/model.tar.gz"
    with tarfile.open(tar_file_path, 'w:gz') as tar:
        tar.add(tmp_model_path, arcname=os.path.basename(tmp_model_path))

    # 4. upload to s3
    deploy_s3_bucket = config.system_config['deploy_s3_bucket']
    deploy_s3_key = f"{config.system_config['deploy_s3_path_prefix']}{glovar.aws_request_id}/model.tar.gz"
    s3_client.upload_file(tar_file_path, deploy_s3_bucket, deploy_s3_key)

    return f"s3://{deploy_s3_bucket}/{deploy_s3_key}"

def create_inference_endpoint(model_name, endpoint_name, delta_path):
    if not model_name in model_config.keys():
        log_obj.warning(f'Invalid model_name: {model_name}')
        raise Exception('Invalid model_name')
    
    # model package data s3 uri
    model_package_uri = prepare_model_package(
                        model_info=model_config[model_name], 
                        delta_path=delta_path
                    )
    
    log_obj.info(f"model_package_uri: {model_package_uri}")

    # deploy model
    role = config.system_config['sagemaker_role']  # execution role for the endpoint
    sess = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs

    region = sess._region_name # region name of the current SageMaker Studio environment
    instance_type = model_config[model_name]['instance_type']
    djl_inference_image_uri = config.system_config['inference_image_uri']

    log_obj.info(f"region: {region}")
    log_obj.info(f"image_uri: {djl_inference_image_uri}")
    log_obj.info(f"role: {role}")

    deploy_model(
        image_uri=djl_inference_image_uri,
        model_data=model_package_uri,
        role=role,
        endpoint_name=endpoint_name,
        instance_type=instance_type,
        sagemaker_session=sess
    )

    return endpoint_name

if __name__ == '__main__':
    # create_inference_endpoint(
    #     model_name="cpm-bee-hackthon",
    #     endpoint_name="endpoint-119",
    #     delta_path="s3://sagemaker-us-west-2-096331270838/cpm-bee/finetune/output/default/results/cpm_bee_finetune-delta-best-1.pt",
    # )
    pass