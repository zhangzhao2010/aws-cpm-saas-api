import boto3

from sagemaker.pytorch.estimator import PyTorch
from model_config import model_config
from config import system_config

import glovar

s3_client = boto3.client('s3')

def create_training_job(model_name, dataset_path, eval_dataset_path, instance_type = 'ml.g5.12xlarge'):
    if model_name not in model_config:
        raise Exception('Invalid model_name')
    
    model_info = model_config[model_name]
    finetune_output_bucket = system_config['finetune_output_bucket']
    finetune_output_s3_path_prefix = f"{system_config['finetune_output_s3_path_prefix']}{glovar.aws_request_id}/"
    default_s3_bucket = system_config['default_s3_bucket']
    role = system_config['sagemaker_role']

    environment = {
        'MODEL_S3_URI': model_info['s3_uri'],
        "TRAIN_DATASET_S3_URI": dataset_path,
        "EVAL_DATASET_S3_URI": eval_dataset_path,
        'MODEL_S3_BUCKET': default_s3_bucket, # The bucket to store pretrained model and fine-tune model
        'FINETUNE_OUTPUT_BUCKET': finetune_output_bucket,
        'FINETUNE_OUTPUT_S3_PATH_PREFIX': finetune_output_s3_path_prefix,
        'PIP_CACHE_DIR': "/opt/ml/sagemaker/warmpoolcache/pip"
    }

    base_job_name = 'cpm-bee-finetune'

    estimator = PyTorch(role=role,
                        entry_point='finetune_cpm_bee.sh',
                        source_dir='./finetune', ## 务必确保该目录下，已经删除了 ckpts目录，即删除了模型文件目录，该目录下不能有大文件
                        base_job_name=base_job_name,
                        instance_count=1,
                        instance_type=instance_type,
                        framework_version='1.13.1',
                        py_version='py39',
                        environment=environment,
                        keep_alive_period_in_seconds=15*60,
                        disable_profiler=True,
                        debugger_hook_config=False)

    estimator.fit(wait=False)
    return estimator.latest_training_job.name, f'{finetune_output_bucket}/{finetune_output_s3_path_prefix}results/cpm_bee_finetune-delta-best.pt'

if __name__ == '__main__':
    # instance_type = 'ml.g5.12xlarge'

    # r = create_training_job(
    #     'cpm-bee-10b',
    #     's3://sagemaker-us-west-2-096331270838/llm/datasets/cpm-bee/bee_data/train.jsonl',
    #     's3://sagemaker-us-west-2-096331270838/llm/datasets/cpm-bee/bee_data/eval.jsonl',
    #     instance_type = instance_type
    # )

    # print(r)
    pass