import boto3

client = boto3.client('sagemaker')

def describe_training_job(traing_job_name):
    return client.describe_training_job(TrainingJobName=traing_job_name)

if __name__ == '__main__':
    # r = describe_training_job('cpm-bee-finetune-default-2023-08-29-07-18-00-985')
    # print(r)
    pass