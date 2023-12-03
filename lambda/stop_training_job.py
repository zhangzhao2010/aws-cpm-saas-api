import boto3

client = boto3.client('sagemaker')

def stop_training_job(traing_job_name):
    return client.stop_training_job(TrainingJobName=traing_job_name)

if __name__ == '__main__':
    pass