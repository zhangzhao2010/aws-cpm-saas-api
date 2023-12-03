import boto3

sagemaker_client = boto3.client('sagemaker')

def describe_inference_endpoint(endpoint_name):
    return sagemaker_client.describe_endpoint(EndpointName=endpoint_name)

if __name__ == '__main__':
    # r = describe_inference_endpoint('endpoint-002')
    # print(r)
    pass