import boto3

sagemaker_client = boto3.client('sagemaker')

def delete_inference_endpoint(endpoint_name):
    sagemaker_client.delete_endpoint(EndpointName=endpoint_name)

if __name__ == '__main__':
    # r = delete_inference_endpoint('endpoint-006')
    # print(r)
    pass