import boto3

sagemaker_client = boto3.client('sagemaker')

def list_endpoints(max_result=100):
    return sagemaker_client.list_endpoints(MaxResults=max_result)

if __name__ == '__main__':
    # r = list_endpoints(100)
    # for endpoint in r['Endpoints']:
    #     print(endpoint)
    # print(r)
    pass