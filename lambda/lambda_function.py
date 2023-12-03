import json
import time
import datetime

import glovar
from create_inference_endpoint import create_inference_endpoint
from describe_inference_endpoint import describe_inference_endpoint
from delete_inference_endpoint import delete_inference_endpoint
from invoke_inference_endpoint import invoke_inference_endpoint
from create_training_job import create_training_job
from stop_training_job import stop_training_job
from describe_training_job import describe_training_job
from list_endpoints import list_endpoints
import inference_endpoint
import logger
import logging

log_obj = logger.Logger('', logging.INFO, logging.INFO)

def get_response_body(error_no=0, error_msg='', data={}):
    return {
        'error_no': error_no,
        'error_msg': error_msg,
        'data': data
    }

def get_response(error_no, error_msg, data = {}):
    response_body = get_response_body(error_no, error_msg, data)
    return {
        'statusCode': 200,
        'headers':{
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
        'body': json.dumps(response_body),
    }

# handler
def lambda_handler(event, context):
    if context != None:
        glovar.aws_request_id = context.aws_request_id
    
    if 'path' not in event:
        log_obj.warn('no path found')
        return get_response(1, 'no path found')
    
    try:
        path = event['path']
        action = path.lstrip('/')
        if action == 'create_inference_endpoint':
            data = json.loads(event['body'])
            model_name = data.get('model_name', '')
            endpoint_name = data.get('endpoint_name', '')
            delta_path = data.get('delta_path', '')
            create_inference_endpoint(model_name, endpoint_name, delta_path)
            endpoint_info = describe_inference_endpoint(endpoint_name=endpoint_name)
            return get_response(0, '', {'endpoint_name': endpoint_name, 'endpoint_status': endpoint_info['EndpointStatus']})
        elif action == 'invoke_inference_endpoint':
            data = json.loads(event['body'])
            endpoint_name = data.get('endpoint_name', '')
            input = data.get('input', '')
            result = invoke_inference_endpoint(endpoint_name=endpoint_name, input=input)
            return get_response(0, '', {'result': json.dumps(result)})
        elif action == 'describe_inference_endpoint':
            endpoint_name = event['queryStringParameters']['endpoint_name']
            try:
                endpoint_info = describe_inference_endpoint(endpoint_name=endpoint_name)
            except Exception as e:
                endpoint_list = list_endpoints(100)
                ed_exsit = False
                for ed in endpoint_list['Endpoints']:
                    if ed['EndpointName'] == endpoint_name:
                        endpoint_info = describe_inference_endpoint(endpoint_name=endpoint_name)
                        ed_exsit = True
                        break
                if not ed_exsit:
                    log_obj.warn('endpoint not found')
                    return get_response(1001, 'endpoint not found', {
                        'endpoint_name': endpoint_name,
                        'endpoint_status': 'NotExsit',
                        'failure_reason': 'endpoint not exsit',
                        'creation_time': '',
                    })
            
            return get_response(0, '', {
                'endpoint_name': endpoint_name,
                'endpoint_status': endpoint_info['EndpointStatus'],
                'failure_reason': endpoint_info['FailureReason'] if 'FailureReason' in endpoint_info else '',
                'creation_time': str(endpoint_info['CreationTime']),
            })
        elif action == 'delete_inference_endpoint':
            data = json.loads(event['body'])
            endpoint_name = data.get('endpoint_name', '')
            try:
                delete_inference_endpoint(endpoint_name=endpoint_name)
                return get_response(0, '', {})
            except Exception as e:
                log_obj.warn(str(e))
                eb_exist = inference_endpoint.exist_endpoint(endpoint_name=endpoint_name)
                if eb_exist:
                    delete_inference_endpoint(endpoint_name=endpoint_name)
                    return get_response(0, '', {})
                else:
                    return get_response(0, '', {})
            
        elif action == 'create_training_job':
            data = json.loads(event['body'])
            model_name = data.get('model_name', '')
            dataset_path = data.get('dataset_path', '')
            eval_dataset_path = data.get('eval_dataset_path', '')
            job_name, output_path = create_training_job(model_name=model_name, dataset_path=dataset_path, eval_dataset_path=eval_dataset_path)
            return get_response(0, '', {
                'job_name': job_name,
                'output_path': output_path,
            })
        elif action == 'stop_training_job':
            data = json.loads(event['body'])
            training_job_name = data.get('job_name', '')
            stop_training_job(traing_job_name=training_job_name)
            return get_response(0, '', {})
        elif action == 'describe_training_job':
            training_job_name = event['queryStringParameters']['job_name']
            job = describe_training_job(traing_job_name=training_job_name)
            return get_response(0, '', {
                'training_job_name': training_job_name,
                'job_status': job['TrainingJobStatus'],
                'job_secondary_status': job['SecondaryStatus'],
                'failure_reason': job['FailureReason'] if 'FailureReason' in job else '',
                'creation_time': str(job['CreationTime']),
            })
        else:
            raise Exception('wrong path')
    except Exception as e:
        return get_response(1, str(e))

if __name__ == '__main__':
    # print(lambda_handler(
    #     {
    #         'path': '/delete_inference_endpoint', 
    #         'queryStringParameters':{'endpoint_name': 'cpm-bee-finetune-default-2023-08-29-07-18-00-985'}, 
    #         'body':'{"endpoint_name":"cpm-bee-111", "model_name": "cpm-bee-hackthon","dataset_path": "s3://sagemaker-us-west-2-096331270838/cpm-bee/dataset/9c98b8284b0811eeba410242ac120003/test.txt","eval_dataset_path":"s3://sagemaker-us-west-2-096331270838/cpm-bee/dataset/9c9886984b0811eeba410242ac120003/test.txt"}'
    #     }, 
    #     None)
    # )
    pass