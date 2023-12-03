from list_endpoints import list_endpoints

def exist_endpoint(endpoint_name):
    endpoint_list = list_endpoints(100)
    for ed in endpoint_list['Endpoints']:
        if ed['EndpointName'] == endpoint_name:
            return True
    return False