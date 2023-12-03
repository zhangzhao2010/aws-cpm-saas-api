import json
import boto3

sagemaker_runtime = boto3.client("sagemaker-runtime")

def invoke_inference_endpoint(endpoint_name, input, max_new_tokens=4096):
    input_dict = json.loads(input)
    input = { 
            "inputs" : [input_dict], 
            "parameters": {"max_new_tokens": max_new_tokens, "repetition_penalty": 1.1, "temperature": 0.5}
        }

    response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                Body=bytes(json.dumps(input), 'utf-8'),
                ContentType='application/json',
                Accept='application/json'
            )
    response_json = json.load(response['Body'])
    for data in response_json['data']:
        return data

if __name__ == '__main__':
    # r = invoke_inference_endpoint(
    #     'endpoint-109', 
    #     '{"input": "知乎，是一个中文互联网高质量问答社区和创作者聚集的原创内容平台，于2011年1月正式上线，以“让人们更好地分享知识、经验和见解，找到自己的解答”为品牌使命。知乎凭借认真、专业、友善的社区氛围、独特的产品机制以及结构化和易获得的优质内容，聚集了中文互联网科技、商业、影视、时尚、文化领域最具创造力的人群，已成为综合性、全品类、在诸多领域具有关键影响力的知识分享社区和创作者聚集的原创内容平台，建立起了以社区驱动的内容变现商业模式。 [1]知乎，2017年11月8日入选时代影响力·中国商业案例TOP30。 [2] 2019年10月21日，胡润研究院发布《2019胡润全球独角兽榜》，知乎排名第138位。 [3] 6月6日原有的“知识市场”业务升级为“知乎大学”。 [4] 7月，知乎完成新一轮融资，融资额接近3亿美元。 [5] 2019年8月12日，知乎宣布完成F轮融资，总额4.34亿美元。 [6]截至2020年12月，知乎上的总问题数超过4400万条，总回答数超过2.4亿条。在付费内容领域，知乎月活跃付费用户数已超过250万，总内容数超过300万，年访问人次超过30亿。 [7]知乎以问答业务为基础，经过近十年的发展，已经承载为综合性内容平台，覆盖“问答”社区、全新会员服务体系“盐选会员”、机构号、热榜等一系列产品和服务，并建立了包括图文、音频、视频在内的多元媒介形式。 [8]2022年4月22日，知乎在港交所实现双重上市 [45] 。", "prompt": "问答", "question": "知乎上线几年了？", "<ans>": ""}'
    # )
    # print(r)
    pass