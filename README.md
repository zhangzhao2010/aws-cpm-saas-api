# aws-cpm-saas-api
SaaS API for CPM-Bee inference and SFT on AWS SageMaker

# 部署步骤

## 1. 创建相关S3桶，上传模型资源到S3

## 2. 创建sagemaker推理镜像
1. 新建EC2，推荐使用ubuntu镜像，给EC2增加S3 ECR的full_access权限
2. 在ECR中新建仓库，名称为`cpm-inference`
2. 进入`./inference-docker`文件夹，按照ECR仓库中`push commands`来生成镜像和push镜像到仓库

## 3. 创建lambda镜像并部署
1. 进入lambda文件夹，根据模板文件config.py.template新建config.py配置文件，并修改相关配置
2. 新建ECR仓库cpm-lambda，按照ECR仓库中`push commands`来生成镜像和push镜像到仓库
3. 镜像方式部署lambda

## 4. 新建API Gateway
按照API文件增加路由，并集成lambda