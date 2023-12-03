#!/bin/bash
rm -rf ./output

mkdir output
cp ./*.py ./output/
cp -r ./models ./output/
cp -r ./finetune ./output/

cd output
mkdir package

pip install -r ../requirements.txt --target ./package

cd package

find . -type d -name "tests" -exec rm -rfv {} +
find . -type d -name "__pycache__" -exec rm -rfv {} +

zip -r ../deployment_package.zip .

cd ..
zip deployment_package.zip *.py
zip -r deployment_package.zip ./models
zip -r deployment_package.zip ./finetune