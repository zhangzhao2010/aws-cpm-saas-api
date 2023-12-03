def parse_s3_url(s3_url):
    s3_url = s3_url.replace("s3://", "")
    bucket_name, object_key = s3_url.split("/", 1)
    return bucket_name, object_key

