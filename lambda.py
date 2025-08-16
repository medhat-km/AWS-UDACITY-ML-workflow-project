####################LAMBDA01####################

import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event['s3_key']## TODO: fill in
    bucket = event['s3_bucket']## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    s3.download_file(bucket, key, "/tmp/image.png")
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event keys:", event.keys())
    print(f"The file to be serialized: {key.split('/')[-1]}")
    print(f"{'Serialization: Done.' if isinstance(image_data, bytes) else 'Serialization: Failed.'}")

    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

####################LAMBDA02####################

import json
import base64
import boto3

# My last modified endpoint
ENDPOINT = "image-classification-2025-08-04-22-02-53-625"

# SageMaker runtime client (available by default in Lambda)
runtime = boto3.client('sagemaker-runtime')


def lambda_handler(event, context):
    """ Lambda function that calls an existing SageMaker endpoint using boto3 (without SageMaker SDK). """
    # Decode the image data
    image = base64.b64decode(event["body"]["image_data"])

    # Invoke the endpoint
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="image/png", Body=image)

    # Get the prediction result
    result = response["Body"].read().decode("utf-8")
    event["body"]["inferences"] = result
    
    # Predictions as list & getting argmax
    result_list = json.loads(result)
    max_index = result_list.index(max(result_list))
    
    # Mainly for testing my lambda
    print(f"The predicted propabilities: {result}")
    print(f"The predicted label: {[1, 0][max_index]}")
    print(f"The predicted class: {["Bicycle", "Motorcycle"][max_index]}")
    
    
    return {
        "statusCode": 200,
        "body": json.dumps(event)
    }
  


####################LAMBDA03####################

import json


THRESHOLD = 0.75


def lambda_handler(event, context):
    """ A function to filter out the inferences based upon a predefined threshold """
    
    # Grab the inferences from the event
    inferences = json.loads(event["body"]["inferences"])
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = max(inferences) >= THRESHOLD
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
