"""
function code between model and API
"""
import os
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile

from model import predict_tag


def get_tags(sample):
    """
    takes in a sample and returns the associated tags
    """
    file_name = sample.__str__()
    fs = FileSystemStorage(location='uploaded_samples/')
    file_path = fs.save(file_name, ContentFile(sample.read()))

    print(file_name)
    print(file_path)
    
    return ['this', 'is', 'a', 'test']


