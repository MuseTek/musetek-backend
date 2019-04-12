"""
    code for interaction between model and API
"""
import os
from django.core.files.storage import default_storage
from django.core.files.storage import FileSystemStorage
from django.core.files.base import ContentFile

from model.predict_tag import get_tags


def get_tags_for_sample(sample):
    """
    takes in a sample and returns the associated tags
    """

    # save file in uploaded_samples folder
    file_name = sample.__str__()
    fs = FileSystemStorage(location='uploaded_samples/')
    actual_file_name = fs.save(file_name, ContentFile(sample.read()))
    file_path = fs.path(actual_file_name)

    print(actual_file_name)
    print(file_path)

    # use model to get tags
    try:
        tags = get_tags('model/weights.hdf5', 6, 44100, True, 'model/label_encoder.p', file_path)
    except Exception as e:
        print(e)
        tags = []
    finally:
        # clean up
        fs.delete(actual_file_name)

    

    return tags


