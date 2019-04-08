from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import parser_classes
from rest_framework.parsers import FileUploadParser
from rest_framework.exceptions import ParseError
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from api.utils import get_tags

class GetTags(APIView):
    """
    generate tags for the given sample

    converts sample into spectrogram and runs data through a model to provide tags
    """

    parser_classes = (FileUploadParser,)

    def put(self, request, format=None):

        if 'file' not in request.data:
            raise ParseError("Empty content")

        f = request.data['file']

        tags = get_tags(f)
        return Response(status=200, data={'file': f.__str__(), 'tags': tags})