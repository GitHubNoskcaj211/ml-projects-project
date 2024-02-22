from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def get_recs(request):
    return Response({'message': 'Here are your games'})