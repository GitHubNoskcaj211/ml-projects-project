from django.urls import path
from . import views

urlpatterns = [
    path('get-recs/', views.get_recs, name='get-recommendations'),
]