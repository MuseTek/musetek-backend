from api import views
from django.urls import path

urlpatterns = [
    path('get-tags/', views.GetTags.as_view())
]
